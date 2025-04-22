import os
import torch
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from utils.util import set_device, output_tour_to_csv, save_dataset, fix_seed
from generate_dataset import CIRPDataset
from models.am import AM4CIRP
from models.naive_models import NaiveModel
from models.state import visualize_routes as vis_routes
from models.state import save_route_info

#添加一个conuter函数 每次输出视频编号
def get_next_output_dir(base_dir='./output'):
    """获取下一个可用的输出目录路径"""
    counter = 1
    while True:
        output_dir = os.path.join(base_dir, f'run_{counter}')
        if not os.path.exists(output_dir):
            return output_dir
        counter += 1

def write_metrics_to_csv(metrics, output_dir, filename="metrics.csv"):
    """
    Write metrics to a CSV file.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics
    output_dir : str
        Output directory path
    filename : str, optional
        Name of the CSV file, by default "metrics.csv"
    """
    import os
    import csv
    import datetime
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the full path
    csv_path = os.path.join(output_dir, filename)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)
    
    # Add timestamp to metrics
    metrics_copy = metrics.copy()  # 创建副本以避免修改原始字典
    metrics_copy['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Define the field names (headers) aligned with eval.py's summary dictionary keys
    fieldnames = [
        'timestamp',
        # Basic metrics
        'avg_obj',
        'std_obj',
        'avg_tour_length',
        'std_tour_length',
        'avg_conflict_cost',
        'std_conflict_cost',
        'avg_actual_tour_length',
        'std_actual_tour_length',
        'total_calc_time',
        'avg_calc_time',
        'std_calc_time',
        # Service metrics
        'percent_served',
        'avg_first_response',
        # Utilization metrics
        'vehicle_utilization',
        'avg_travel_time',
        'avg_wait_time',
        'avg_station_utilization',
        # Queue metrics
        'avg_queue_length',
        'max_queue_length',
        'avg_queue_time',
        'max_queue_time',
        # Charging event metrics
        'avg_charge_events',
        'avg_charge_per_event',
        # Energy metrics
        'total_travel_energy',
        'total_charge_energy',
        #'total_supply_energy'
    ]
    
    # Add any other fields that might be in the metrics dict but not predefined
    existing_keys = list(fieldnames)  # Start with aligned keys
    for key in metrics_copy.keys():
        if key not in existing_keys:
            fieldnames.append(key)  # Append any extra keys found
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        
        # Write header if file didn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the metrics, using 'N/A' for missing keys
        row_data = {k: metrics_copy.get(k, 'N/A') for k in fieldnames}
        writer.writerow(row_data)
    
    print(f"Metrics saved to {csv_path}")

def eval(dataset_path: str,
         eval_batch_size: int = 256,
         max_load_size: int = -1,
         model_type: str = "rl",
         model_path: str = None,
         model_dir: str = None,
         decode_type: str = "sampling",
         search_width: int = 12800,
         max_batch_size: int = 128,
         #penalty_coef: float = 100,
         conflict_coef: float = 10.0,  # 替代 penalty_coef
#          - `search_width`: 搜索宽度,默认12800
# - `max_batch_size`: 最大批量大小,默认128
# - `penalty_coef`: 惩罚系数,默认100
         vehicle_speed: float = 41,
         #车辆速度(km/h),默认41
         wait_time: float = 0.5,
         time_horizon: float = 6,  # 改小一点 12-》6
         random_seed: int = 1234,
         gpu: int = -1,
         num_workers: int = 8, # 4-》8
         #- `gpu`: GPU编号,默认-1表示使用CPU
         #- `num_workers`: 并行工作进程数,默认4
         visualize_routes: bool = True,  # 是否可视化路线 改为是 #另外一种情况 复现时用到valid eval函数时关闭这个
         output_dir: str = None) -> Dict[str, Any]:
    

    #Dict[str, Any]: 返回包含评估结果的字典
    #-----------------
    # set random seed
    #-----------------
    fix_seed(random_seed)
    
    #------------------------------
    # device settings (gpu or cpu)
    #------------------------------
    use_cuda, device = set_device(gpu)

    #---------
    # dataset
    #---------
    dataset = CIRPDataset().load_from_pkl(dataset_path, load_dataopts=False, max_load_size=max_load_size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=eval_batch_size,
                                             shuffle=None,
                                             num_workers=num_workers)

    #-------
    # model
    #-------
    if model_type == "rl":
        # load a trained model
        if model_path is not None: 
            model_dir = os.path.split(model_path)[0]
        elif model_dir is not None:
            model_path = f"{model_dir}/model_bestepoch.pth"
        else:
            assert False, "specify the one from model_path and model_dir :("

        params = argparse.ArgumentParser()
        with open(f"{model_dir}/cmd_args.dat", "r") as f:
            params.__dict__ = json.load(f)
        model = AM4CIRP(loc_dim=params.loc_dim,
                        depot_dim=params.depot_dim,
                        vehicle_dim=params.vehicle_dim,
                        emb_dim=params.emb_dim,
                        num_heads=params.num_heads,
                        num_enc_layers=params.num_enc_layers,
                        dropout=params.dropout,
                        device=device)
        model.load_state_dict(torch.load(model_path))
        if use_cuda:
            model.to(device)
    elif model_type in ["naive_greedy", "naive_random", "wo_move"]:
        model = NaiveModel(model_type, device)
    else:
        raise TypeError("Invalid model_type!")

    #------------
    # evaluation
    #------------
    actual_tour_length_list = []
#     存储实际路线长度
# 考虑了真实物理距离（乘以网格比例因子后的距离）
# 单位可能是公里
    tour_length_list = []
#     存储规范化/标准化后的路线长度
# 用于计算模型内部的损失函数
# 通常是网格单位下的距离
#     down_list = []
# #     存储惩罚值
# # 记录电量耗尽(down)的基站产生的惩罚
# # 用于评估解决方案的质量
#     num_down_list = []

#     存储断电基站的数量
# 记录每个批次中电量耗尽的基站总数
# 用于统计分析
    conflict_cost_list = []
    calc_time_list = []
#     存储计算时间
# 记录每个批次处理所需的时间
# 用于评估模型的计算效率
    model.eval()
#     将模型设置为评估模式
# 关闭一些训练时才需要的功能(如 dropout, batch normalization)
# 确保评估的一致性
    for batch_id, batch in enumerate(tqdm(dataloader)):
       # 使用 tqdm 显示进度条遍历数据加载器
# batch_id: 批次编号
# batch: 当前批次的数据
        start_time = time.perf_counter()
#         记录处理每个批次的开始时间
# 用于计算计算时间
        
        if use_cuda:
            batch = {key: value.to(device) for key, value in batch.items()}
        # add options
        batch.update({
            "time_horizon": time_horizon,
            "vehicle_speed": vehicle_speed,
            "wait_time": wait_time
        })

        # output tours
        if model_type == "rl":
            if decode_type == "greedy":
                with torch.inference_mode():
#                     上下文管理器:
# 禁用梯度计算
# 减少内存使用
# 提升推理速度
                    cost_dict, vehicle_ids, node_ids, mask = model.greedy_decode(batch)
                    #返回值:
# cost_dict: 包含成本信息的字典
# vehicle_ids: 车辆ID
# node_ids: 节点ID
# mask: 掩码信息

            elif decode_type == "sampling":
                with torch.inference_mode():
#                     greedy_decode: 贪婪解码 - 每次选择最优动作
# sample_decode: 采样解码 - 从多个可能的动作中采样选择
                    cost_dict, vehicle_ids, node_ids, mask = model.sample_decode(batch, search_width, max_batch_size)
            else:
                NotImplementedError
        elif model_type == "naive_random":
            cost_dict, vehicle_ids, node_ids, mask = model.sample_decode(batch, search_width, max_batch_size)
        elif model_type in ["naive_greedy", "wo_move"]:
            cost_dict, vehicle_ids, node_ids, mask = model.decode(batch)
        else:
            NotImplementedError

        calc_time_list.append(time.perf_counter() - start_time)
        tour_length_list.append(cost_dict["tour_length"])
        # down_list.append(cost_dict["penalty"])
        # num_down_list.append(cost_dict["penalty"] * batch["loc_coords"].size(1))
        conflict_cost_list.append(cost_dict["conflict_cost"])
# 大约在行 245 左右，修改如下代码



        #loc_coords：表示基站坐标数据
# .size(1)：获取第1维度的大小，即基站的数量
# 这表示当前批次中的基站总数
# cost_dict["penalty"]
# 表示惩罚值（一个基站断电的惩罚系数）
# 通常是一个0到1之间的值，表示断电的比例
# penalty * loc_coords.size(1)
#这个penalty本来就是100
# 将惩罚值乘以基站总数
# 得到实际断电的基站数量 之后改

# 计算每个批次中电量耗尽（断电）的基站数量
# 将惩罚比例转换为实际的断电基站个数
# 保存这些数据以便后续计算平均值、标准差等统计指标
# 例如：

# 如果有100个基站(loc_coords.size(1)=100)
# 惩罚值为0.2(penalty=0.2)
# 则断电基站数量为20个(100 * 0.2 = 20)
        actual_tour_length_list.append(cost_dict["tour_length"] * batch["grid_scale"].squeeze(-1))
        # Add this new list to collect detailed metrics
        detailed_metrics_list = []
                # Check if we have access to detailed metrics and add them if available
        if "detailed_metrics" in cost_dict:
            detailed_metrics_list.append(cost_dict["detailed_metrics"])
#         batch["grid_scale"]
# 网格比例因子
# 用于将网格单位转换为实际物理距离(如公里)
# 例如：如果网格单位是1，而实际1个网格=2公里，则scale=2
# - 惩罚值反映了系统的服务质量
# - penalty=0.2 表示20%的基站断电
# - 惩罚值越低说明系统性能越好
# - 而是通过惩罚值来平衡:
#   - 路径规划效率
#   - 服务质量要求
# - 将断电情况转换为可量化的数值
        #---------------
        # visualization
        #---------------
        # if visualize_routes:
        #     if not output_dir:    # 6 set defualt output path
        #         output_dir = './output'  # 设置一个默认目录路径
        #     os.makedirs(output_dir, exist_ok=True)
        #     vis_routes(vehicle_ids, node_ids, batch, f"{output_dir}/batch{batch_id}", device)
        #     save_route_info(batch, vehicle_ids, node_ids, mask, f"{output_dir}/batch{batch_id}")
        

        if visualize_routes:
        #之后改 在最开始我加了一个编号的函数
            if not output_dir:  # 如果没有指定输出目录
                base_dir = './output'
                os.makedirs(base_dir, exist_ok=True)
                # 获取新的编号目录
                output_dir = get_next_output_dir(base_dir)
            
            # 创建本次运行的输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 可视化和保存路由信息
            vis_routes(vehicle_ids, node_ids, batch, f"{output_dir}/batch{batch_id}", device)
            save_route_info(batch, vehicle_ids, node_ids, mask, f"{output_dir}/batch{batch_id}")
    # 返回值:
    #batch, vehicle_ids, node_ids, mask
# batch: 当前批次的数据
# vehicle_ids: 车辆ID
# node_ids: 节点ID
# mask: 掩码信息
# 这些信息可以用于后续的分析和可视化
    
    #------------------
    # calculation time
    #------------------
    avg_calc_time = np.mean(calc_time_list)
#     反映每个批次处理的平均耗时
# 用于评估模型的一般性能
# 反映计算时间的波动程度
# 评估模型性能的稳定性
# 反映处理整个数据集的总耗时
# 用于评估实际应用场景中的时间成本
# 例如: 如果avg_calc_time=0.5秒
# 则处理一个批次平均需要0.5秒
# 如果总共有100个批次
# 则处理整个数据集的总耗时为50秒
# 例如: 如果std_calc_time=0.1秒
# 则说明每个批次的处理时间波动较小
# 反映模型在不同批次上的一致性
# 例如: 如果std_calc_time=0.1秒
# 则说明每个批次的处理时间波动较小
# 反映模型在不同批次上的一致性 之后改
    std_calc_time = np.std(calc_time_list)
    total_calc_time = np.sum(calc_time_list)
    
    #-----------------
    # objective value
    #-----------------
    tour_length = torch.cat(tour_length_list, dim=0) # [eval_size] #    
    #down = torch.cat(down_list, dim=0) # [eval_size]
    conflict_cost = torch.cat(conflict_cost_list, dim=0)
    # 合并所有批次的数据
    # all_costs = tour_length + penalty_coef * down # [eval_size]
    #all_costs = tour_length # [eval_size]
    all_costs = tour_length + conflict_coef * conflict_cost
    #  计算总成本
    # 总成本 = 路线长度 + 惩罚系数 * 断电数
    avg_obj = torch.mean(all_costs).cpu().item() #改目标函数
    std_obj = torch.std(all_costs, unbiased=False).cpu().item()
    # 总成本的统计
    avg_tour_length = torch.mean(tour_length).cpu().item()
    std_tour_length = torch.std(tour_length, unbiased=False).cpu().item()
    # 路线长度的统计
    # 计算平均路线长度和标准差
    # avg_down = torch.mean(down).cpu().item()
    # std_down = torch.std(down, unbiased=False).cpu().item()
    # 惩罚值的统计  
    # 计算平均惩罚值和标准差
    #num_down = torch.cat(num_down_list, dim=0)
    # avg_num_down = torch.mean(num_down).cpu().item()
    # std_num_down = torch.std(num_down, unbiased=False).cpu().item()
    # 断电基站数量的统计
    # 计算平均断电基站数量和标准差
    # 统计计算部分也需相应修改
    avg_conflict_cost = torch.mean(conflict_cost).cpu().item()
    std_conflict_cost = torch.std(conflict_cost, unbiased=False).cpu().item()
    actual_tour_length = torch.cat(actual_tour_length_list, dim=0)
    avg_actual_tour_length = torch.mean(actual_tour_length).cpu().item()
    std_actual_tour_length = torch.std(actual_tour_length, unbiased=False).cpu().item()
    # 实际路线长度的统计
    # 计算平均实际路线长度和标准差

# Replace the current summary creation with this expanded version
    summary = {
        "avg_calc_time": avg_calc_time,
        "std_calc_time": std_calc_time,
        "avg_obj": avg_obj,
        "std_obj": std_obj,
        "avg_tour_length": avg_tour_length,
        "std_tour_length": std_tour_length,
        "avg_conflict_cost": avg_conflict_cost,
        "std_conflict_cost": std_conflict_cost,
        "total_calc_time": total_calc_time,
        "avg_actual_tour_length": avg_actual_tour_length,
        "std_actual_tour_length": std_actual_tour_length
    }
    
    # Process detailed metrics if available
    if detailed_metrics_list:
        # Initialize counters for each metric
        detailed_metrics_sums = {}
        metrics_count = len(detailed_metrics_list)
        
        # Sum up all metrics across batches
        for batch_metrics in detailed_metrics_list:
            for key, value in batch_metrics.items():
                if key not in detailed_metrics_sums:
                    detailed_metrics_sums[key] = 0
                detailed_metrics_sums[key] += value
        
        # Calculate averages and add to summary
        for key, total in detailed_metrics_sums.items():
            summary[key] = total / metrics_count
    
    # save log
    if output_dir is not None:
        if output_dir is not None:
           os.makedirs(output_dir, exist_ok=True)
        log_fname = f"{output_dir}/summary.json"
        with open(log_fname, "w") as f:
            json.dump(summary, f)
        
        write_metrics_to_csv(summary, output_dir)

    return summary

if __name__ == "__main__":
    import datetime
    import argparse
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument("--random_seed",      type=int, default=1234)
    parser.add_argument("--gpu",              type=int, default=-1)
    parser.add_argument("--num_workers",      type=int, default=8) # 4-》8
    parser.add_argument("--visualize_routes", action="store_true")
    parser.add_argument("--output_dir",       type=str, default=f"results/results_{now}")
    # dataset settings
    parser.add_argument("--dataset_path",    type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_load_size",   type=int, default=-1)
    # model settings
    parser.add_argument("--model_type",     type=str, default="rl")
    parser.add_argument("--model_path",     type=str, default=None)
    parser.add_argument("--model_dir",      type=str, default=None)
    parser.add_argument("--decode_type",    type=str, default="sampling")
    parser.add_argument("--search_width",   type=int, default=12800)
    parser.add_argument("--max_batch_size", type=int, default=12800)
    #parser.add_argument("--penalty_coef",   type=float, default=100)
    parser.add_argument("--conflict_coef", type=float, default=10.0, help="充电站冲突成本系数")
    
    # other parameters
    parser.add_argument("--vehicle_speed", type=float, default=41.0)
    parser.add_argument("--wait_time",     type=float, default=0.5)
    parser.add_argument("--time_horizon",  type=float, default=12.0)
    args = parser.parse_args()

    # prepare a directory
    if args.visualize_routes:
        os.makedirs(args.output_dir, exist_ok=True)

    eval(dataset_path=args.dataset_path,
         eval_batch_size=args.eval_batch_size,
         max_load_size=args.max_load_size,
         model_type=args.model_type,
         model_path=args.model_path,
         model_dir=args.model_dir,
         decode_type=args.decode_type,
         search_width=args.search_width,
         max_batch_size=args.max_batch_size,
         #penalty_coef=args.penalty_coef,
         conflict_coef=args.conflict_coef,  # 替代 penalty_coef
         vehicle_speed=args.vehicle_speed,
         wait_time=args.wait_time,
         time_horizon=args.time_horizon,
         random_seed=args.random_seed,
         gpu=args.gpu,
         num_workers=args.num_workers,
         visualize_routes=args.visualize_routes,
         output_dir=args.output_dir)