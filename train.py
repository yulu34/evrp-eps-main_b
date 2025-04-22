# 标准库导入
import argparse          # 用于解析命令行参数
import datetime         # 处理日期和时间
import json            # 处理JSON数据格式
import os              # 操作系统相关功能,如文件路径操作

# 深度学习相关库
import torch           # PyTorch深度学习框架
import torch.optim as optim  # PyTorch优化器模块

# 进度条显示
from tqdm import tqdm  # 用于显示循环进度条

# 自定义模块导入
from models.am import AM4CIRP  # 注意力机制模型(Attention Model)用于CIRP问题
import models.baselines as rl_baseline  # 强化学习baseline实现
from utils.util import set_device, fix_seed  # 工具函数:设置设备(GPU/CPU)和固定随机种子
from generate_dataset import CIRPDataset  # 数据集生成和加载类

def main(args: argparse.Namespace) -> None:
    # set the random seed
    # 设置随机种子
    # 使得每次运行的结果相同
    # save parameter settings
#     -> None - 函数返回值类型提示:

# -> 表示返回类型注解
# None 表示这个函数不返回任何值
    if os.path.exists(args.checkpoint_dir):
        response = input(f"The directory '{args.checkpoint_dir}' already exists. Do you want to overwrite it? [y/n]: ").strip().lower()
        if response != 'y':
            assert False, "If you don't want to overwrite the checkpoint directory, please specify another checkpoint_dir."
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(f'{args.checkpoint_dir}/cmd_args.dat', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        # save the command line arguments to a file
        # 以JSON格式保存命令行参数到文件
        # 以缩进2的格式保存
        # indent=2表示缩进2个空格
        # 这样可以使得保存的文件更易读

    # set random seed
    fix_seed(args.random_seed)
    
    # device settings (gpu or cpu)
    use_cuda, device = set_device(args.gpu)

    # dataset
    dataset = CIRPDataset().load_from_pkl(args.dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers)

    # model & optimizer
    model = AM4CIRP(loc_dim=args.loc_dim,
                    depot_dim=args.depot_dim, 
                    vehicle_dim=args.vehicle_dim,
                    emb_dim=args.emb_dim,
                    num_heads=args.num_heads,
                    num_enc_layers=args.num_enc_layers,
                    dropout=args.dropout,
                    device=device)
    #建立model
    # model是一个AM4CIRP类的实例
    # AM4CIRP是一个注意力机制模型类
    # 这个模型是用于解决CIRP问题的
    # model是一个神经网络模型

    if use_cuda:
        model.to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # baseline for the REINFOCE
    if args.baseline == "rollout":
#         工作原理：
# 使用当前策略/模型进行多次采样
# 计算这些采样的平均回报作为baseline
# 更准确但计算开销大
# 主要特点：
# 基于模型的实际性能
# 能够自适应调整
# 计算量较大
# 更稳定的训练过程
        baseline = rl_baseline.RolloutBaseline(model, dataset.opts, args, device=device)
    elif args.baseline == "exponential": 
#         使用指数移动平均来更新baseline
# baseline = beta * old_baseline + (1-beta) * new_value
# 其中beta是平滑系数(0.8)
# 主要特点：
# 计算简单快速
# 内存效率高
# 可能不如Rollout准确
# 依赖于适当的beta值选择
        baseline = rl_baseline.ExponentialBaseline(args.beta, device=device)
    else:
        raise TypeError("Invalid baseline type :(")

    # train
    model.train()
    for epoch in range(args.epochs+1):
        # 每隔 checkpoint_interval 个 epoch 保存一次模型
        if epoch % args.checkpoint_interval == 0:
            # 1. 打印保存信息
            print(f"Epoch {epoch}: saving a model to {args.checkpoint_dir}/model_epoch{epoch}.pth...", 
                  end="", flush=True)
            
            # 2. 保存模型
            # 将模型移到 CPU
            torch.save(model.cpu().state_dict(), 
                      f"{args.checkpoint_dir}/model_epoch{epoch}.pth")
            
            # 3. 将模型移回原设备
            model.to(device)
            
            # 4. 打印完成信息
            print("done.")

        with tqdm(dataloader) as tq:
            for batch_id, batch in enumerate(tq):
                if use_cuda:
                    batch = {key: value.to(device) for key, value in batch.items()}
                # add options
                batch.update({
                    "time_horizon":  args.time_horizon,
                    "vehicle_speed": args.vehicle_speed,
                    "wait_time":     args.wait_time
                })

#                 time_horizon (时间范围)
# 定义了规划的总时间窗口（以小时为单位）
# 例如 time_horizon=12.0 表示规划12小时内的路线
# 限制了电动车能够运行的最大时间
# 影响可以访问的基站数量和充电次数
# 2. vehicle_speed (车辆速度)
# 定义电动车的行驶速度（km/h）
# 例如 vehicle_speed=41.0 表示时速41公里
# 影响：
# 车辆在给定时间内可以行驶的距离
# 到达基站和充电站的时间
# 能源消耗率
# 3. wait_time (等待时间)
# 定义车辆在每个基站的停留时间（小时）
# 例如 wait_time=0.5 表示在每个基站停留30分钟
# 用于：
# 给基站充电的时间
# 维护和检查的时间
# 影响整体路线规划的时间预算

                # # output tours
                # if args.debug:
                #     rewards, logprobs, reward_dict, tours = model(batch, "sampling", fname=f"{batch_id}")
                #     print(f"tour_length: {reward_dict['tour_length'].mean().item()}")
                #     #print(f"penalty: {reward_dict['penalty'].mean().item()}") #之后改 pennalty针对冲突和queue
                #     print(f"tour: {tours[0][0]}")
                # else:
                #     rewards, logprobs = model(batch, "sampling")
                if args.debug:
                    rewards, logprobs, reward_dict, tours = model(batch, "sampling", fname=f"{batch_id}")
                    print(f"tour_length: {reward_dict['tour_length'].mean().item()}")
                    print(f"conflict_cost: {reward_dict['conflict_cost'].mean().item()}")
                    print(f"tour: {tours[0][0]}")
                else:
                    rewards, logprobs = model(batch, "sampling")
# 模型输出：

# rewards: 每条路线的奖励值
# logprobs: 路线的对数概率
# reward_dict: 包含详细评估指标
# tour_length: 路线长度
# penalty: 违反约束的惩罚值
# tours: 具体的路线规划结果
# 采样方式：

# "sampling": 使用采样方式生成路线
# 相对于贪婪搜索，可以产生更多样的解
# 输出内容：

# 路线长度均值
# 惩罚项均值（违反约束的程度）
# 第一个批次中第一个样本的具体路线
                # calc baseline
                baseline_v, _ = baseline.eval(batch, rewards)
# rewards: 形状为 [batch_size]，表示每个样本的奖励值
# baseline_v: 基线值，用于减少方差，形状与 rewards 相同
                # calc loss
                advantage = (rewards - baseline_v).detach() # [batch_size]
#                 计算优势函数：实际奖励与基线值的差
# detach(): 分离计算图，防止梯度传递到 baseline
# 形状: [batch_size]
                loss = (advantage * logprobs).mean() # batch-wise mean [1]
                # logprobs: 策略网络输出的对数概率，形状 [batch_size]
# advantage * logprobs: 策略梯度的核心计算，形状 [batch_size]
# mean(): 对batch求平均，得到标量loss
                # backprop
                model_optimizer.zero_grad()
                # 清空梯度

                loss.backward()
                # 反向传播计算梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, norm_type=2)
                # 梯度裁剪，防止梯度爆炸
                # norm_type=2表示L2范数裁剪
                # 计算L2范数
                # 计算所有参数的L2范数
                # args.clip_grad_norm: 裁剪阈值
                model_optimizer.step()
                # 更新模型参数
                tq.set_postfix(cost=rewards.mean().item())
                # 更新进度条后缀信息
                # cost: 当前批次的平均奖励值
                # tqdm: 进度条库

        # logging
        # if epoch % args.log_interval == 0:
        #     print()

        baseline.epoch_callback(model, epoch)
        # 更新基线
        # epoch_callback: 每个epoch结束时调用


if __name__ == "__main__":
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    #------------------
    # general settings
    #------------------
    parser.add_argument("--random_seed",    type=int, default=1234)
    parser.add_argument("--gpu",            type=int, default=-1)
    parser.add_argument("--num_workers",    type=int, default=8) # 4-》8
    parser.add_argument("--checkpoint_dir", type=str, default=f"checkpoints/model_{now.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--debug",          action="store_true")

    #------------------
    # dataset settings
    #------------------
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--problem",      type=str, default="cirp")
    parser.add_argument("--coord_dim",    type=int, default=2)
    # coordinate dimension
    # 2D or 3D
    parser.add_argument("--num_samples",  type=int, default=1)
    parser.add_argument("--num_depots",   type=int, default=1)
    parser.add_argument("--num_locs",     type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=3)
    parser.add_argument("--vehicle_cap",  type=int, default=10) ## Vehicles are homogeneous (all have same capacity)
#vehicle_cap = [vehicle_cap for _ in range(num_vehicles)] # [num_vehicle]
#vehicle_cap是相同的所以车辆容量是同质的 之后改的时候可以考虑异质  10* 0.8 = 8 battery/energy capacity
    #-------------------
    # training settings
    #-------------------
    parser.add_argument("--batch_size",          type=int,   default=128)
    parser.add_argument("--epochs",              type=int,   default=1) #改小一点 为1 但不能为0
    parser.add_argument("--log_interval",        type=int,   default=20)
    parser.add_argument("--checkpoint_interval", type=int,   default=1)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    # learning rate
    # 0.0001
    # 1e-4
    parser.add_argument("--clip_grad_norm",      type=float, default=1.0)
    # gradient clipping
    # 1.0
    parser.add_argument("--dropout",             type=float, default=0.2)
    # for greedy baseline
    parser.add_argument("--num_greedy_samples",  type=int,   default=128) #改小 1280-》1
    # greedy sampling
    # greedy sampling
    # greedy sampling是指在每个时间步选择当前最优的动作
    parser.add_argument("--greedy_batch_size",   type=int,   default=128)

    #----------------
    # model settings
    #----------------
    parser.add_argument("--loc_dim",        type=int, default=3) #之后改
    parser.add_argument("--depot_dim",      type=int, default=4)
    parser.add_argument("--vehicle_dim",    type=int, default=11)
    parser.add_argument("--emb_dim",        type=int, default=128)
    # embedding dimension
    # 128
#     GPU优化：128是2的幂次(2^7)，对GPU计算友好
# 矩阵运算：适合现代硬件架构的内存对齐
# 多头注意力：容易被8整除(本模型使用8个注意力头)
# 在Transformer类架构中常见的维度选择:
# - 128 (小型模型)
# - 256 (中型模型)
# - 512 (大型模型)
# model = AM4CIRP(
#     emb_dim=args.emb_dim,     # 128维嵌入
#     num_heads=args.num_heads,  # 8个注意力头
#     # 每个注意力头的维度 = 128/8 = 16维
# )
    parser.add_argument("--num_heads",      type=int, default=8)
    # number of attention heads
    # 8
    parser.add_argument("--num_enc_layers", type=int, default=2)
    # number of encoder layers
    # 2
#     2层是一个经验性的平衡点
# 问题复杂度匹配

# 路由规划问题相对简单，不需要太深的网络
# 2层足以捕获基站位置和车辆状态的关系
# 避免过拟合

# 较少的层数减少模型参数
# 有助于防止在小数据集上过拟合

    #-------------------
    # baseline settings
    #-------------------
    parser.add_argument("--baseline", type=str,   default="rollout")
    parser.add_argument("--bl_alpha", type=float, default=0.05)
    # baseline alpha
    # 0.05
#     平滑更新

# 较小的α值(0.05)确保基线值平滑更新
# 防止基线值剧烈波动
# 稳定训练

# 基线需要相对稳定以提供可靠的比较基准
# 0.05提供了足够慢的更新速度
# 经验值

# 在强化学习中是常用的学习率
# 在实践中证明有效
    parser.add_argument("--beta",     type=float, default=0.8) # beta是充电上线 不必要充满 之后改 考虑把这个去掉或者很容易满足
    
    #------------------
    # other parameters
    #------------------
    parser.add_argument("--vehicle_speed", type=float, default=41.0)
    parser.add_argument("--wait_time",     type=float, default=0.5)
    parser.add_argument("--time_horizon",  type=float, default=12.0)
    parser.add_argument("--conflict_coef", type=float, default=10.0, help="充电站冲突成本系数")
    #默认值 

    args = parser.parse_args()

    main(args)