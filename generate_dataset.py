import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.util import save_dataset, load_dataset
import random
import json
import argparse
import _pickle as cpickle
from multiprocessing import Pool

class CIRPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.size = 0
        self.opts = None

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def generate(self,
                 num_samples: int, 
                 num_locs: int, 
                 num_depots: int, 
                 num_vehicles: int, 
                 vehicle_cap: float, #可以输出吗 
                 # vehicle_discharge_rate: float, #删除
                 depot_discharge_rate: list,  #删除
                 # discharge_lim_ratio: float = 0.1, # 充电下限 保证不会充完电回不去 之后改的时候去掉这个设置或者想想怎么改  删除
                 cap_ratio: float = 0.8, #真的影响吗
                 grid_scale: float = 100.0, 
                 random_seed: int = 1234):
        """
        please specify the random_seed usually. specify nothing when generating eval dataset in the rollout baseline.

        Paramters
        ---------
        
        Returns
        -------
        """
        self.dataset = self.generate_dataset(num_samples=num_samples,
                                             num_locs=num_locs,
                                             num_depots=num_depots,
                                             num_vehicles=num_vehicles,
                                             vehicle_cap=vehicle_cap,
                                             #vehicle_discharge_rate=vehicle_discharge_rate,
                                             depot_discharge_rate=depot_discharge_rate,
                                             # discharge_lim_ratio=discharge_lim_ratio,
                                             cap_ratio=cap_ratio,
                                             grid_scale=grid_scale,
                                             random_seed=random_seed)
        self.size = len(self.dataset)
        return self

    def load_from_pkl(self,
                      dataset_path: str,
                      load_dataopts: bool = True,
                      max_load_size: int = -1):
        """
        Paramters
        ---------
        dataset_path: str
            path to a dataset
        load_dataopts: bool
            whether or not options(argparse) of datasets are loaded
        """
        assert os.path.splitext(dataset_path)[1] == ".pkl" 
        #os.path.splitext(dataset_path) 将路径拆分为文件名和扩展名，例如 "data.pkl" 拆成 ("data", ".pkl")，[1] 取扩展名。
        if max_load_size > 0:
            self.dataset = load_dataset(dataset_path)[:max_load_size]
        else:
            self.dataset = load_dataset(dataset_path)
        self.size = len(self.dataset) #如果 self.dataset 是一个包含 100 个样本的列表，则 self.size = 100。
        if load_dataopts:
            self.opts = argparse.ArgumentParser()
            data_params_dir = os.path.split(dataset_path)[0]
            with open(f"{data_params_dir}/data_cmd_args.json", "r") as f: 
                #构造 JSON 文件路径，例如 "data/syn_ev12_n50/data_cmd_args.json
                self.opts.__dict__ = json.load(f)
                #将 JSON 内容直接赋值给 self.opts 的内部字典，覆盖默认的 ArgumentParser 属性。
                #print(dataset.opts.__dict__)  # 输出数据集的生成参数
        return self
    
    #这个方法用于从 pickle 文件中加载数据集和相关配置
    #dataset_path 是数据集的路径，load_dataopts 是一个布尔值，指示是否加载数据集的选项
    #max_load_size 是一个整数，表示要加载的样本数量的上限
    #如果 max_load_size 大于 0，则只加载前 max_load_size 个样本
    #否则加载整个数据集
    #self.dataset 是一个列表，包含加载的数据集
    #self.size 是数据集的大小
    #self.opts 是一个 argparse.ArgumentParser 对象，包含数据集生成时使用的参数
    #如果 load_dataopts 为 True，则加载数据集的选项
    #使用 json.load() 从 JSON 文件中读取数据
    #并将其赋值给 self.opts.__dict__
    #这样就可以通过 self.opts 访问数据集的生成参数
    #例如，self.opts.num_locs 可以获取数据集中客户点的数量

    def generate_instance(self,
                          num_locs: int,
                          num_depots: int, 
                          num_vehicles: int, 
                          vehicle_cap: float, 
                          #"vehicle_cap": torch.FloatTensor(vehicle_cap) * cap_ratio
                          #它是一个张量，表示车辆的电池容量。cap_ratio 是一个比例因子，用于缩放车辆的电池容量。
                          #如果 cap_ratio 是 0.8，那么车辆的电池容量将是原始容量的 80%。  #可以输出吗 cap_ratio vehicle cap #真的影响吗
                          #torch.FloatTensor(vehicle_cap) * cap_ratio
                          # vehicle_discharge_rate: float,
                          depot_discharge_rate_candidates: float, 
                          # discharge_lim_ratio: float = 0.1,
                          cap_ratio: float = 0.8, #可以输出吗 cap_ratio vehicle cap #真的影响吗
                          grid_scale: float = 100.0,
                          random_seed: int = None):
        if random_seed is not None:
            random_seed = int(random_seed) 
            torch.manual_seed(random_seed)
            random.seed(random_seed)

        coord_dim = 2
        num_nodes = num_locs + num_depots
        #-----------------------
        # vehicles (homogeneous) #车辆是同质的 之后改可以改成异质的
        #-----------------------
        #num_nodes: 总节点数 (客户点 + 充电站)
#        位置 ID 的范围是 [num_locs, num_nodes-1]，这个范围正好对应充电站的 ID
#例如，如果：

#num_locs = 50 (50个客户点)
#num_depots = 12 (12个充电站)
#num_vehicles = 3 (3辆车)
#那么：

#num_nodes = 62 (50 + 12)
#充电站的 ID 范围是 [50, 61]
#输出可能是 tensor([53, 58, 51])，表示 3 辆车分别初始停放在 ID 为 53、58、51 的充电站
        vehicle_cap = [vehicle_cap for _ in range(num_vehicles)] # [num_vehicle]
        vehicle_initial_position_id = torch.randint(num_locs, num_nodes, (num_vehicles, )) # [num_vehicles]
        # vehicle_discharge_rate = torch.FloatTensor([vehicle_discharge_rate for _ in range(num_vehicles)]) #删除
        vehicle_consump_rate = torch.FloatTensor([0.161 * grid_scale for _ in range(num_vehicles)]) 
        #-----------
        # locations
        #-----------
        # TODO : wide-range capacity candidates
        # capacity_consump = {   #删除 #统一 优先实验
        #     2.34: [0.6, 0.7],
        #     11.7: [1.1, 1.5, 1.7],
        #     35.1: np.arange(1.1, 6.0, 0.1).tolist(),
        #     46.8: np.arange(1.1, 6.0, 0.1).tolist()
        # } #不同耗电率的字典
#         容量 2.34 kWh:

# 可能的消耗率: [0.6, 0.7]
# 这是最小容量等级，消耗率选择也较低
# 容量 11.7 kWh:

# 可能的消耗率: [1.1, 1.5, 1.7]
# 中等容量等级，有三个消耗率选项
# 容量 35.1 kWh:

# 消耗率范围: 1.1 到 6.0，步长 0.1
# 使用 np.arange() 生成一个连续序列
# 较大容量，有更多消耗率选项
# 容量 46.8 kWh:

# 消耗率范围: 同样是 1.1 到 6.0，步长 0.1
# 最大容量等级
        # weights = [6, 9, 51, 59] #删除 
        loc_coords = torch.FloatTensor(num_locs, coord_dim).uniform_(0, 1) # [num_locs x coord_dim] 
        #torch.FloatTensor(num_locs, coord_dim)：创一个形状为 [num_locs x coord_dim] 的二维张量
        #base（loc->cus）feature dimension是6 e feature dimension d = 6)   横坐标 纵坐标 容量 耗电速率 t时刻的电量 预计down掉的时间
        #电站 有4个特征  之后改  横坐标 纵坐标 放电率 是否被车k访问
        #车有11个特征 横纵坐标 是否在cust中 当前EV cycle中的阶段 准备持续时间 充电准备时间 整理准备时间 移动准备时间 不可移动的时间（应该是算出来的） 
        # 电容量 t时刻的剩余电量 
        #         loc_dim=args.loc_dim,         # 基站位置特征维度 (7)
        # depot_dim=args.depot_dim,     # 充电站特征维度 (4)
        # vehicle_dim=args.vehicle_dim, # 车辆特征维度 (11)
        # loc_cap = torch.FloatTensor(random.choices(list(map(lambda x: round(x, 2), capacity_consump.keys())), k=num_locs, weights=weights)) # [num_locs] #删除
        #根据四种耗电 速率的权重随机选择容量
        # 6: 2.34 kWh
        # 9: 11.7 kWh
        # 51: 35.1 kWh
        # 59: 46.8 kWh
        # loc_initial_battery = (torch.rand(num_locs) * .5 + .5) * loc_cap  # 50 - 100% of the capacity [num_locs]   #删除
        # conditional probability
        #loc_consump_list = [] #删除
        #创建一个空列表用于存储每个位置的消耗率。
        # for cap in loc_cap:  #遍历每个位置的容量 删除
        #     loc_consump_list.append(random.choices(capacity_consump[round(cap.item(), 2)], k=1))
            #cap.item(): 将PyTorch张量转换为Python数值
            #capacity_consump[rounded_cap]: 获取该容量对应的可能消耗率列表
            #random.choices(capacity_consump[round(cap.item(), 2)], k=1): 从可能的消耗率中随机选择一个
            #k=1表示只选择一个元素
            #append()方法将随机选择的消耗率添加到 loc_consump_list 列表中
        # loc_consump_list: [num_locs x 1]
        #loc_consump_list 是一个二维列表，其中每个元素都是一个包含单个值的列表。
        #所以我们需要将其转换为一维张量
        # loc_consump_rate = torch.FloatTensor(loc_consump_list).squeeze(1) # [num_locs] 删除
        #--------
        # depots
        #--------
        depot_coords = torch.FloatTensor(num_depots, coord_dim).uniform_(0, 1) # [num_depots x coord_dim]
        
        depot_discharge_rate = torch.FloatTensor(random.choices(depot_discharge_rate_candidates, k=num_depots, weights=[0, 1])) # [num_depots] #同权重 #之后改 #可以输出吗  #真的影响吗
        # depot_discharge_rate_candidates: 3.0, 50.0 这是概率 20%可能性放电率是3.0 80%可能性放电率是50.0
        # 0.2: 3.0
        # 0.8: 50.0
        # ensure num. of depots whose discharge = 50 is more than 50 %
#         使用 random.choices() 随机选择放电率
# 将结果转换为 PyTorch 张量
# 生成形状为 [num_depots] 的一维张量
# 如果 num_depots = 5，可能生成：

# 由于权重设置，大多数充电站会获得 50.0 的高放电率
# 少数充电站会获得 3.0 的低放电率
        min_depot_count = int(0.5 * len(depot_discharge_rate)) #之后改 都是大电量 #可以输出吗  #真的影响吗
        if torch.count_nonzero(depot_discharge_rate > 10) < min_depot_count:
            idx = random.sample(range(len(depot_discharge_rate)), k=min_depot_count)
            depot_discharge_rate[idx] = 50.0
            #确保至少 50% 的充电站具有高放电率（> 10），这样可以保证充电服务质量。
        #-----------------------
           
        return {
            "grid_scale": torch.FloatTensor([grid_scale]),
            "loc_coords": loc_coords,
            # "loc_cap": loc_cap * cap_ratio, #删除
            # "loc_consump_rate": loc_consump_rate, #删除
            # "loc_initial_battery": loc_initial_battery * cap_ratio, #删除
            "depot_coords": depot_coords,
            "depot_discharge_rate": depot_discharge_rate,
            "vehicle_cap": torch.FloatTensor(vehicle_cap) * cap_ratio,
            "vehicle_initial_position_id": vehicle_initial_position_id,
            #"vehicle_discharge_rate": vehicle_discharge_rate, #删除
            "vehicle_consump_rate": vehicle_consump_rate, # 这个不能删除 因为这个关联到司机move时的电量 应该是个定值 在统一vehicle_consump_rate时考虑这点 谨慎
            # "vehicle_discharge_lim": discharge_lim_ratio * torch.FloatTensor(vehicle_cap) #删除
        }
        #这是返回数据信息包括电网大小 坐标 位置 容量 耗电速率 初始电量 充电站坐标 放电速率 车辆容量 初始位置 放电速率 耗电速率 放电下限
        #这就构成一个instance

    def generate_dataset(self,
                         num_samples: int, 
                         num_locs: int, 
                         num_depots: int, 
                         num_vehicles: int, 
                         vehicle_cap: float, 
                         # vehicle_discharge_rate: float,  #删除
                         depot_discharge_rate: list, 
                         # discharge_lim_ratio: float = 0.1, #删除
                         cap_ratio: float = 0.8,
                         grid_scale: float = 100.0,
                         random_seed: int = 1234):
        seeds = random_seed + np.arange(num_samples)
        return [
            self.generate_instance(num_locs=num_locs,
                                   num_depots=num_depots,
                                   num_vehicles=num_vehicles,
                                   vehicle_cap=vehicle_cap,
                                   # vehicle_discharge_rate=vehicle_discharge_rate, #删除
                                   depot_discharge_rate_candidates=depot_discharge_rate,  
                                   # discharge_lim_ratio=discharge_lim_ratio,  #删除
                                   cap_ratio=cap_ratio,
                                   grid_scale=grid_scale,
                                   random_seed=seed)
            for seed in tqdm(seeds)
        ]
#生成多个实例的数据集 多次调用generate_instance方法
#每次调用时使用不同的随机种子
#tqdm是一个用于显示进度条的库
#tqdm(seeds)会返回一个可迭代对象，包含所有的随机种子
#在for循环中，使用每个随机种子调用generate_instance方法
#生成一个实例，并将其添加到列表中
    def generate_dataset_para(self,
                              num_samples: int, 
                              num_locs: int, 
                              num_depots: int, 
                              num_vehicles: int, 
                              vehicle_cap: float, 
                              #vehicle_discharge_rate: float, #删除
                              depot_discharge_rate: list,
                              #discharge_lim_ratio: float = 0.1,
                              cap_ratio: float = 0.8,
                              grid_scale: float = 100.0,
                              random_seed: int = 1234,
                              num_cpus: int = 4):
        seeds = random_seed + np.arange(num_samples)
        #创建随机种子序列：
        #random_seed + np.arange(num_samples) 生成一个从 random_seed 开始的随机种子序列
        #例如，如果 random_seed 是 1234，num_samples 是 5，那么 seeds 将是 [1234, 1235, 1236, 1237, 1238]
        #使用多进程生成数据集
        #使用多进程来加速数据集生成
        #使用 tqdm 显示进度条
        with Pool(num_cpus) as pool:
            dataset = list(pool.starmap(self.generate_instance, tqdm([(num_locs, 
                                                                       num_depots,
                                                                       num_vehicles,
                                                                       vehicle_cap,
                                                                       #vehicle_discharge_rate, #删除
                                                                       depot_discharge_rate,
                                                                       #discharge_lim_ratio,
                                                                       cap_ratio,
                                                                       grid_scale,
                                                                       seed) for seed in seeds], total=len(seeds))))
        return dataset
    #这是一个并行生成数据集的方法，使用多进程来加速数据集生成。 提高效率
    #使用 Pool 类创建一个进程池，指定要使用的 CPU 核心数
    #使用 starmap 方法并行调用 generate_instance 方法
    #将每个随机种子和其他参数传递给 generate_instance 方法
    # 生成的数据集将被转换为列表并返回               
#参数映射：[(num_locs, num_depots, num_vehicles, ..., seed) for seed in seeds]
        #每个参数都被传递给 generate_instance 方法

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--type", type=str, nargs="*", default=["all"])
    parser.add_argument("--num_samples", type=int, nargs="*", default=[12, 1, 1]) #改一下数量
    
    parser.add_argument("--num_depots", type=int, default=1) 
    parser.add_argument("--num_locs", type=int, default=50)
    parser.add_argument("--num_vehicles", type=int, default=20)
    #问题规模参数
    parser.add_argument("--vehicle_cap", type=float, default=60.0) # all the vehicles have the same capacity
    # parser.add_argument("--vehicle_discharge_rate", type=float, default=10.0) #删除
    parser.add_argument("--depot_discharge_rate", type=float, nargs="*", default=[3.0, 50.0])
    #车辆和充电站参数
    parser.add_argument("--cap_ratio", type=float, default=0.8)  # 不必要充满 
    # parser.add_argument("--discharge_lim_ratio", type=float, default=0.1)
    parser.add_argument("--grid_scale", type=float, default=100.0)
    #约束参数
    parser.add_argument("--parallel", action="store_true")# 是否使用多进程生成数据集
    parser.add_argument("--num_cpus", type=int, default=4)  #   多进程使用的 CPU 核心数
    #多进程参数

    args = parser.parse_args()
    #解析命令行参数并存储到 args 对象中
    os.makedirs(args.save_dir, exist_ok=True)
    #创建用于保存数据集的目录
# exist_ok=True 表示：
# 如果目录已存在，不会报错
# 如果目录不存在，则创建新目录
    # validation check
    if args.type[0] == "all":
        assert len(args.num_samples) == 3
    else:
        assert len(args.type) == len(args.num_samples)
    num_samples = np.sum(args.num_samples)

    if args.parallel:
        dataset = CIRPDataset().generate_dataset_para(num_samples=num_samples,
                                                      num_locs=args.num_locs,
                                                      num_depots=args.num_depots,
                                                      num_vehicles=args.num_vehicles,
                                                      vehicle_cap=args.vehicle_cap,
                                                      #vehicle_discharge_rate=args.vehicle_discharge_rate, #删除
                                                      depot_discharge_rate=args.depot_discharge_rate,
                                                      #discharge_lim_ratio=args.discharge_lim_ratio, 
                                                      cap_ratio=args.cap_ratio,
                                                      grid_scale=args.grid_scale,
                                                      random_seed=args.random_seed,
                                                      num_cpus=args.num_cpus)
    else:
        dataset = CIRPDataset().generate_dataset(num_samples=num_samples,
                                                num_locs=args.num_locs,
                                                num_depots=args.num_depots,
                                                num_vehicles=args.num_vehicles,
                                                vehicle_cap=args.vehicle_cap,
                                                #vehicle_discharge_rate=args.vehicle_discharge_rate, #删除
                                                depot_discharge_rate=args.depot_discharge_rate,
                                                #discharge_lim_ratio=args.discharge_lim_ratio,
                                                cap_ratio=args.cap_ratio,
                                                grid_scale=args.grid_scale,
                                                random_seed=args.random_seed)
    if args.type[0] == "all":
        types = ["train", "valid", "eval"]
        #底线 必须提供3个样本数量（训练集、验证集、测试集
        #如果只提供一个样本数量，则会引发 AssertionError
        #assert len(args.num_samples) == 3
        #如果提供了3个样本数量，则将其分配给 types 列表
        #types = ["train", "valid", "eval"]
        #这表示数据集将分为训练集、验证集和测试集
        #根据提供的样本数量生成数据集

    else:
        types = args.type
    num_sample_list = args.num_samples
    num_sample_list.insert(0, 0)
#   
    start = 0
    for i, type_name in enumerate(types):
        start += num_sample_list[i]
        end = start + num_sample_list[i+1]
        divided_datset = dataset[start:end]
        save_dataset(divided_datset, f"{args.save_dir}/{type_name}_dataset.pkl")
        # # 在列表开头插入0
    #   初始状态：num_sample_list = [1000, 100, 50]
    # 插入0后：num_sample_list = [0, 1000, 100, 50]

#     训练集：dataset[0:1000]
# 验证集：dataset[1000:1100]
# 测试集：dataset[1100:1150]
# 插入0的目的是为了让索引计算更容易，避免了特殊处理第一个切片的情况。
    # save paramters
    with open(f'{args.save_dir}/data_cmd_args.json', 'w') as f:
        #保存在数据集目录下
        #文件名为 data_cmd_args.json
        json.dump(args.__dict__, f, indent=2)
        #args.__dict__: 将命令行参数对象转换为字典
# 包含所有设置的参数值
# indent=2: JSON 文件格式化，便于阅读

#命令行调用
# 首先通过命令行执行程序：
# python generate_dataset.py --save_dir data/syn_ev12_n50 --num_samples 1000 100 50
#主程序执行顺序
# 1. 解析命令行参数
#arser = argparse.ArgumentParser()
# ...设置各种参数...
#args = parser.parse_args()
# 2. 创建保存目录
#os.makedirs(args.save_dir, exist_ok=True)
# 3. 验证参数
#assert len(args.type) == len(args.num_samples)
# 4. 生成数据集
#dataset = CIRPDataset().generate_dataset(...)
# 5. 保存数据集
#save_dataset(divided_datset, f"{args.save_dir}/{type_name}_dataset.pkl")
# 6. 保存参数
#with open(f'{args.save_dir}/data_cmd_args.json', 'w') as f:
#    json.dump(args.__dict__, f, indent=2)
# 7. 完成
# 8. 退出程序
