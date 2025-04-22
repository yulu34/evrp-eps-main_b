# 深度学习相关
import torch            # PyTorch深度学习框架

# 数据处理相关
import pickle          # 用于序列化和反序列化Python对象
import numpy as np     # 科学计算库，提供多维数组支持

# 可视化相关
import matplotlib.pyplot as plt    # 主要的绘图库
import matplotlib.cm as cm        # 颜色映射模块
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # 在图上添加图像和注释
from matplotlib import patches    # 绘制各种形状(矩形、圆形等)

# 系统和工具相关
import os             # 操作系统接口，处理文件和路径
import math           # 数学函数库
import copy           # 对象复制
import subprocess     # 执行系统命令

# 类型提示
from typing import Dict, List    # 类型注解，提供代码提示和检查

DPI = 70 # 之后改 这个是真正起作用的 清晰度可用就行 # 之后改真正影响图片质量的是state里面的参数 之后改 每英寸的点数 150-》70
SMALL_VALUE = 1e-9
BIT_SMALL_VALUE = 1e-3
SAVE_PICTURE = True # 是否保存每一帧图片 #可以改为True # 之后改 改为true 不想生成视频也可以只把这个关了 会有数据 # 在readme中如果调用valid需要做图 这个必须打开 不打开这个只打开visual没用
# 关掉这个因为不关在reproduce时会生成很多 reproduce的可视化是做图 不是做视频
SAVE_HISTORY = True # 是否保存历史 之后改这个具体保持的是什么
OUTPUT_INTERVAL = 0.02 # 输出时间间隔，控制动画更新频率 之后改这个可用就行  #时间切片 0.02-》
FPS = 40 # 之后改 清晰度可用就行 #这里改成了 60 -》40
UNEQUAL_INTERVAL = False # 是否使用不等时间间隔
COEF = 1.0  # 一般系数，用于调整某些计算的比例
V_COEF = 1.0   # 车辆相关的系数，用于调整车辆相关的计算

#--------------------------
# when input is route_list
#--------------------------
def visualize_routes2(route_list: List[List[int]],
                      inputs: Dict[str, torch.tensor], 
                      fname: str, 
                      device: str) -> None:
    state = CIRPState(inputs, device, fname)
    count = [2 for _ in range(len(route_list))]
    while not state.all_finished():  # 当还有未完成的车辆时继续循环
        # 1. 选择下一个要更新的车辆
        selected_vehicle_id = state.get_selected_vehicle_id()
        
        # 2. 获取该车辆的下一个目标节点
        node_id = route_list[selected_vehicle_id][count[selected_vehicle_id]]
        # route_list: 包含所有车辆路线的列表
# 结构示例:
# route_list = [
#     [0, 5, 2, 8],    # 车辆0的路线：访问节点0->5->2->8
#     [1, 3, 6, 4],    # 车辆1的路线：访问节点1->3->6->4
#     [7, 9, 2, 1]     # 车辆2的路线：访问节点7->9->2->1
# ]

# count: 记录每辆车当前执行到路线的哪个位置的数组
# 例如：count = [0, 2, 1] 表示：
# - 车辆0在路线的第0个位置
# - 车辆1在路线的第2个位置
# - 车辆2在路线的第1个位置

# selected_vehicle_id: 当前选中的车辆ID
# 例如：selected_vehicle_id = 1

# route_list[selected_vehicle_id]: 获取选中车辆的完整路线
# count[selected_vehicle_id]: 获取该车辆当前的位置
# node_id: 最终获取到的目标节点ID
        node_ids = torch.LongTensor([node_id])  # 转换为张量格式
        
        # 3. 更新状态
        state.update(node_ids)  # 更新系统状态(包括车辆位置、电量等)
        
        # 4. 更新该车辆的计数器
        count[selected_vehicle_id] += 1  # 移动到下一个节点
    
    if SAVE_PICTURE:
        state.output_gif()
    if SAVE_HISTORY:
        state.output_batt_history()
        state.output_mask_history()  
        state.output_action_history()  # 新增
        state.output_mask_calc_history()

#-------------------------------------------------
# when input is the selected vehicle & node order
#-------------------------------------------------
def visualize_routes(vehicle_ids: torch.tensor, 
                     node_ids: torch.tensor, 
                     inputs: Dict[str, torch.tensor], 
                     fname: str, 
                     device: str):
    if vehicle_ids.dim() < 2:  # 如果是单批次数据
        # 扩展为批次维度为1的数据
        vehicle_ids = vehicle_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        node_ids = node_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        
    state = CIRPState(inputs, device, fname)
    # 创建状态对象
    state = CIRPState(inputs, device, fname)
    count = 0
    
    # 循环直到所有车辆完成任务
    while not state.all_finished():
        # 验证当前选择的车辆ID是否正确
        assert (state.next_vehicle_id != vehicle_ids[:, count]).sum() == 0
        # 更新状态
        state.update(node_ids[:, count])
        count += 1
    
    if SAVE_PICTURE:
        state.output_gif()
    if SAVE_HISTORY:
        state.output_batt_history()
        state.output_mask_history()  
        state.output_action_history()  # 新增

# vehicle_ids = [0, 1, 0, 2, 1, ...]  # 表示每一步由哪辆车执行
# node_ids = [3, 5, 2, 7, 4, ...]     # 表示每一步访问哪个节点

def save_route_info(inputs: Dict[str, torch.tensor], 
                    vehicle_ids: torch.tensor, 
                    node_ids: torch.tensor, 
                    mask: torch.tensor, 
                    output_dir: str) -> None:
    if vehicle_ids.dim() < 2:  # 如果维度小于2(即只有一个批次)
        # unsqueeze(0): 在第0维增加一个维度
        # expand(1, node_ids.size(-1)): 扩展到指定形状
        vehicle_ids = vehicle_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        node_ids = node_ids.unsqueeze(0).expand(1, node_ids.size(-1))
        if mask.dim() < 2:
            mask = mask.unsqueeze(0).expand(1, mask.size(-1) if mask.dim() > 0 else node_ids.size(-1))
        #         # 原始 vehicle_ids: [5, 2, 3]  # 1维张量
        # vehicle_ids.unsqueeze(0)  # [[5, 2, 3]]  # 2维张量
        # # expand 后: [[5, 2, 3]]  # 保持形状不变，因为已经是正确的形状
    sample = 0  # 选择第一个样本
    
    # .tolist() 将 PyTorch 张量转换为 Python 列表
    loc_coords = inputs["loc_coords"][sample].tolist()  # 获取基站坐标
    depot_coords = inputs["depot_coords"][sample].tolist()  # 获取充电站坐标
    veh_init_pos_ids = inputs["vehicle_initial_position_id"][sample].tolist()  # 获取车辆初始位置ID

    depot_discharge_rates = inputs["depot_discharge_rate"][sample]
    small_depots_threshold = 10.0 
    small_depots_list = (depot_discharge_rates < small_depots_threshold).tolist()
    depot_discharge_rates_list = depot_discharge_rates.tolist()

    #     inputs = {
    #     "loc_coords": torch.tensor([
    #         [[1.0, 2.0], [3.0, 4.0]],  # 第一个样本的基站坐标
    #         [[5.0, 6.0], [7.0, 8.0]]   # 第二个样本的基站坐标
    #     ]),
    #     "depot_coords": torch.tensor([
    #         [[0.0, 0.0], [1.0, 1.0]],  # 第一个样本的充电站坐标
    #         [[2.0, 2.0], [3.0, 3.0]]   # 第二个样本的充电站坐标
    #     ]),
    #     "vehicle_initial_position_id": torch.tensor([
    #         [0, 1, 2],  # 第一个样本的车辆初始位置
    #         [3, 4, 5]   # 第二个样本的车辆初始位置
    #     ])
    # }
    
    # # 当 sample = 0 时：
    # loc_coords = [[1.0, 2.0], [3.0, 4.0]]
    # depot_coords = [[0.0, 0.0], [1.0, 1.0]]
    # veh_init_pos_ids = [0, 1, 2]

    # 标记不可用的充电站
    ignored_depots = (inputs["depot_discharge_rate"][sample] < 10.0).tolist()
    
    # 将车辆初始位置的充电站标记为可用
    for veh_init_pos_id in veh_init_pos_ids:
        ignored_depots[veh_init_pos_id - len(loc_coords)] = False
        # 之后改 统一充电率时要大于10

    vehicle_id = vehicle_ids[sample]  # 获取车辆ID序列
    node_id = node_ids[sample]       # 获取节点ID序列
    mask = mask[sample]              # 获取掩码序列
    # 为每辆车创建空路线列表
    routes = [[] for _ in range(torch.max(vehicle_id)+1)]
    
    # 验证数据长度一致性
    assert len(vehicle_id) == len(node_id) and len(node_id) == len(mask)
    # 将每辆车的初始位置添加到对应的路线中
    for veh_id, veh_init_pos_id in enumerate(veh_init_pos_ids):
        routes[veh_id].append(veh_init_pos_id)
    # add subsequent positions
    for step, skip in enumerate(mask):
        if skip:  # 如果是掩码位，终止添加
            break
        else:  # 将节点添加到对应车辆的路线中
            routes[vehicle_id[step]].append(node_id[step].item())
    route_info = {
        "loc_coords": loc_coords,
        "depot_coords": depot_coords,
        "ignored_depots": ignored_depots,
        "small_depots": small_depots_list,
        "depot_discharge_rates": depot_discharge_rates_list,
        "route": routes
    }
    save_dir = f"{output_dir}-sample{sample}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存PKL文件
    pkl_path = f"{save_dir}/route_info.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(route_info, f)
        # 2. 生成可读的文本文件
    txt_path = f"{save_dir}/route_info_readable.txt"
    with open(txt_path, "w") as f:
        f.write("=== Route Information ===\n\n")
        
        # 写入基站坐标
        f.write("Location Coordinates:\n")
        for i, coord in enumerate(loc_coords):
            f.write(f"Location {i}: ({coord[0]:.4f}, {coord[1]:.4f})\n")
        f.write("\n")
        
        # 写入充电站坐标
        f.write("Depot Coordinates and Status:\n")
        for i, coord in enumerate(depot_coords):
            ignored_status = "Ignored" if ignored_depots[i] else "Active"
            small_status = "Yes" if small_depots_list[i] else "No"
            rate = depot_discharge_rates_list[i]
            f.write(f"  Depot {i}: ({coord[0]:.4f}, {coord[1]:.4f}) - Rate: {rate:.2f} - Small Rate (<{small_depots_threshold}): {small_status} - Availability: {ignored_status}\n")
        f.write("\n")
        
        # 写入路线信息
        f.write("Vehicle Routes:\n")
        for veh_id, route in enumerate(routes):
            f.write(f"Vehicle {veh_id}: {route}\n")
            # 添加详细路线描述
            f.write("Detailed path:\n")
            if len(route) > 1:
                for i in range(len(route)-1):
                    curr_id = route[i]
                    next_id = route[i+1]
                    
                    curr_coord = loc_coords[curr_id] if curr_id < len(loc_coords) else depot_coords[curr_id - len(loc_coords)]
                    next_coord = loc_coords[next_id] if next_id < len(loc_coords) else depot_coords[next_id - len(loc_coords)]
                    
                    distance = ((curr_coord[0] - next_coord[0])**2 + (curr_coord[1] - next_coord[1])**2)**0.5
                    
                    f.write(f"      Step {i+1}: {curr_id} -> {next_id} | "
                            f"Coords: ({curr_coord[0]:.4f}, {curr_coord[1]:.4f}) -> "
                            f"({next_coord[0]:.4f}, {next_coord[1]:.4f}) | "
                            f"Distance: {distance:.4f}\n")
            else:
                f.write("      (No movement)\n")
            f.write("\n")
        #之后改 这里加了详细路径数据方便调试
        # 之后运行着看一下这个pkl里面的数据
        # 这部分就像之前VRP里面打印路径output一样 只不过重定向到文件夹 值得大量分析

class CIRPState(object):
    def __init__(self,
                 input: dict,
                 device: str,
                 fname: str = None):
            # 确保 episode_step 在最开始就初始化
        self.episode_step = 0
        self.fname = fname
        #-----------
        # locations
        #-----------
        # static 
        self.loc_coords       = input["loc_coords"]         # [batch_size x num_locs x coord_dim]
        # self.loc_cap          = input["loc_cap"]            # [batch_size x num_locs]
        # self.loc_consump_rate = input["loc_consump_rate"]   # [batch_size x num_locs]
        # # dynamic
        # self.loc_curr_battery = input["loc_initial_battery"].clone() # [batch_size x num_locs]
        #数据分为动态数据和静态数据
        #--------
        # depots
        #--------
        # static
        self.depot_coords = input["depot_coords"] # [batch_size x num_depots x coord_dim]
        self.depot_discharge_rate = input["depot_discharge_rate"] # [batch_size x num_depots]
        # depots whose discharge rate is less than threshold
        self.th = 10.0
        self.small_depots = self.depot_discharge_rate < self.th # [batch_size x num_depots]
        # 这个之后改成全部都是大于10的电站 这样ignored——depot应该都是true
        
        #-------------------
        # locations & depot
        #-------------------
        self.coords = torch.cat((self.loc_coords, self.depot_coords), 1)

        #----------
        # vehicles
        #----------
        # TODO: speed should depend on edges, not vehicles
        # static
        self.vehicle_cap = input["vehicle_cap"].clone().detach()   # [batch_size x num_vehicles]
        # self.vehicle_discharge_lim = input["vehicle_discharge_lim"].clone().detach() # [batch_size x num_vehicles]
        # self.vehicle_discharge_rate = input["vehicle_discharge_rate"] # [batch_size x num_vehicles]
        # dynamic
        self.vehicle_position_id  = input["vehicle_initial_position_id"].clone() # node ids in which vehicles are [batch_size x num_vehicles]
        self.vehicle_curr_battery = input["vehicle_cap"].clone() # initialized with battery fully chareged [batch_size x num_vehicles]
        # 车辆当前电量，初始为满电
        self.vehicle_unavail_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device) # [batch_size x num_vehicles]
        self.wait_vehicle = torch.full(self.vehicle_cap.size(), False, device=device) # [batch_size x num_vehicles] stores whether the vehicle is waiting or not
        self.phase_id = {"move": 0, "pre": 1, "charge": 2, "post": 3}
            # 四个状态字典
        self.phase_id_max = max(self.phase_id.values())
         # 获取最大状态ID
        self.vehicle_phase = torch.full(self.vehicle_cap.size(), self.phase_id["post"], dtype=torch.long, device=device) # [batch_size x num_vehicles] # 0 -> "move", 1 -> "charge"
        # 初始化车辆状态为"post" [batch_size x num_vehicles]
        self.vehicle_consump_rate = input["vehicle_consump_rate"].clone()
        self.vehicle_position_id_prev  = input["vehicle_initial_position_id"].clone() # for visualization
        # 车辆前一个位置（用于可视化）
        self.vehicle_move_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
        self.vehicle_pre_time  = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
        self.vehicle_work_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
        self.vehicle_post_time = torch.zeros(self.vehicle_cap.size(), dtype=torch.float, device=device)
      # 问题实例(batch)实际上是指一个完整的配送场景 一个问题实例包含：  
      # 基站信息：
    #   #   loc_info = {
    #     "loc_coords": [[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]],  # 3个基站的坐标
    #     "loc_cap": [100, 150, 120],                           # 每个基站的电池容量
    #     "loc_initial_battery": [80, 90, 70],                  # 每个基站的初始电量
    #     "loc_consump_rate": [5, 6, 4]                         # 每个基站的耗电速率
    # }
    # 充电站信息：
    #     depot_info = {
    #     "depot_coords": [[0.2, 0.3], [0.6, 0.7]],            # 2个充电站的坐标
    #     "depot_discharge_rate": [20, 25]                      # 充电站的充电速率
    # }
    # 车辆信息：
    #     vehicle_info = {
    #     "vehicle_cap": [200, 180],                           # 2辆车的电池容量
    #     "vehicle_initial_position_id": [0, 1],               # 车辆初始位置
    #     "vehicle_discharge_rate": [15, 12],                  # 车辆放电速率
    #     "vehicle_consump_rate": [2, 2.5]                     # 车辆行驶耗电率
    # }
      # 所以在做图时会一个batch一个batch往后推动
      # 同时处理多个不同场景
#       如果 batch_size > 1 时（理论上可以，但当前代码未实现）：
# batch_size = 3
# inputs = {
#     "场景1": {...},  # 不同的基站/充电站/车辆配置
#     "场景2": {...},
#     "场景3": {...}
# }

        #-----------
        # paramters
        #-----------
        self.batch_size   = self.loc_coords.size(0)
        self.coord_dim    = self.loc_coords.size(-1)
        self.num_locs     = self.loc_coords.size(1)
        self.num_depots   = self.depot_coords.size(1)
        self.num_vehicles = self.vehicle_cap.size(1)
        self.num_nodes    = self.num_locs + self.num_depots
        self.wait_time    = torch.FloatTensor([input["wait_time"]]).to(device)
        self.time_horizon = torch.FloatTensor([input["time_horizon"]]).to(device)
        self.speed        = V_COEF * (torch.FloatTensor([input["vehicle_speed"]]).to(device) / input["grid_scale"]).squeeze(-1) # [batch_size x 1] -> [batch_size]
        #self.max_cap      = torch.max(torch.tensor([torch.max(self.loc_cap).item(), torch.max(self.vehicle_cap).item()])).to(device) 
        #之后改
        # Change this to only use vehicle capacity:
        self.max_cap = torch.max(self.vehicle_cap).to(device)
        self.device       = device
        #self.loc_min_battery = 0.0 # TODO #之后改 这里看在改为客户时是否有问题
        depot_prepare_time = 0.17 # 10 mins
        loc_prepare_time   = 0.5  # 30 mins
        # 在充电站和基站的准备/完成时间
        # 之后改 在充电站和基站有很多 在基站可以有这个时间 长一点 在充电站（客户）也可以有这个时间当作服务客户的时间 能不改尽量不改
        self.pre_time_depot  = torch.FloatTensor([depot_prepare_time]).to(device)
        self.post_time_depot = torch.FloatTensor([depot_prepare_time]).to(device)
        self.post_time_loc   = torch.FloatTensor([loc_prepare_time]).to(device)
        self.pre_time_loc    = torch.FloatTensor([loc_prepare_time]).to(device)
        self.return_depot_within_time_horizon = False  # 是否需要在时间范围内返回充电站
        
        # 添加用于详细指标的跟踪变量
        self.loc_first_visit_time = torch.full((self.batch_size, self.num_locs), -1.0, dtype=torch.float, device=device)
        self.total_travel_time = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        self.total_service_time = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        self.total_charge_time = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        self.total_wait_time = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        self.depot_busy_time = torch.zeros((self.batch_size, self.num_depots), dtype=torch.float, device=device)
        self.sum_queue_length_time = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        self.max_queue_length = torch.zeros(self.batch_size, dtype=torch.int, device=device)
        self.vehicle_queuing_time = torch.zeros((self.batch_size, self.num_vehicles), dtype=torch.float, device=device)
        self.num_charge_events = torch.zeros(self.batch_size, dtype=torch.int, device=device)
        self.total_charge_energy = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        self.total_travel_energy = torch.zeros(self.batch_size, dtype=torch.float, device=device)
        self.total_supply_energy = torch.zeros(self.batch_size, dtype=torch.float, device=device)
    
        #-------
        # utils
        #-------
        self.loc_arange_idx     = torch.arange(self.num_locs).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
                # self.loc_arange_idx = torch.arange(self.num_locs)          # 创建从0到num_locs-1的序列
                #               .to(self.device)                      # 移动到指定设备(CPU/GPU)
                #               .unsqueeze(0)                         # 增加batch维度
                #               .expand(self.batch_size, -1)          # 扩展到batch_size大小
                #                 torch.arange(3)           # [0, 1, 2]
                # unsqueeze(0)             # [[0, 1, 2]]
                # expand(2, -1)            # [[0, 1, 2],
                #                         #  [0, 1, 2]]  # 如果batch_size=2
        self.depot_arange_idx   = torch.arange(self.num_locs, self.num_nodes).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
        #         torch.arange(3, 5)       # [3, 4]
        # unsqueeze(0)             # [[3, 4]]
        # expand(2, -1)            # [[3, 4],
        #                         #  [3, 4]]  # 如果batch_size=2
        self.node_arange_idx    = torch.arange(self.num_nodes).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
        #         torch.arange(5)          # [0, 1, 2, 3, 4]
        # unsqueeze(0)             # [[0, 1, 2, 3, 4]]
        # expand(2, -1)            # [[0, 1, 2, 3, 4],
        #                         #  [0, 1, 2, 3, 4]]  # 如果batch_size=2
        self.vehicle_arange_idx = torch.arange(self.num_vehicles).to(self.device).unsqueeze(0).expand(self.batch_size, -1)
        
#         快速匹配和掩码生成：

# 用于生成掩码来标识特定节点或车辆
# 用于节点ID的快速匹配和比较
# 批处理支持：

# 通过expand操作支持批量处理多个场景
# 每个batch共享相同的索引结构
# 高效索引：

# 避免重复创建索引张量
# 支持并行处理多个实例
        #----------------------
        # common dynamic state
        #----------------------
        self.loc_visited = torch.full((self.batch_size, self.num_locs), False, dtype=torch.bool, device=device)
    
        self.next_vehicle_id = torch.zeros(self.batch_size, dtype=torch.long, device=device) # firstly allocate 0-th vehicles
        self.skip = torch.full((self.batch_size,), False, dtype=bool, device=device) # [batch_size]
        self.end  = torch.full((self.batch_size,), False, dtype=bool, device=device) # [batch_size]
#         skip: 标记是否跳过当前场景的处理
# end: 标记场景是否已完成
        self.current_time = torch.zeros(self.batch_size, dtype=torch.float, device=device) # [bath_size]
        self.tour_length = torch.zeros(self.batch_size, dtype=torch.float, device=device) # [batch_size]
#         current_time: 追踪每个场景的当前时间
# tour_length: 记录每个场景的总路径长度
        #self.penalty_empty_locs = torch.zeros(self.batch_size, dtype=torch.float, device=device) # [batch_size]
        self.accumulated_conflict_cost = torch.zeros(self.batch_size, dtype=torch.float, device=device)  # 新增：累计冲突成本
        next_vehicle_mask = torch.arange(self.num_vehicles).to(self.device).unsqueeze(0).expand(self.batch_size, -1).eq(self.next_vehicle_id.unsqueeze(-1)) # [batch_size x num_vehicles]
        # 记录基站电量耗尽的惩罚值
        self.mask = self.update_mask(self.vehicle_position_id[next_vehicle_mask], next_vehicle_mask)
        # 基于车辆位置和车辆掩码更新状态掩码
        # 控制车辆可访问的节点
        self.charge_queue = torch.zeros((self.batch_size, self.num_depots, self.num_vehicles), dtype=torch.long, device=device)
# 初始化充电站的车辆排队情况
# shape: [batch_size × num_depots × num_vehicles]
# next_vehicle_mask = torch.arange(self.num_vehicles)        # 创建车辆索引序列
#                     .to(self.device)                       # 移至指定设备
#                     .unsqueeze(0)                          # 增加批次维度
#                     .expand(self.batch_size, -1)           # 扩展到批次大小
#                     .eq(self.next_vehicle_id.unsqueeze(-1))# 与选中车辆ID比较

# 假设 batch_size=2, num_vehicles=3, num_depots=2:
# self.next_vehicle_id = [0, 0]  # 每个批次从0号车开始

# self.skip = [False, False]  # 两个场景都未跳过
# self.end = [False, False]   # 两个场景都未完成

# self.current_time = [0.0, 0.0]  # 两个场景的当前时间
# self.tour_length = [0.0, 0.0]   # 两个场景的路径长度

# # 充电队列示例
# self.charge_queue = [
#     # batch 0
#     [[0, 0, 0],  # depot 0的车辆队列
#      [0, 0, 0]], # depot 1的车辆队列
#     # batch 1
#     [[0, 0, 0],  # depot 0的车辆队列
#      [0, 0, 0]]  # depot 1的车辆队列
# ]
# 这些变量共同构成了系统的状态追踪机制，用于：

# 跟踪每个场景的进度
# 记录时间和距离信息
# 管理充电队列状态
# 计算性能指标(如惩罚值)
# 控制车辆的可访问节点
        #-------------------
        # for visualization
        #-------------------
        if fname is not None:
             #之后改 增加新的信息全面了解这个玩具
            self.action_names = [
                # 来自上一步 update 的信息
                "curr_vehicle_id",       # 执行上一步动作的车辆 ID
                "next_node_id",          # 上一步选择的目标节点 ID
                "do_wait",               # 上一步动作是否为"等待"
                "travel_time",           # 上一步动作计算出的行驶时间
                "charge_time",           # 上一步动作计算出的充电/供电时间
                
                # 来自上一步 update_state 的信息
                "elapsed_time",          # 上一步状态更新实际推进的时间
                
                # 系统状态指标
                "tour_length",           # 累积总行驶距离
                #"penalty_empty_locs",    # 累积总基站断电惩罚
                
                # 派生状态信息
                #"down_locs_count",       # 电量耗尽的基站数量
                "queued_vehicles_count", # 排队中的车辆数量
                "charging_vehicles_count", # 充电中的车辆数量
                #"supplying_vehicles_count" # 供电中的车辆数量
                "accumulated_conflict_cost",  # 新增：累计冲突成本
            ]
            
            # 初始化动作历史记录字典
            self.action_histories = {
                action_name: [[] for _ in range(self.batch_size)] 
                for action_name in self.action_names
            }
            
            
            # 保存上一步的动作信息的临时存储
            self.last_action_info = {
                "curr_vehicle_id": None,
                "next_node_id": None,
                "do_wait": None,
                "travel_time": None, 
                "charge_time": None,
                "elapsed_time": None
            }

            # 之后改定义掩码字典
            self.mask_names = [
                # 电量相关掩码 (3个)
                # "loc_is_down",      # 基站电量耗尽
                # "loc_is_full",      # 基站电量满
                # "loc_is_normal",    # 基站电量正常
                
                # 节点访问掩码 (1个)
                "vehicle_position_id", # 车辆当前位置
                
                # 充电站掩码 (2个)
                "small_depots",      # 低功率充电站
                "charge_queue",      # 充电队列
                
                # 车辆相关掩码 (2个)
                "vehicle_phase",     # 车辆阶段
                "wait_vehicle",      # 等待车辆
    
                "queued_vehicles",   # <--- 添加：车辆是否在充电站排队 (顺序号 > 1)
                "charging_vehicles", # <--- 添加：车辆是否正在充电 (非排队)

                # 位置类型掩码 (2个)
                "at_depot",         # 在充电站
                "at_loc",          # 在基站
                
                # 状态和更新掩码 (3个)
                "next_vehicle_mask", # 下一个车辆
                "skip",             # 跳过标记
                "end"              # 结束标记
            ]
            
            # 初始化掩码历史记录字典
            self.mask_histories = {
                mask_name: [[] for _ in range(self.batch_size)] 
                for mask_name in self.mask_names
            }
            #             # 假设 batch_size = 2，系统运行过程中的某个时刻：
            # self.mask_histories = {
            #     "loc_is_down": [
            #         [True, False, True],    # 批次0的基站断电历史
            #         [False, False, True]    # 批次1的基站断电历史
            #     ],
            #     "vehicle_phase": [
            #         [0, 1, 2],    # 批次0的车辆阶段历史
            #         [1, 2, 0]     # 批次1的车辆阶段历史
            #     ]
            #     # ...其他状态的历史记录
            # }
            # 之后改 加上掩码计算过程看到底谁在起作用 初始化掩码计算过程日志 (放在这里)
                # 初始化队列相关的历史数据
            self.queue_related_names = [
                "charge_queue",          # 充电队列状态
                "queued_vehicles",       # 哪些车辆在排队
                "vehicle_unavail_time",  # 车辆不可用时间
                "charging_vehicles"      # 哪些车辆正在充电
            ]
            self.queue_histories = {
                name: [[] for _ in range(self.batch_size)]
                for name in self.queue_related_names
            }
            # 初始化掩码计算日志
            self.queue_calc_log = [[] for _ in range(self.batch_size)]
        
            self.mask_calc_log = [[] for _ in range(self.batch_size)]
            self.vehicle_batt_history = [[[] for __ in range(self.num_vehicles)] 
                                    for _ in range(self.batch_size)]
            # self.loc_batt_history = [[[] for __ in range(self.num_locs)] 
            #                         for _ in range(self.batch_size)]
            self.time_history = [[] for _ in range(self.batch_size)]
            #self.down_history = [[] for _ in range(self.batch_size)]
                    # 可视化初始状态
            all_batch = torch.full((self.batch_size, ), True, device=self.device)
            self.visualize_state_batch(all_batch)
            self.queue_cost_log = []
            # visualize initial state
        #----------------------
        # common dynamic state
        #----------------------
        self.next_vehicle_id = torch.zeros(self.batch_size, dtype=torch.long, device=device)
        next_vehicle_mask = torch.arange(self.num_vehicles).to(self.device).unsqueeze(0).expand(self.batch_size, -1).eq(self.next_vehicle_id.unsqueeze(-1))
        
        # 现在所有必要的属性都已初始化，可以安全调用update_mask
        self.mask = self.update_mask(self.vehicle_position_id[next_vehicle_mask], next_vehicle_mask)
        
        # 可视化初始状态
        if self.fname is not None:
            all_batch = torch.full((self.batch_size, ), True, device=self.device)
            self.visualize_state_batch(all_batch)
            self.queue_cost_log = []  # 存储(时间点, charge_queue状态)

# # 对于batch_size=2的情况，数据结构如下：

# self.action_histories = {
#     "curr_vehicle_id": [
#         [0, 1, 2],     # 批次0的车辆ID历史
#         [1, 0, 2]      # 批次1的车辆ID历史
#     ],
#     # ... 其他动作历史
# }

# self.mask_histories = {
#     "loc_is_down": [
#         [True, False, True],    # 批次0的基站状态
#         [False, False, True]    # 批次1的基站状态
#     ],
#     # ... 其他掩码历史
# }
    def reset(self, input):
        self.__init__(input)

    def get_coordinates(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.LongTensor [batch_size]

        Returns
        -------
        coords: torch.FloatTensor [batch_size x coord_dim]
        """
        return self.coords.gather(1, node_id[:, None, None].expand(self.batch_size, 1, self.coord_dim)).squeeze(1)
    
    def is_depot(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.Tensor [batch_size]:
        """
        return node_id.ge(self.num_locs)

    def get_loc_mask(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.Tensor [batch_size]:
        """
        return self.loc_arange_idx.eq(node_id.unsqueeze(-1))

    def get_depot_mask(self, node_id: torch.Tensor):
        """
        Paramters
        ---------
        node_id: torch.Tensor [batch_size]:
        """
        return self.depot_arange_idx.eq(node_id.unsqueeze(-1))
    
    def get_vehicle_mask(self, vehicle_id: torch.Tensor):
        """
        Paramters
        ---------
        vehicle_id: torch.Tensor [batch_size]:
        """
        return self.vehicle_arange_idx.eq(vehicle_id.unsqueeze(-1))

# # 假设有2个批次，每个批次3辆车
# self.next_vehicle_id = [0, 1]  # 批次0选择车辆0，批次1选择车辆1
# self.vehicle_position_id = [
#     [5, 2, 3],  # 批次0的3辆车位置
#     [4, 6, 1]   # 批次1的3辆车位置
# ]

# # get_vehicle_mask生成掩码
# mask = [[True, False, False],   # 批次0标记车辆0
#         [False, True, False]]   # 批次1标记车辆1

# # 返回结果：[5, 6]
# # - 批次0的车辆0在位置5
# # - 批次1的车辆1在位置6
    def get_curr_nodes(self):
        return self.vehicle_position_id[self.get_vehicle_mask(self.next_vehicle_id)]

    def update(self,
               next_node_id: torch.Tensor):
        """
        Paramters
        ---------
        next_node_id: torch.LongTensor [batch_size]
            ids of nodes where the currently selected vehicles visit next
        """
#         关键点: 当策略选择的下一个节点 等于 车辆当前所在的节点时 (curr_node_id == next_node_id)，就触发了 do_wait。

# 这意味着，策略通过选择当前节点作为下一个目标来显式地让车辆执行 "等待" 操作。
# 它标记了哪些车辆当前正处于 "等待" 状态。如果 self.wait_vehicle[b, v] 为 True，表示批次 b 中的车辆 v 正在执行等待操作。这个状态由上面 do_wait 的计算结果更新。

# 增加不可用时间: 当车辆被标记为等待 (self.wait_vehicle 为 True) 时，固定的 self.wait_time 会被加到该车辆的 vehicle_work_time 中，进而影响 vehicle_unavail_time。
# 这些 pre/post 时间似乎代表了车辆每次到达或离开一个节点（基站或充电站）时固定的准备/整理时间，无论是否执行 "等待" 操作。它们在车辆处于特定阶段（如 pre/post 阶段）时被计入不可用时间。

# 而 wait_time 只有在策略显式选择停留在当前节点时才会被触发和计入。
#do_wait是学出来的 是否执行 "等待" 这个动作是由强化学习模型（策略）决定的，具体表现为模型选择车辆当前所在的节点作为下一个目标。当车辆处于等待状态 
# (wait_vehicle 为 True) 时，wait_time 会被计入其不可用时间，并且在此期间车辆不移动，但可能会进行充/放电操作（根据代码逻辑，
# 等待时似乎仍在进行供电/充电）。

        curr_vehicle_id = self.next_vehicle_id # [batch_size]
        curr_vehicle_mask = self.get_vehicle_mask(curr_vehicle_id) # [batch_size x num_vehicles]

        # 新增：标记已访问客户点
        # 找出哪些批次选择了客户点作为目标且未被跳过
        is_loc_target = (next_node_id < self.num_locs) & ~self.skip
        if is_loc_target.any():
            # 获取需要更新状态的批次的索引
            batch_indices = torch.arange(self.batch_size, device=self.device)[is_loc_target]
            # 获取这些批次选择的客户点 ID
            loc_indices = next_node_id[is_loc_target]
            # 更新 self.loc_visited 状态
            self.loc_visited[batch_indices, loc_indices] = True

            # 更新首次访问时间（只记录尚未访问过的节点）
            for b, loc_idx in zip(batch_indices, loc_indices):
                if self.loc_first_visit_time[b, loc_idx] < 0:
                    self.loc_first_visit_time[b, loc_idx] = self.current_time[b]
            
            # 记录充电事件（如果从充电站到客户点）
            for b, v_id in zip(batch_indices, curr_vehicle_id[is_loc_target]):
                prev_node_id = self.vehicle_position_id_prev[b, v_id]
                if prev_node_id >= self.num_locs:  # 如果上一个节点是充电站
                    self.num_charge_events[b] += 1

        # assert (self.vehicle_phase[curr_vehicle_mask] == self.phase_id["post"]).sum() == self.batch_size, "all sected vehicles should be in post phase"
# self.next_vehicle_id = [0, 1]  # 批次0处理车辆0，批次1处理车辆1
# # 假设有2个批次(batch_size=2)，每个批次3辆车(num_vehicles=3)
# self.wait_vehicle = [
#     [False, False, False],  # batch 0的3辆车状态
#     [False, False, False]   # batch 1的3辆车状态
# ]

# # 假设有2个批次(batch_size=2)，每个批次3辆车(num_vehicles=3)
# self.next_vehicle_id = [0, 2]  # 批次0选择车辆0，批次1选择车辆2

# curr_vehicle_id = self.next_vehicle_id

# # 用于后续操作，如：
# # 1. 更新车辆位置
# self.vehicle_position_id.scatter_(-1, curr_vehicle_id.unsqueeze(-1), next_node_id.unsqueeze(-1))

# # 2. 计算车辆当前位置
# curr_node_id = self.vehicle_position_id.gather(-1, curr_vehicle_id.unsqueeze(-1)).squeeze(-1)

# # 3. 更新车辆状态
# self.vehicle_phase.scatter_(-1, curr_vehicle_id.unsqueeze(-1), self.phase_id["move"])

# curr_node_id =  [5, 3]     # 当前节点
# next_node_id =  [5, 4]     # 下一节点
# self.skip =     [False, False]
# curr_vehicle_id = [0, 1]   # 当前处理的车辆ID

# # 计算do_wait
# do_wait = (curr_node_id == next_node_id) & ~self.skip
# # do_wait = [True, False]  
# # - batch 0：节点相同(5=5)且未跳过，等待
# # - batch 1：节点不同(3≠4)，不等待

# # 更新wait_vehicle
# # batch 0的车辆0设为True(等待)
# # batch 1的车辆1保持False(不等待)
# self.wait_vehicle -> [
#     [True,  False, False],  # batch 0
#     [False, False, False]   # batch 1
# ]
        #-------------------------------------------
        # update currrently selected vehicle's plan
        #-------------------------------------------
        # calculate travel distance & time of the currently selected vehicle
        curr_node_id = self.vehicle_position_id.gather(-1, curr_vehicle_id.unsqueeze(-1)).squeeze(-1) # [batch_size]
        curr_coords = self.get_coordinates(curr_node_id) # [batch_size x coord_dim]
        next_coords = self.get_coordinates(next_node_id) # [batch_size x coord_dim]
        travel_distance = COEF * torch.linalg.norm(curr_coords - next_coords, dim=-1) # [batch_size]
        travel_time = travel_distance / self.speed # [batch_size]

        # check waiting vehicles
        do_wait = (curr_node_id == next_node_id) & ~self.skip # [batch_size] wait: stay at the same place
        # 存储此次动作信息
        self.last_action_info = {
            "curr_vehicle_id": curr_vehicle_id.clone().detach(),
            "next_node_id": next_node_id.clone().detach(),
            "do_wait": do_wait.clone().detach(),
            "travel_time": travel_time.clone().detach(),
            "charge_time": None  # 将在后续计算中填充
        }
        self.wait_vehicle.scatter_(-1, index=curr_vehicle_id.unsqueeze(-1), src=do_wait.unsqueeze(-1))

# # 假设有2个批次(batch_size=2)，每个批次3辆车(num_vehicles=3)
# self.wait_vehicle = [
#     [False, False, False],  # batch 0的3辆车状态
#     [False, False, False]   # batch 1的3辆车状态
# ]

# curr_node_id =  [5, 3]     # 当前节点
# next_node_id =  [5, 4]     # 下一节点
# self.skip =     [False, False]
# curr_vehicle_id = [0, 1]   # 当前处理的车辆ID

# # 计算do_wait
# do_wait = (curr_node_id == next_node_id) & ~self.skip
# # do_wait = [True, False]  
# # - batch 0：节点相同(5=5)且未跳过，等待
# # - batch 1：节点不同(3≠4)，不等待

# # scatter_操作后的结果
# self.wait_vehicle -> [
#     [True,  False, False],  # batch 0的车辆0设为True(等待)
#     [False, False, False]   # batch 1的车辆1保持False(不等待)
# ]
        # update the plan of the selected vehicles
        self.vehicle_unavail_time.scatter_(-1, curr_vehicle_id.unsqueeze(-1), travel_time.unsqueeze(-1))
        self.vehicle_position_id_prev.scatter_(-1, curr_vehicle_id.unsqueeze(-1), curr_node_id.unsqueeze(-1))
        self.vehicle_position_id.scatter_(-1, curr_vehicle_id.unsqueeze(-1), next_node_id.unsqueeze(-1))
        self.vehicle_phase.scatter_(-1, curr_vehicle_id.unsqueeze(-1), self.phase_id["move"])

# # 假设batch_size=2，num_vehicles=3
# curr_vehicle_id = [0, 1]   # 批次0选择车辆0，批次1选择车辆1
# curr_node_id = [5, 3]      # 当前位置
# next_node_id = [7, 4]      # 目标位置
# travel_time = [0.5, 0.3]   # 移动时间

# # 更新前：
# vehicle_unavail_time = [
#     [0.0, 0.0, 0.0],  # batch 0
#     [0.0, 0.0, 0.0]   # batch 1
# ]

# # 更新后：
# vehicle_unavail_time = [
#     [0.5, 0.0, 0.0],  # batch 0的车辆0设为0.5
#     [0.0, 0.3, 0.0]   # batch 1的车辆1设为0.3
# ]

# # 同样地，位置和状态也会更新：
# vehicle_position_id_prev: 记录[5, 3]作为前一个位置
# vehicle_position_id: 更新为新位置[7, 4]
# vehicle_phase: 相应车辆设置为移动状态(0)
        #---------------------------
        # estimate store phase time
        #---------------------------
        # moving 
        self.vehicle_move_time.scatter_(-1, curr_vehicle_id.unsqueeze(-1), travel_time.unsqueeze(-1)) # [batch_size x num_vehicles]
        
        # pre/post-operation time
        at_depot = self.is_depot(next_node_id).unsqueeze(-1)
        at_loc = ~at_depot
        curr_vehicle_at_loc = curr_vehicle_mask & at_loc
        curr_vehicle_at_depot = curr_vehicle_mask & at_depot
        self.vehicle_pre_time  += curr_vehicle_at_loc * self.pre_time_loc + curr_vehicle_at_depot * self.pre_time_depot
        self.vehicle_post_time += curr_vehicle_at_loc * self.post_time_loc + curr_vehicle_at_depot * self.post_time_depot
         # 之后改 考虑这里的pre 和post不是固定的还有可能是计算得来的
        #         # 假设:
        # curr_vehicle_at_loc = [
        #     [True, False, False],   # batch 0: 车辆0在基站
        #     [False, True, False]    # batch 1: 车辆1在基站
        # ]
        
        # curr_vehicle_at_depot = [
        #     [False, True, False],   # batch 0: 车辆1在充电站
        #     [True, False, False]    # batch 1: 车辆0在充电站
        # ]
        
        # 结果:
        # batch 0:
        # - 车辆0: 增加基站准备时间(0.5小时)
        # - 车辆1: 增加充电站准备时间(0.17小时)
        
        # batch 1:
        # - 车辆0: 增加充电站准备时间(0.17小时)
        # - 车辆1: 增加基站准备时间(0.5小时)
        # charge/supply time
        destination_loc_mask = self.get_loc_mask(next_node_id) # [batch_size x num_locs]
        #-------------------------------------
        # supplying time (visiting locations)
        #-------------------------------------
        unavail_depots = self.get_unavail_depots2(next_node_id).unsqueeze(-1).expand_as(self.depot_coords)
        depot_coords = self.depot_coords + 1e+6 * unavail_depots
        loc2depot_min = COEF * torch.linalg.norm(self.get_coordinates(next_node_id).unsqueeze(1) - depot_coords, dim=-1).min(-1)[0] # [batch_size] 
        # discharge_lim = torch.maximum(loc2depot_min.unsqueeze(-1) * self.vehicle_consump_rate, self.vehicle_discharge_lim) # [batch_size x num_vehicles]
        # 之后改 我找不到这里对应论文的地方
        #逻辑设计 保护车辆能够到最近的充电站 但是这里是到loc不是到base
        # 之后改 
        # veh_discharge_lim = (self.vehicle_curr_battery - (travel_distance.unsqueeze(-1) * self.vehicle_consump_rate) - discharge_lim).clamp(0.0)
        # demand_on_arrival = torch.minimum(((self.loc_cap - (self.loc_curr_battery - self.loc_consump_rate * (travel_time.unsqueeze(-1) + self.pre_time_loc)).clamp(0.0)) * destination_loc_mask).sum(-1, keepdim=True), 
        #                                    veh_discharge_lim) # [batch_size x num_vehicles]
        # 之后改 之后看不懂了 再过来看看
        #         # 假设场景:
        # loc_cap = [100, 100, 100]           # 基站容量
        # loc_curr_battery = [80, 70, 90]     # 基站当前电量
        # loc_consump_rate = [5, 4, 6]        # 基站耗电速率
        # travel_time = 0.5                   # 行驶时间
        # pre_time_loc = 0.2                  # 准备时间
        # destination_loc_mask = [1, 0, 0]    # 目标是第一个基站
        # veh_discharge_lim = 50              # 车辆最大放电量
        
        # # 1. 计算到达时基站电量
        # battery_consumption = [5*(0.5+0.2), 4*(0.5+0.2), 6*(0.5+0.2)]
        # # = [3.5, 2.8, 4.2]
        # arrival_battery = [76.5, 67.2, 85.8]
        
        # # 2. 计算需求量
        # demand = [23.5, 32.8, 14.2]
        # masked_demand = [23.5, 0, 0]  # 只考虑目标基站
        # total_demand = 23.5
        
        # # 3. 最终供应量
        # demand_on_arrival = min(23.5, 50) = 23.5
        # split supplying TODO: need clippling ?

        # charge_time_tmp = demand_on_arrival / (self.vehicle_discharge_rate - (self.loc_consump_rate * destination_loc_mask).sum(-1, keepdim=True)) # [batch_sizee x num_vehicles]
        # cannot_supplly_full = ((veh_discharge_lim - charge_time_tmp * self.vehicle_discharge_rate) < 0.0) # [batch_size x num_vehicles]
        # next_vehicles_sd  = curr_vehicle_at_loc & cannot_supplly_full  # vehicles that do split-delivery [batch_size x num_vehicles]
        # next_vehicles_nsd = curr_vehicle_at_loc & ~cannot_supplly_full # vehicles that do not split-delivery [batch_size x num_vehicles]
#         sd: split-delivery (分批供电)
# nsd: non-split-delivery (完整供电)
# 假设场景：
# demand_on_arrival = 50      # 基站需要50kWh电量
# vehicle_discharge_rate = 20  # 车辆放电速率20kW
# loc_consump_rate = 5        # 基站消耗速率5kW
# veh_discharge_lim = 30      # 车辆可放电量30kWh

# # 1. 计算理想充电时间
# charge_time_tmp = 50 / (20 - 5) = 3.33小时

# # 2. 检查是否能完全充电
# needed_power = 3.33 * 20 = 66.6kWh > 30kWh(可用电量)
# # 因此需要分批供电

# # 3. 实际充电时间
# 分批供电时间 = 30 / 20 = 1.5小时  # 用完所有可用电量
        # charge_time = (charge_time_tmp * next_vehicles_nsd).sum(-1) # [batch_size]
        # charge_time += ((veh_discharge_lim / self.vehicle_discharge_rate) * next_vehicles_sd).sum(-1) # [batch_size]
        #---------------------------------
        # charging time (visiting depots)
        #---------------------------------
        charge_time = torch.zeros_like(travel_time)  # 初始化充电时间
        curr_depot_mask = self.get_depot_mask(next_node_id) # [batch_size x num_depots]
        charge_time += (((self.vehicle_cap - (self.vehicle_curr_battery - (travel_distance.unsqueeze(-1) * self.vehicle_consump_rate)).clamp(0.0)) / ((self.depot_discharge_rate * curr_depot_mask).sum(-1, keepdim=True) + SMALL_VALUE)) * curr_vehicle_at_depot).sum(-1) # charge time for split supplying (loc will not be fully [charged)
        # 在计算完 charge_time 后更新 last_action_info
        self.last_action_info["charge_time"] = charge_time.clone().detach()
        #--------------------------------------------------------
        # update unavail_time (charge_time) of selected vehicles
        #--------------------------------------------------------
        self.vehicle_work_time += charge_time.clamp(0.0).unsqueeze(-1) * (curr_vehicle_mask & ~self.wait_vehicle) # [charging_batch_size]
        self.vehicle_work_time += self.wait_time * (curr_vehicle_mask & self.wait_vehicle) # waiting vehicles
        
        #----------------------------------------------------------------------
        # select a vehicle that we determine its plan while updating the state
        # (greddy approach: select a vehicle whose unavail_time is minimum)
        #----------------------------------------------------------------------
        # align the phase of the selected vehicles to "post"
        num_not_post = 1 # temporaly initial value
        while num_not_post > 0:
            vechicle_unavail_time_min, next_vehicle_id = self.vehicle_unavail_time.min(dim=-1) # [batch_size], [batch_size]
            next_vehicle_mask = self.get_vehicle_mask(next_vehicle_id) # [batch_size x num_vehicles]
            not_post_batch = (self.vehicle_phase[next_vehicle_mask] != self.phase_id["post"]) # [batch_size]
            num_not_post = (not_post_batch).sum()
            if num_not_post > 0:
                self.update_state(vechicle_unavail_time_min, self.vehicle_position_id[next_vehicle_mask], next_vehicle_id, next_vehicle_mask, not_post_batch, align_phase=True)
        # now, all the vehicle selected in all the batchs should be in charge phase
        # update the state at the time when the selected vehicles finish charging
        vechicle_unavail_time_min, next_vehicle_id = self.vehicle_unavail_time.min(dim=-1) # [batch_size], [batch_size]
        self.next_vehicle_id = next_vehicle_id
        #保存选中的车辆ID，用于后续处理
        next_vehicle_mask = self.get_vehicle_mask(next_vehicle_id) # [batch_size x num_vehicles]
        #成一个掩码，标识被选中的车辆
        next_node_id = self.vehicle_position_id[next_vehicle_mask]
        #获取被选中车辆当前所在的节点ID
        all_batch = torch.full((self.batch_size, ), True, device=self.device)
        #创建一个全True的张量，表示更新所有批次
        self.update_state(vechicle_unavail_time_min, next_node_id, next_vehicle_id, next_vehicle_mask, all_batch, align_phase=False)

        #-------------
        # update mask
        #-------------
        self.mask = self.update_mask(next_node_id, next_vehicle_mask)

        #--------------------------
        # validation check of mask
        #--------------------------
        #以防万一
    
        all_zero = self.mask.sum(-1) == 0 # [batch_size]
        assert not all_zero.any(), "there is no node that the vehicle can visit!"

    def update_state(self, 
                     elapsed_time: torch.Tensor,
                     next_node_id: torch.Tensor,
                     next_vehicle_id: torch.Tensor, 
                     next_vehicle_mask: torch.Tensor, 
                     update_batch: torch.Tensor,
                     align_phase: bool):
        """
        Parameters
        ----------
        elapsed_time: torch.FloatTensor [batch_size]
        next_node_id: torch.LongTensor [batch_size]
        next_vehicle_id: torch.LongTensor [batch_size]
        next_vehicle_mask: torch.BoolTensor [batch_size x num_vehicles]
        update_batch: torch.BoolTensor [batch_size]
        align_phase: bool
        """
        self.last_action_info["elapsed_time"] = elapsed_time.clone().detach()
            # 首先定义queued_vehicles和update_vehicles
        _charge_phase = (self.vehicle_phase == self.phase_id["charge"])
        _at_depot = self.is_depot(self.vehicle_position_id)
        queued_vehicles = (self.charge_queue.sum(1) > 1)  # [batch_size x num_vehicles]
        update_vehicles = ~queued_vehicles & update_batch.unsqueeze(-1)  # [batch_size x num_vehicles]
    
        #之后改
        batch = 0  # 主要记录第一个批次的详情 
        current_step_log = None
        if self.fname is not None and hasattr(self, 'queue_calc_log') and update_batch[batch]:
            current_step_log = {
                "step": self.episode_step,
                "time": self.current_time[batch].item(),
                "elapsed_time": elapsed_time[batch].item(),
                "align_phase": align_phase,
                "selected_vehicle": next_vehicle_id[batch].item(),
                "vehicle_position": self.vehicle_position_id[batch, next_vehicle_id[batch]].item(),
                "next_node": next_node_id[batch].item(),
                "queue_state_before": {},
                "queue_state_after": {},
                "calculations": []
            }
            
            # 记录更新前的队列状态
            current_step_log["queue_state_before"]["charge_queue"] = self.charge_queue[batch].clone().detach().cpu().tolist()
            current_step_log["queue_state_before"]["vehicle_phase"] = self.vehicle_phase[batch].clone().detach().cpu().tolist()
            current_step_log["queue_state_before"]["vehicle_position_id"] = self.vehicle_position_id[batch].clone().detach().cpu().tolist()
            current_step_log["queue_state_before"]["vehicle_unavail_time"] = self.vehicle_unavail_time[batch].clone().detach().cpu().tolist()

            # 现在update_vehicles已经定义，可以在日志中使用它
            if current_step_log is not None:
                current_step_log["time_update"] = {
                    "update_vehicles_mask": update_vehicles[batch].cpu().tolist(),
                    "elapsed_time": elapsed_time[batch].item()
                }

        #-------------------
        # clip elapsed_time
        #-------------------
        remaing_time = (self.time_horizon - self.current_time).clamp(0.0) # [batch_size]
        elapsed_time = torch.minimum(remaing_time, elapsed_time)
        #         # 假设场景：
        # time_horizon = 10.0  # 总时间限制为10小时
        # current_time = 8.5   # 当前已用时间8.5小时
        # original_elapsed_time = 2.0  # 需要2小时完成操作
        
        # # 1. 计算剩余时间
        # remaining_time = (10.0 - 8.5).clamp(0.0) = 1.5  # 剩余1.5小时
        
        # # 2. 限制实际可用时间
        # actual_elapsed_time = min(1.5, 2.0) = 1.5  # 只能使用1.5小时
        #---------------------------------------------
        # moving vehicles (consuming vehicle battery)
        #---------------------------------------------
        moving_vehicles = (self.vehicle_phase == self.phase_id["move"]) & update_batch.unsqueeze(-1) # [batch_size x num_vehicles]
        moving_not_wait_vehicles = moving_vehicles & ~self.wait_vehicle # [batch_size x num_vehicles]
        self.vehicle_curr_battery -= self.vehicle_consump_rate * (self.speed * elapsed_time).unsqueeze(-1) * moving_not_wait_vehicles # Travel battery consumption
        # update total tour length
        # self.tour_length[update_batch] += moving_vehicles.sum(-1)[update_batch] * self.speed[update_batch] * elapsed_time[update_batch]
        self.tour_length += moving_vehicles.sum(-1) * self.speed * elapsed_time * update_batch.float()

        #-------------------------------
        # charging / supplying vehicles 
        #-------------------------------
        at_depot = self.is_depot(self.vehicle_position_id) # [batch_size x num_vehicles]
        charge_phase_vehicles = (self.vehicle_phase == self.phase_id["charge"]) # [batch_size x num_vehicles]
                
        # --- 修改：计算充电和排队车辆 ---
        currently_charging = torch.full_like(at_depot, False, device=self.device)
        currently_queued = torch.full_like(at_depot, False, device=self.device)

        for b in range(self.batch_size):
            if not update_batch[b]:
                continue
                
            for d_idx in range(self.num_depots):
                depot_id = self.num_locs + d_idx
                
                # 查找当前在该充电站且处于充电阶段的车辆
                vehicles_at_depot = torch.where(
                    (self.vehicle_position_id[b] == depot_id) & 
                    charge_phase_vehicles[b]
                )[0]

                # 记录计算过程
                if current_step_log is not None and b == batch:
                    calc_info = {
                        "depot_id": depot_id,
                        "depot_idx": d_idx,
                        "vehicles_at_depot": vehicles_at_depot.cpu().tolist()
                    }
                    current_step_log["calculations"].append(calc_info)
                
                if len(vehicles_at_depot) > 0:
                    # 获取这些车辆的队列号
                    queue_numbers = self.charge_queue[b, d_idx, vehicles_at_depot]

                    # 记录队列号
                    if current_step_log is not None and b == batch:
                        calc_info["queue_numbers"] = queue_numbers.cpu().tolist()
                    # 找出队列号最小的有效车辆（队列号>0）
                    valid_indices = queue_numbers > 0
                    if valid_indices.any():
                        valid_vehicles = vehicles_at_depot[valid_indices]
                        valid_queue_nums = queue_numbers[valid_indices]

                        # 记录有效车辆和队列号
                        if current_step_log is not None and b == batch:
                            calc_info["valid_vehicles"] = valid_vehicles.cpu().tolist()
                            calc_info["valid_queue_nums"] = valid_queue_nums.cpu().tolist()
                            
                        # 获取最小队列号和对应车辆
                        min_idx = torch.argmin(valid_queue_nums)
                        min_vehicle = valid_vehicles[min_idx]
                        min_queue = valid_queue_nums[min_idx]

                        # 记录选择结果
                        if current_step_log is not None and b == batch:
                            calc_info["min_vehicle"] = min_vehicle.item()
                            calc_info["min_queue"] = min_queue.item()
                            calc_info["charging_vehicle"] = min_vehicle.item()
                            calc_info["queued_vehicles"] = []
                        # 只有最小队列号的车辆在充电
                        currently_charging[b, min_vehicle] = True
                        
                        # 其他车辆在排队
                        for v_idx, v in enumerate(valid_vehicles):
                            if valid_queue_nums[v_idx] > min_queue:
                                currently_queued[b, v] = True
                                # 记录排队车辆
                                if current_step_log is not None and b == batch:
                                    calc_info["queued_vehicles"].append(v.item())
        
        # --- 修改：实际的充电和排队车辆用于更新操作 ---
        charging_vehicles = charge_phase_vehicles & at_depot & currently_charging & update_batch.unsqueeze(-1)
        queued_vehicles = charge_phase_vehicles & at_depot & currently_queued & update_batch.unsqueeze(-1)

         # 记录计算结果
        if current_step_log is not None:
            current_step_log["charging_vehicles_mask"] = charging_vehicles[batch].cpu().tolist()
            current_step_log["queued_vehicles_mask"] = queued_vehicles[batch].cpu().tolist()
        #-----------------------------
        # charging (depot -> vehicle)
        #-----------------------------
     #之后改 这里重新定义了排队状态
     #   queued_vehicles = self.charge_queue.sum(1) > 1 # [batch_size x num_vehicles]
        
        # > 1 这个条件意味着，只有当车辆在某个充电站的到达顺序号是 2 或更高时，它才被认为是 queued_vehicles
    #之后改 这里重新定义了充电状态
    #     charging_vehicles = charge_phase_vehicles & at_depot & update_batch.unsqueeze(-1) & ~queued_vehicles # [batch_size x num_vehicles]
        # 只有非排队车辆的 vehicle_unavail_time 会随着 elapsed_time 减少。排队车辆的时间暂停。
        # 只有非排队车辆（且在充电站、处于充电阶段）才能实际进行充电 (self.vehicle_curr_battery 增加）。
        charging_vehicle_position_idx = (self.vehicle_position_id - self.num_locs) * charging_vehicles.long()
        self.vehicle_curr_battery += self.depot_discharge_rate.gather(-1, charging_vehicle_position_idx) * elapsed_time.unsqueeze(-1) * charging_vehicles.float() # [batch_size x num_vehicles]
        #---------------------------------
        # supplying (vehicle -> location)
        #---------------------------------
        # supplying_vehicles = charge_phase_vehicles & ~at_depot & update_batch.unsqueeze(-1) # [batch_size x num_vehicles]
        # location battery charge
        # NOTE: In locs where a vehicle is staying, the battery of the locs incrases by loc_consump_rate * elapsed_time, not vehicle_discarge_rate * elasped_time.
        # 之后改 这里有点看不懂
        # However, as the battery of the locs should be full when a vehicle is staying and it is clamped by max_cap later, we ignore this mismatch here.
        # supplying_vehicle_position_idx = self.vehicle_position_id * supplying_vehicles.long() # [batch_size x num_vehicles]
        # self.loc_curr_battery.scatter_reduce_(-1, 
                                                # supplying_vehicle_position_idx, 
                                                # self.vehicle_discharge_rate * elapsed_time.unsqueeze(-1) * supplying_vehicles.float(), 
                                                # reduce="sum")
        # vechicle battery consumption (consumption rate is different b/w waiting vehicles and not waiting ones)
        # 之后改排队  这里按照是否排队算电量变化
        # #         # 假设场景：batch_size=2，num_vehicles=3
        # supplying_vehicles = [
        #     [True, False, True],    # batch 0
        #     [True, True, False]     # batch 1
        # ]
        
        # self.wait_vehicle = [
        #     [True, False, False],   # batch 0
        #     [False, True, False]    # batch 1
        # ]
        
        # # 1. 非等待车辆
        # supplying_not_wait_vehicles = [
        #     [False, False, True],   # batch 0: 车辆2
        #     [True, False, False]    # batch 1: 车辆0
        # ]
        
        # # 2. 等待车辆
        # supplying_wait_vehicles = [
        #     [True, False, False],   # batch 0: 车辆0
        #     [False, True, False]    # batch 1: 车辆1
        # ]
        
        # 电量更新
        # 非等待车辆：按vehicle_discharge_rate消耗
        # 等待车辆：按loc_consump_rate消耗
        # not waiting
          # 识别非等待的供电车辆
        # supplying_not_wait_vehicles = supplying_vehicles & ~self.wait_vehicle 
        
        # 更新电量
        # self.vehicle_curr_battery -= (
        #     self.vehicle_discharge_rate *                  # 车辆放电速率
        #     elapsed_time.unsqueeze(-1) *                  # 经过的时间
        #     supplying_not_wait_vehicles.float()           # 掩码转换为浮点数
        # )      # waiting
        #       # 识别等待的供电车辆
        # supplying_wait_vehicles = supplying_vehicles & self.wait_vehicle
        
        # 获取等待车辆所在位置的索引
        # supplying_vehicle_position_idx_ = (
        #     self.vehicle_position_id *                    # 车辆位置ID
        #     supplying_wait_vehicles.long()                # 掩码转为整数
        # )
        
        # # 更新电量
        # self.vehicle_curr_battery -= (
        #     self.loc_consump_rate.gather(                 # 获取对应基站的耗电速率
        #         -1, 
        #         supplying_vehicle_position_idx_
        #     ) * 
        #     elapsed_time.unsqueeze(-1) *                 # 经过的时间
        #     supplying_wait_vehicles.float()              # 掩码转换为浮点数
        # )   
        #----------------------------------
        # battery consumption of locations
        #----------------------------------
        # 之后改 删除looc电量变化
        # self.loc_curr_battery -= self.loc_consump_rate * (elapsed_time * update_batch.float()).unsqueeze(-1)
        
        # TODO:
        # print(self.vehicle_curr_battery[self.vehicle_curr_battery<0])
        # location battery is always greater (less) than 0 (capacity)
        self.vehicle_curr_battery = self.vehicle_curr_battery.clamp(min=0.0)
        self.vehicle_curr_battery = self.vehicle_curr_battery.clamp(max=self.vehicle_cap)

        #----------------
        # update penalty
        #----------------
        #down_locs = (self.loc_curr_battery - self.loc_min_battery) <= 0.0 # SMALL_VALUE [batch_size x num_locs]
        # 我也需要记录这个值 之后改 衡量两种算法下有什么不同
        # ignore penalty in skipped episodes
        #num_empty_locs = ((-self.loc_curr_battery + self.loc_min_battery) / self.loc_consump_rate) * down_locs * (~self.skip.unsqueeze(-1)) # [batch_size x num_locs]
       
    #        num_empty_locs = (
    #     (-self.loc_curr_battery + self.loc_min_battery) /  # 计算低于最小电量的差值
    #     self.loc_consump_rate                              # 除以耗电速率得到时间
    # ) * down_locs                                          # 只考虑电量耗尽的基站
    #   * (~self.skip.unsqueeze(-1))                        # 排除被跳过的批次
        # empty_locs = ((-self.loc_curr_battery + self.loc_min_battery)[down_locs] / self.loc_consump_rate[down_locs]) # 1d
        # num_empty_locs = torch.zeros((self.batch_size, self.num_locs), dtype=torch.float, device=self.device).masked_scatter_(down_locs, empty_locs)
        # num_empty_locs[self.skip] = 0.0 # ignore penalty in skipped episodes
        # self.penalty_empty_locs += num_empty_locs.sum(-1) * update_batch / self.num_locs # [batch_size]
        # 计算充电站冲突成本
        _charge_phase = (self.vehicle_phase == self.phase_id["charge"])
        _at_depot = self.is_depot(self.vehicle_position_id)
        queued_vehicles = (self.charge_queue.sum(1) > 1)
        num_waiting_vehicles = (queued_vehicles & _charge_phase & _at_depot).sum(dim=1).float()
        self.accumulated_conflict_cost += num_waiting_vehicles * elapsed_time * update_batch.float()
    #    #    self.penalty_empty_locs += (
    #     num_empty_locs.sum(-1) *     # 所有基站的电量不足时间总和
    #     update_batch /               # 只更新指定批次
    #     self.num_locs               # 归一化（除以基站数量）
    # )
        # location battery is always greater (less) than minimum battery (capacity)
        #self.loc_curr_battery = self.loc_curr_battery.clamp(min=self.loc_min_battery)
        #self.loc_curr_battery = self.loc_curr_battery.clamp(max=self.loc_cap)
        #         # 确保电量不低于最小值
        # self.loc_curr_battery = self.loc_curr_battery.clamp(min=self.loc_min_battery)
        # # 确保电量不超过容量上限
        # self.loc_curr_battery = self.loc_curr_battery.clamp(max=self.loc_cap)
# # 假设场景:
# loc_curr_battery = [10, 5, 0]      # 当前电量
# loc_min_battery = 20               # 最小电量要求
# loc_consump_rate = [2, 2, 2]      # 耗电速率
# down_locs = [True, True, True]    # 所有基站都电量不足
# skip = [False]                    # 批次未被跳过

# # 计算过程:
# # 1. 电量差值: 
# (-[10, 5, 0] + 20) = [10, 15, 20]

# # 2. 除以耗电速率:
# [10, 15, 20] / [2, 2, 2] = [5, 7.5, 10]  # 电量不足持续时间

# # 3. 应用down_locs掩码:
# [5, 7.5, 10] * [True, True, True] = [5, 7.5, 10]

# # 4. 应用skip掩码:
# 最终结果: [5, 7.5, 10]  # 因为批次未被跳过，保留所有值
        #之后改  记录时间更新信息
        if current_step_log is not None:
            current_step_log["time_update"] = {
                "update_vehicles_mask": update_vehicles[batch].cpu().tolist(),
                "elapsed_time": elapsed_time[batch].item()
            }
        #---------------------
        # update unavail_time
        #---------------------
        # decrease unavail_time
            #  之后改 修改：只有非排队车辆的时间会流逝
       # queued_vehicles = self.charge_queue.sum(1) > 1 # [batch_size x num_vehicles]
        update_vehicles = ~queued_vehicles & update_batch.unsqueeze(-1) # [batch_size x num_vehicles]
        self.vehicle_unavail_time -= elapsed_time.unsqueeze(-1) * update_vehicles
        self.vehicle_unavail_time = self.vehicle_unavail_time.clamp(min=0.0)

                # 计算充电站冲突成本，添加在更新时间之前
        # 检查每个充电站是否有车辆排队（queue number > 1）
        _charge_phase = (self.vehicle_phase == self.phase_id["charge"])
        _at_depot = self.is_depot(self.vehicle_position_id)
        queued_vehicles = (self.charge_queue.sum(1) > 1)  # [batch_size × num_vehicles]
        # 计算每个批次中排队等待的车辆总数
        num_waiting_vehicles = (queued_vehicles & _charge_phase & _at_depot).sum(dim=1).float()  # [batch_size]
        # 累加冲突成本：等待车辆数 × 经过时间
        self.accumulated_conflict_cost += num_waiting_vehicles * elapsed_time * update_batch.float()


        #---------------------
        # update current time
        #---------------------
        # 在 elapsed_time 应用后，更新指标（约在行 1150 附近）
        self.current_time += elapsed_time * update_batch


        # 更新与时间相关的指标
        for b in range(self.batch_size):
            if not update_batch[b]:
                continue
                
            # 更新行驶时间和能量
            travel_time_b = (moving_vehicles[b] & ~self.wait_vehicle[b]).sum().float() * elapsed_time[b]
            self.total_travel_time[b] += travel_time_b
            self.total_travel_energy[b] += travel_time_b * self.vehicle_consump_rate.mean()
            
            # 更新供电时间和能量
            #supply_time_b = supplying_vehicles[b].sum().float() * elapsed_time[b]
            # self.total_service_time[b] += supply_time_b
            # self.total_supply_energy[b] += supply_time_b * self.vehicle_discharge_rate.mean()
            
            # 更新充电时间和能量
            charge_time_b = charging_vehicles[b].sum().float() * elapsed_time[b]
            self.total_charge_time[b] += charge_time_b
            self.total_charge_energy[b] += charge_time_b * self.depot_discharge_rate.mean()
            
            # 更新等待时间
            wait_time_b = (self.wait_vehicle[b]).sum().float() * elapsed_time[b]
            self.total_wait_time[b] += wait_time_b
            
            # 更新充电站忙碌时间
            for d_idx in range(self.num_depots):
                depot_id = self.num_locs + d_idx
                busy = ((self.vehicle_position_id[b] == depot_id) & charging_vehicles[b]).any()
                if busy:
                    self.depot_busy_time[b, d_idx] += elapsed_time[b]
            
            # 更新队列相关指标
            current_queue_length = (queued_vehicles[b]).sum().item()
            self.sum_queue_length_time[b] += current_queue_length * elapsed_time[b]
            self.max_queue_length[b] = max(self.max_queue_length[b], current_queue_length)
            
            # 更新车辆排队时间
            for v_idx in range(self.num_vehicles):
                if queued_vehicles[b, v_idx]:
                    self.vehicle_queuing_time[b, v_idx] += elapsed_time[b]

        #---------------------
        # visualize the state
        #---------------------
        if self.fname is not None:
            # 确定需要可视化的批次：
            vis_batch = update_batch & (torch.abs(elapsed_time) > SMALL_VALUE)#之后改 这里删除了～skip
            self.visualize_state_batch(vis_batch)
        # update unavail time
        if align_phase:
            # 检查是否在充电站
            at_depot = self.is_depot(next_node_id).unsqueeze(-1)  # [batch_size x 1]
            # 获取需要更新的车辆掩码
            next_vehicles_on_update_batch = next_vehicle_mask & update_batch.unsqueeze(-1)
            # 清零这些车辆的不可用时间
            self.vehicle_unavail_time *= ~next_vehicles_on_update_batch
            #-------------------------
            # moving -> pre operation: 
            #-------------------------
            if current_step_log is not None:
                current_step_log["queue_update_align"] = {
                    "operation": "入队",
                    "details": []
                }
            # 之后改--- 修改：车辆到达充电站入队逻辑 ---
            next_vehicle_on_move = next_vehicles_on_update_batch & (self.vehicle_phase == self.phase_id["move"])
            
            if next_vehicle_on_move.sum() > 0:
                # 添加准备时间
                self.vehicle_unavail_time += self.vehicle_pre_time * next_vehicle_on_move
                # 重置移动时间
                self.vehicle_move_time *= ~next_vehicle_on_move
            
                # 处理充电队列
                head = self.charge_queue.min(-1)[0]  # 获取每个充电站的最小队列号
                destination_depot_mask = self.get_depot_mask(next_node_id)  # 目标充电站掩码
                # 之后改            # 车辆在充电站的掩码
                for b in range(self.batch_size):
                    for d_idx in range(self.num_depots):
                        if update_batch[b] and destination_depot_mask[b, d_idx]:
                            # 获取该充电站当前最大队列号
                            current_max = torch.max(self.charge_queue[b, d_idx])
                            # 为新到达车辆分配队列号
                            arriving_vehicle = next_vehicle_id[b]
                            # 记录入队信息
                            if current_step_log is not None and b == batch:
                                update_info = {
                                    "depot_id": self.num_locs + d_idx,
                                    "arriving_vehicle": arriving_vehicle.item(),
                                    "waiting": self.wait_vehicle[b, arriving_vehicle].item(),
                                    "current_max_queue": current_max.item()
                                }
                                current_step_log["queue_update_align"]["details"].append(update_info)
                            
                            if not self.wait_vehicle[b, arriving_vehicle]:  # 只有非等待车辆才入队
                                new_queue_num = current_max + 1
                                self.charge_queue[b, d_idx, arriving_vehicle] = new_queue_num
                                
                                # 记录分配的队列号
                                if current_step_log is not None and b == batch:
                                    update_info["assigned_queue"] = new_queue_num.item()
                                
                #之后改 这里出现了编号跳跃
                #  创建更新队列的掩码
                #update_query_mask = (~self.wait_vehicle & next_vehicle_on_move).unsqueeze(1) & destination_depot_mask.unsqueeze(-1)
                
                # 更新充电队列
                #self.charge_queue += (head + 1).unsqueeze(-1) * update_query_mask

            #---------------------------
            # pre operation -> charging
            #---------------------------
            next_vehicle_on_pre = next_vehicles_on_update_batch & (self.vehicle_phase == self.phase_id["pre"]) # [batch_size x num_vehicles]
            if next_vehicle_on_pre.sum() > 0:
                self.vehicle_unavail_time += self.vehicle_work_time * next_vehicle_on_pre
                self.vehicle_pre_time *= ~next_vehicle_on_pre # reset pre time
            
            #----------------------------
            # charging -> post operation
            #----------------------------
            # charging -> post operation
            next_vehicle_on_charge = next_vehicles_on_update_batch & (self.vehicle_phase == self.phase_id["charge"])
            if next_vehicle_on_charge.sum() > 0:
                # 添加后处理时间
                self.vehicle_unavail_time += self.vehicle_post_time * next_vehicle_on_charge
                # 重置工作时间
                self.vehicle_work_time *= ~next_vehicle_on_charge
            #--------------
            # update phase
            #--------------
            self.vehicle_phase += next_vehicles_on_update_batch.long()
            #将选中车辆的阶段向前推进一步
        else:
            #----------------------------------------------
            # post operation -> move (determine next node)
            #----------------------------------------------
            # update charge-queue
            # do not change charge queue when the next vehicle is waiting
            # because the waiting vehicles are not added to the queue
                   # 更新充电队列
            #之后改 --- 修改：处理车辆离开充电站时的出队逻辑 ---


            if current_step_log is not None:
                current_step_log["queue_update_not_align"] = {
                    "operation": "出队",
                    "details": []
                }

            destination_depot_mask = (
                self.get_depot_mask(next_node_id) & 
                ~self.wait_vehicle[next_vehicle_mask].unsqueeze(-1)
            )

            # 之后改            # 将离开的车辆队列号置零
            for b in range(self.batch_size):
                if update_batch[b]:
                    veh_id = next_vehicle_id[b]
                    for d_idx in range(self.num_depots):
                        if destination_depot_mask[b, d_idx]:
                            # 记录出队信息
                            if current_step_log is not None and b == batch:
                                update_info = {
                                    "depot_id": self.num_locs + d_idx,
                                    "leaving_vehicle": veh_id.item(),
                                    "previous_queue": self.charge_queue[b, d_idx, veh_id].item()
                                }
                                current_step_log["queue_update_not_align"]["details"].append(update_info)
                            
                            # 重置队列号
                            self.charge_queue[b, d_idx, veh_id] = 0

            # 减少充电队列计数
            self.charge_queue -= destination_depot_mask.long().unsqueeze(-1)
            self.charge_queue = self.charge_queue.clamp(0)
           # 这相当于将它的队列号移除或减少。（注意：这里直接减 1 可能不完全精确地反映队列移动，但目的是将其标记为不再排队）。
            
            # 重置等待标志
            self.wait_vehicle.scatter(
                -1, 
                next_vehicle_id.to(torch.int64).unsqueeze(-1), 
                False
            )
            # 重置后处理时间
            self.vehicle_post_time *= ~next_vehicle_mask
        
        #-------------------
        # update skip batch
        #-------------------
        # end flags

        # 在 update_state 方法结束前
        if current_step_log is not None:
            current_step_log["queue_state_after"]["charge_queue"] = self.charge_queue[batch].clone().detach().cpu().tolist()
            current_step_log["queue_state_after"]["vehicle_phase"] = self.vehicle_phase[batch].clone().detach().cpu().tolist()
            current_step_log["queue_state_after"]["vehicle_position_id"] = self.vehicle_position_id[batch].clone().detach().cpu().tolist()
            current_step_log["queue_state_after"]["vehicle_unavail_time"] = self.vehicle_unavail_time[batch].clone().detach().cpu().tolist()
            
            # 保存本次计算日志
            self.queue_calc_log[batch].append(current_step_log)

        # [前面的代码保持不变]
        
        # 在更新 skip 和 end 状态之前先记录当前状态
        old_skip = self.skip.clone()
        old_end = self.end.clone()
        
        # 用于可视化的特殊标志 - 添加这个新变量
        should_visualize = torch.ones_like(self.skip, dtype=torch.bool)
        
        #-------------------
        # 更新skip和end状态
        #-------------------
        # 检查1：所有客户点是否已访问
        all_locs_visited = self.loc_visited.all(dim=1)  # [batch_size]
        
        # 检查2：哪些车辆处于"移动"阶段
        moving_vehicles = (self.vehicle_phase == self.phase_id["move"])  # [batch_size, num_vehicles]
        
        # 检查3：当前时间是否达到上限
        time_up = self.current_time >= self.time_horizon  # [batch_size]
        
        # 初始化新的skip和end状态
        new_skip = torch.zeros_like(self.skip)
        new_end = torch.zeros_like(self.end)
        
        # 对每个批次独立处理
        for batch_idx in range(self.batch_size):
            # 获取当前选择的车辆ID（针对此批次）
            veh_id = next_vehicle_id[batch_idx].item()
            
            # 如果所有客户点已被访问，进入"仅移动"收尾阶段
            if all_locs_visited[batch_idx]:
                # 如果选中的车辆不处于移动阶段，则跳过此步
                if veh_id < self.num_vehicles and not moving_vehicles[batch_idx, veh_id]:
                    new_skip[batch_idx] = True
                    # 关键修改：在收尾阶段，不可视化非移动车辆
                    should_visualize[batch_idx] = False
                
                # 检查是否还有车辆处于移动阶段
                if not moving_vehicles[batch_idx].any():
                    # 如果没有车辆处于移动阶段，且所有客户点已访问，则设置结束标志
                    new_end[batch_idx] = True
            
            # 时间到达上限也会导致结束
            if time_up[batch_idx]:
                new_end[batch_idx] = True
                new_skip[batch_idx] = True
        
        # 更新状态
        self.end = new_end
        self.skip = new_skip | self.end  # 确保一旦end为True，skip也为True
        
        # 记录状态变化
        if self.fname is not None and batch_idx == 0:  # 仅记录第一个批次
            # 记录阶段转换
            if not old_end[0] and self.end[0]:
                print(f"批次0在时间 {self.current_time[0]:.3f} 结束。所有客户点已访问: {all_locs_visited[0].item()}, "
                    f"无移动车辆: {not moving_vehicles[0].any().item()}, 时间耗尽: {time_up[0].item()}")
        
        #---------------------
        # 可视化状态更新
        #---------------------
        if self.fname is not None:
            # 修改可视化触发条件：使用新的should_visualize标志代替直接使用~self.skip
            vis_batch = update_batch & should_visualize & (torch.abs(elapsed_time) > SMALL_VALUE)
            if vis_batch.any():  # 只有当有批次需要可视化时才调用
                self.visualize_state_batch(vis_batch)
# # 假设一个批次中有2辆车
# # 1. 充电到后处理阶段
# vehicle_phase = [
#     "charge",    # 车辆0正在充电
#     "move"       # 车辆1正在移动
# ]
# next_vehicle_on_charge = [True, False]
# # 添加后处理时间，重置充电时间

# # 2. 更新车辆阶段
# vehicle_phase = [
#     "post",     # 车辆0进入后处理阶段
#     "move"      # 车辆1保持移动
# ]

# # 3. 后处理到移动阶段
# # - 更新充电队列
# # - 重置等待状态
# # - 准备下一次移动

# # 4. 检查完成状态
# if current_time >= time_horizon:
#     end = True
#     skip = True  # 跳过后续处理
    #---------
    # masking
    #---------
    def update_mask(self, next_node_id, next_vehicle_mask, relaxed_mask=True):
        #---------------------------------------------------------------
        # mask: 0 -> infeasible, 1 -> feasible [batch_size x num_nodes]
        #---------------------------------------------------------------
        """更新掩码，确定哪些节点可访问
    
        Parameters
        ----------
        next_node_id: torch.LongTensor [batch_size]
            下一个节点的ID
        next_vehicle_mask: torch.BoolTensor [batch_size x num_vehicles]
            标识哪些车辆被选中的掩码

        Returns
        -------
        mask: torch.LongTensor [batch_size x num_nodes]
            0表示不可访问，1表示可访问
        """
        
        batch = 0  # 之后改 看这里可不可以调整 为简化，主要记录第一个批次的详情 看看多个批次一样吗
        current_step_log = {
            "step": self.episode_step,
            "time": self.current_time[batch].item(),
            "acting_vehicle": self.next_vehicle_id[batch].item(),
            "current_node": next_node_id[batch].item(),
            "rule_masks": {},
            "intermediate_results": {}
        }
        # 在断言前添加这段代

        
        #创建全1矩阵，初始状态所有节点都可访问
        mask = torch.ones(self.batch_size, self.num_nodes, dtype=torch.int32, device=self.device) # [batch_size x num_nodes]
        # Only add to mask_calc_log if it exists (when fname is not None)
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["rule_masks"]["initial"] = mask[batch].clone().detach().cpu()

        # 规则1: 车辆无法返回的节点
        self.mask_unreturnable_nodes(mask, next_node_id, next_vehicle_mask)

        # EV cannot discharge power when its battery rearches the limit, so EV should return to a depot at that time.
        # 规则1: 电量达到放电下限时必须返回充电站
        # rearch_discharge_lim = (self.vehicle_curr_battery <= self.vehicle_discharge_lim + SMALL_VALUE)[next_vehicle_mask]
    # Only add to mask_calc_log if it exists
        # if self.fname is not None and hasattr(self, 'mask_calc_log'):
        #     current_step_log["intermediate_results"]["discharge_limit_triggered"] = rearch_discharge_lim[batch].item()

        # self.return_to_depot_when_discharge_limit_rearched(mask, next_vehicle_mask)
    # Only add to mask_calc_log if it exists
#         if self.fname is not None and hasattr(self, 'mask_calc_log'):
#             current_step_log["rule_masks"]["after_discharge_limit"] = mask[batch].clone().detach().cpu()
# #         当车辆电量达到放电限制时，必须返回充电站
# 将不是充电站的节点标记为不可访问
        # mask 0: if a selected vehicle is out of battery, we make it return to a depot
            # 规则2: 屏蔽无法返回的节点
        # self.mask_unreturnable_nodes(mask, next_node_id, next_vehicle_mask)
        # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["rule_masks"]["after_unreturnable"] = mask[batch].clone().detach().cpu()

        # 规则3: 屏蔽已被其他车辆访问的节点
        reserved_loc = torch.full((self.batch_size, self.num_nodes), False, device=self.device)
        reserved_loc.scatter_(1, self.vehicle_position_id, True)
        # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["intermediate_results"]["reserved_nodes"] = reserved_loc[batch].clone().detach().cpu()

        # # 保存规则3应用前的掩码状态
        # if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["rule_masks"]["before_visited_locs"] = mask[batch].clone().detach().cpu()
        
        # 重置充电站部分为False
        reserved_loc[:, self.num_locs:] = False
        
        # 保存修改后的reserved_loc（充电站重置后）
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["intermediate_results"]["reserved_nodes_after_reset"] = reserved_loc[batch].clone().detach().cpu()


#         标记车辆无法到达的节点
# 考虑电量和时间限制
        # mask 2: forbits vehicles to move between two different depots
        # i.e., if a selcted vechile is currently at a depot, it cannot visit other depots in the next step (but it can stay in the same depot)

        #取消depot到depot的限制
        # mask 3: vehicles cannot visit a location/depot that other vehicles are visiting
        self.mask_visited_locs(mask)
    # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["rule_masks"]["after_visited_locs"] = mask[batch].clone().detach().cpu()
        


        # 在这里插入规则3.5的代码
        # 规则3.5: 屏蔽已经访问过的客户点
        # 将 self.loc_visited [batch_size, num_locs] 扩展到 [batch_size, num_nodes]
        padded_visited_mask = torch.cat(
            (self.loc_visited,
            torch.full((self.batch_size, self.num_depots), False, dtype=torch.bool, device=self.device)),
            dim=1
        )
        # 记录日志
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["intermediate_results"]["visited_locations"] = self.loc_visited[batch].clone().detach().cpu()
            current_step_log["intermediate_results"]["padded_visited_mask"] = padded_visited_mask[batch].clone().detach().cpu()
        # 保存应用规则3.5前的掩码
        mask_before_rule3_5 = mask.clone()
        # 应用规则3.5：已访问客户点不可再访问
        mask *= ~padded_visited_mask
        # 记录日志
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["rule_masks"]["before_rule3_5"] = mask_before_rule3_5[batch].clone().detach().cpu()
            current_step_log["rule_masks"]["after_rule3_5"] = mask[batch].clone().detach().cpu()

        # 规则4: 屏蔽从一个充电站到另一个充电站的移动
        at_depot = next_node_id.ge(self.num_locs)
        # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["intermediate_results"]["at_depot"] = at_depot[batch].item()
        
        self.mask_depot_to_other_depots(mask, next_node_id)
        #Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
           current_step_log["rule_masks"]["after_depot_to_depot"] = mask[batch].clone().detach().cpu()
        
        # mask 4: forbit vehicles to visit depots that have small discharge rate
#         标记其他车辆正在访问的节点为不可访问
# 避免多车同时访问同一节点
        # 规则5: 屏蔽低功率充电站
        small_depots = self.get_unavail_depots(next_node_id)
        # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["intermediate_results"]["small_depots"] = small_depots[batch].clone().detach().cpu()
        
        self.remove_small_depots(mask, next_node_id)
        # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["rule_masks"]["after_small_depots"] = mask[batch].clone().detach().cpu()
    
        # mask 5: in skipped episodes(instances), the selcted vehicles always stay in the same place
#         标记放电速率过低的充电站为不可访问
# 确保充电效率
# 这个之后改就没了 一次所有站点都符合条件
# 相对于注释掉 不如 直接改成符合所有条件
        # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["intermediate_results"]["is_skipped"] = self.skip[batch].item()
        
        self.mask_skipped_episodes(mask, next_node_id)
        # 在这里添加新代码 (在保存最终掩码之前)
        if mask.sum(-1).min() == 0:  # If any vehicle has no valid nodes
            empty_mask_indices = torch.where(mask.sum(-1) == 0)[0]
            for idx in empty_mask_indices:
                current_node = next_node_id[idx]
                mask[idx, current_node] = 1

        # Only add to mask_calc_log if it exists
        if self.fname is not None and hasattr(self, 'mask_calc_log'):
            current_step_log["rule_masks"]["after_skipped"] = mask[batch].clone().detach().cpu()
            
            # 保存最终掩码和完整日志
            current_step_log["final_mask"] = mask[batch].clone().detach().cpu()
            
            # Only append to mask_calc_log if it exists
            self.mask_calc_log[batch].append(current_step_log)
        # 在这里添加新代码 (在保存最终掩码之前)

        if mask.sum(-1).min() == 0:  # If any vehicle has no valid nodes
            empty_mask_indices = torch.where(mask.sum(-1) == 0)[0]
            for idx in empty_mask_indices:
                current_node = next_node_id[idx]
                mask[idx, current_node] = 1

        # 在这里添加调试信息 (就在断言检查之前)
        all_zero = mask.sum(-1) == 0  # [batch_size]
        if all_zero.any() and relaxed_mask:
            for batch_idx in range(self.batch_size):
                        # Find where we have no valid nodes
                for batch_idx in torch.where(all_zero)[0]:
                    # Allow staying at the current node as a fallback
                    if all_zero[batch_idx]:
                        print(f"No valid nodes for batch {batch_idx}!")
                        print(f"Current time: {self.current_time[batch_idx]}")
                        print(f"Vehicle battery: {self.vehicle_curr_battery[batch_idx]}")
                        print(f"Vehicle position: {self.vehicle_position_id[batch_idx]}")
                        print(f"Mask state: {mask[batch_idx]}")
                        print(f"Is at depot: {self.is_depot(next_node_id[batch_idx])}")
                        print(f"Vehicle phase: {self.vehicle_phase[batch_idx]}")
                        print(f"Vehicle unavailable time: {self.vehicle_unavail_time[batch_idx]}")
                        mask[batch_idx, next_node_id[batch_idx]] = 1

        # Original assertion check
        assert not all_zero.any(), "there is no node that the vehicle can visit!"
        
        return mask
# 假设：batch_size=2, num_nodes=5
# # 初始掩码:
# mask = [
#     [1, 1, 1, 1, 1],  # batch 0：所有节点可访问
#     [1, 1, 1, 1, 1]   # batch 1：所有节点可访问
# ]

# # 应用各种限制后:
# mask = [
#     [0, 1, 0, 1, 0],  # batch 0：只有节点1和3可访问
#     [1, 0, 1, 0, 1]   # batch 1：只有节点0,2,4可访问
# ]
    # def return_to_depot_when_discharge_limit_rearched(self, mask, next_vehicle_mask):
    #     rearch_discharge_lim = (self.vehicle_curr_battery <= self.vehicle_discharge_lim + SMALL_VALUE)[next_vehicle_mask] # [batch_size]
    #     mask[:, :self.num_locs] *= ~rearch_discharge_lim.unsqueeze(-1) # zero out all nodes in the sample where the selected EV rearches the discharge limit
# # 假设场景：
# batch_size = 2
# num_locs = 3
# num_vehicles = 2

# # 车辆当前电量
# vehicle_curr_battery = [
#     [50, 30],  # batch 0的两辆车
#     [20, 40]   # batch 1的两辆车
# ]

# # 车辆放电限制
# vehicle_discharge_lim = [
#     [25, 25],  # batch 0的两辆车的放电限制
#     [25, 25]   # batch 1的两辆车的放电限制
# ]

# # 选中的车辆
# next_vehicle_mask = [
#     [True, False],   # batch 0选中第一辆车
#     [False, True]    # batch 1选中第二辆车
# ]

# # 初始掩码
# mask = [
#     [1, 1, 1, 1, 1],  # batch 0: 所有节点可访问
#     [1, 1, 1, 1, 1]   # batch 1: 所有节点可访问
# ]

# # 检查结果：
# # batch 0: 车辆电量50 > 25，未达到限制
# # batch 1: 车辆电量40 > 25，未达到限制
# # 所以掩码保持不变
    def mask_unreturnable_nodes(self, mask, next_node_id, next_vehicle_mask):
        """
        There are two patterns:
            1. unreturnable to depot within time horizon
            2. unreturnable to depot without running out of vehicle battery
        """
        # mask 1: guarantee that all the vehicles return to depots within the time horizon
        remaining_time = (self.time_horizon - self.current_time).unsqueeze(-1).clamp(0.0) # [batch_size x 1]
        current_coord = torch.gather(self.coords, 1, next_node_id.view(-1, 1, 1).expand(-1, 1, self.coord_dim)) # [batch_size, 1, coord_dim]
        #unavail_depots = self.small_depots.unsqueeze(-1).expand_as(self.depot_coords) # [batch_size x num_depots x coord_dim]
        #depot_coords = self.depot_coords + 1e+6 * unavail_depots # set large value for removing small depots [batch_size x num_depots x coord_dim]
            # 获取充电站坐标
        depot_coords = self.depot_coords
        #  计算当前位置到所有基站的距离
        current_to_loc = COEF * torch.linalg.norm(self.loc_coords - current_coord, dim=-1)
        
        # 计算所有基站到最近充电站的距离
        loc_to_depot = COEF * torch.min(
            torch.cdist(self.loc_coords, depot_coords), -1)[0]
        # 保护措施
        # 计算到达基站所需时间
        current_to_loc_time = current_to_loc / self.speed.unsqueeze(-1)
        # 计算从基站到充电站所需时间
        loc_to_depot_time = loc_to_depot / self.speed.unsqueeze(-1)
        # 如果当前位置就在基站，则添加等待时间
        #wait_time = self.wait_time * (torch.abs(current_to_loc) < SMALL_VALUE)
        #--------------------------------------------------------------------------------------------------------------
        # vehicles can visit only locations that the vehicles can return to depots within time horizon after the visit
        # i.e. travel time t_(current_node -> next_loc -> depot) + current_time <= time_horizon
        #--------------------------------------------------------------------------------------------------------------
        # loc_charge_time should be wait_time not zero in the locations where a vehicle is waiting,
        # but ignore that because those locations are masked out later 
        runout_battery_loc = ((current_to_loc + loc_to_depot) * self.vehicle_consump_rate[next_vehicle_mask].unsqueeze(-1)) > self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1) + BIT_SMALL_VALUE
    
        
#                 # 假设场景：
#         current_to_loc = 10       # 到基站距离10km
#         loc_to_depot = 15        # 基站到充电站15km
#         vehicle_consump_rate = 2  # 行驶耗电率2kWh/km
#         loc_consump_rate = 5     # 基站耗电率5kWh/h
#         wait_time = 0.5          # 等待时间0.5h
#         vehicle_curr_battery = 60 # 当前电量60kWh
        
#         # 计算：
#         # 1. 行驶耗电 = (10 + 15) * 2 = 50 kWh
#         # 2. 等待耗电 = 5 * 0.5 = 2.5 kWh
#         # 总耗电 = 52.5 kWh < 60 kWh
#         # 结果：False (电量足够)
#         被注释掉的部分：

# 这部分原本是考虑给基站充电时的耗电，但现在似乎已改为使用其他方式计算。
        # if its battery is zero, the vehicle should return to a depot (this is used only when vehicle_consump_rate = 0)
        battery_zero = torch.abs(self.vehicle_curr_battery[next_vehicle_mask]) < SMALL_VALUE # [batch_size]
        # mask for unreturnable locations
        unreturnable_loc = battery_zero.unsqueeze(-1) | runout_battery_loc # [batch_size x num_locs]
        if self.return_depot_within_time_horizon:
            # ignore the battery change of visited locations (either way, they are masked out later)
            # loc_battery_on_arrival = (self.loc_curr_battery - self.loc_consump_rate * (current_to_loc_time + self.pre_time_loc)).clamp(self.loc_min_battery) # [batch_size x num_locs]
            #计算到达时基站电量
            #curr_demand = torch.minimum(self.loc_cap - loc_battery_on_arrival, self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1)) # [batch_size x num_locs]
            #计算需求量
            # time limit
            #loc_charge_time = curr_demand / (self.vehicle_discharge_rate[next_vehicle_mask].unsqueeze(-1) - self.loc_consump_rate) # [batch_size x num_locs]
            exceed_timehorizon_loc = (
                current_to_loc_time +     # 到达基站时间
                self.pre_time_loc +       # 准备时间
                #loc_charge_time +         # 充电时间
                self.post_time_loc +      # 完成时间
                loc_to_depot_time       # 返回充电站时间
                #wait_time                 # 等待时间
            ).gt(remaining_time + BIT_SMALL_VALUE)  # 是否超过剩余时间
            unreturnable_loc |= exceed_timehorizon_loc
        #         # 假设场景:
        # remaining_time = 5.0         # 剩余5小时
        # current_to_loc_time = 1.0    # 到基站需要1小时
        # pre_time_loc = 0.5          # 准备时间0.5小时
        # loc_charge_time = 2.0       # 充电需要2小时
        # post_time_loc = 0.5         # 完成时间0.5小时
        # loc_to_depot_time = 1.5     # 返回充电站1.5小时
        # wait_time = 0.0             # 无等待时间
        
        # # 计算总时间
        # total_time = 1.0 + 0.5 + 2.0 + 0.5 + 1.5 + 0.0 = 5.5小时
        
        # # 检查结果
        # exceed_timehorizon_loc = 5.5 > 5.0  # True
        # # 因此这个基站被标记为不可访问
        #---------------------------------------------------------------------------------------
        # vehicles can visit only depots that the vehicles can arrive there within time horizon
        # i.e. travel time t_(current_node -> depot) + current_time <= time_horizon
        #---------------------------------------------------------------------------------------
        # 获取不可用充电站的掩码并扩展维度
        unavail_depots2 = self.get_unavail_depots(next_node_id).unsqueeze(-1).expand_as(self.depot_coords)
        
        # 将不可用充电站的坐标设置为很大的值(1e+6)，这样在后续计算距离时会自动被排除
        depot_coords2 = self.depot_coords + 1e+6 * unavail_depots2
        # 计算当前位置到充电站的距离
        current_to_depot = COEF * torch.linalg.norm(depot_coords2 - current_coord, dim=-1)
        
        # 计算到达充电站所需时间
        current_to_depot_time = current_to_depot / self.speed.unsqueeze(-1)     # battery
        # 计算到达充电站需要的电量
        curr2depot_batt = current_to_depot * self.vehicle_consump_rate[next_vehicle_mask].unsqueeze(-1)
        
        # 获取车辆当前电量
        veh_batt = self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1)
        
        # 检查是否有足够电量到达充电站
        runout_battery_depot = curr2depot_batt >= veh_batt + BIT_SMALL_VALUE
        unreturnable_depot = runout_battery_depot
        if self.return_depot_within_time_horizon:
            # 检查是否能在时间限制内到达充电站
            exceed_timehorizon_depot = (current_to_depot).gt(remaining_time + BIT_SMALL_VALUE)
            # 更新不可达充电站掩码
            unreturnable_depot |= exceed_timehorizon_depot

            #             # 假设场景：
            # batch_size = 2
            # num_depots = 3
            # current_coord = [
            #     [0.1, 0.2],  # batch 0当前位置
            #     [0.3, 0.4]   # batch 1当前位置
            # ]
            # depot_coords = [
            #     [[0.5, 0.5], [0.7, 0.7], [0.9, 0.9]],  # batch 0的充电站
            #     [[0.4, 0.4], [0.6, 0.6], [0.8, 0.8]]   # batch 1的充电站
            # ]
            # vehicle_curr_battery = [100, 80]  # 当前电量
            # vehicle_consump_rate = [2, 2.5]   # 耗电率
            # speed = [1.0, 1.0]               # 车速
            # remaining_time = [5.0, 4.0]      # 剩余时间
            
            # # 计算结果示例：
            # current_to_depot = [
            #     [0.4, 0.6, 0.8],  # batch 0到各充电站距离
            #     [0.3, 0.5, 0.7]   # batch 1到各充电站距离
            # ]
            
            # curr2depot_batt = [
            #     [0.8, 1.2, 1.6],  # batch 0所需电量
            #     [0.75, 1.25, 1.75] # batch 1所需电量
            # ]
            
            # # 判断结果：
            # unreturnable_depot = [
            #     [False, False, True],   # batch 0：第3个充电站不可达
            #     [False, True, True]     # batch 1：第2,3个充电站不可达
            # ]
        # there should be at least one depot that the vehicle can reach
        # 初始化精度列表，从高精度到低精度
        i = 0
        atol_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        
        # 当存在车辆没有可达充电站时循环
        while (((~unreturnable_depot).int().sum(-1) == 0) & ~self.skip).any():
            # 找出没有可达充电站的批次
            all_zero = ((~unreturnable_depot).int().sum(-1) == 0).unsqueeze(-1)
            
            # 检查电量是否接近耗尽
            close_to_runout_battery = torch.isclose(
                curr2depot_batt,  # 到达充电站需要的电量
                veh_batt,         # 当前电量
                atol=atol_list[i] # 当前精度
            )
            
            # 如果需要在时间限制内返回
            if self.return_depot_within_time_horizon:
                # 检查是否接近时间限制
                close_to_timehorizon = torch.isclose(
                    current_to_depot_time,
                    remaining_time,
                    atol=atol_list[i]
                )
                close_to_runout_battery |= close_to_timehorizon
            
            # 将接近的充电站标记为可达
            unreturnable_depot[all_zero & close_to_runout_battery] = False
            
            # 尝试下一个精度级别
            i += 1

            #             # 假设场景：
            # curr2depot_batt = [99.999, 50.001]  # 需要的电量
            # veh_batt = [100.0, 50.0]            # 当前电量
            
            # # 第一次迭代 (atol=1e-6)：
            # # 都标记为不可达，因为差异大于1e-6
            
            # # 第二次迭代 (atol=1e-5)：
            # # 第一个充电站标记为可达，因为 |99.999 - 100.0| < 1e-5
            if i >= len(atol_list):
                print(self.depot_discharge_rate[all_zero.squeeze(-1)])
                print(f"battery consumption b/w loc2depot: {(current_to_depot * self.vehicle_consump_rate[next_vehicle_mask].unsqueeze(-1))[all_zero.squeeze(-1)].tolist()}")
                print(f"selected vehicle's battery: {self.vehicle_curr_battery[next_vehicle_mask].unsqueeze(-1)[all_zero].tolist()}")
                print(torch.where(all_zero))
                print(f"travel time b/w loc2depot: {current_to_depot_time[all_zero.squeeze(-1)].tolist()}")
                print(f"remaining time: {remaining_time[all_zero].tolist()}")
                assert False, "some vehicles could not return to any depots within time horizon due to numerical error."
        # 如果尝试所有精度都失败，输出详细信息并报错
#         安全保障：确保每个车辆都有至少一个可达的充电站
# 渐进式检查：通过逐步放宽误差容限来找到可行解
# 调试支持：在完全失败时提供详细的诊断信息
        #--------------------------
        # update unreturnable mask
        #--------------------------
        unreturnable_mask = torch.cat((unreturnable_loc, unreturnable_depot), 1) # [batch_size x num_nodes]
        mask *= ~unreturnable_mask
    
    def mask_visited_locs(self, mask: torch.Tensor):
        """
        Remove visited locs from the next-node candidates
        
        Parameters
        ----------
        mask: torch.LongTensor [batch_size x num_nodes]
        """
        reserved_loc = torch.full((self.batch_size, self.num_nodes), False, device=self.device)
#         创建一个全False的张量
# 形状为 [batch_size × num_nodes]
# 初始状态表示没有节点被访问
        reserved_loc.scatter_(1, self.vehicle_position_id, True)
        # 使用scatter_操作将车辆当前所在位置标记为True
        reserved_loc[:, self.num_locs:] = False
#         将充电站部分重置为False
# 表示充电站可以被多辆车同时访问
# 之后改loc不能被多辆车同时访问 不懂这里为什么这么设置 有可能是这个原因 之前的测试中有些电站没有被访问
        mask *= ~reserved_loc
#         ~reserved_loc: 取反，True变False，False变True
# 将已访问节点在mask中对应位置设为0

# 假设场景：batch_size=2, num_nodes=5, num_locs=3

# 1. 初始reserved_loc
# reserved_loc = [
#     [False, False, False, False, False],  # batch 0
#     [False, False, False, False, False]   # batch 1
# ]

# # 2. 假设vehicle_position_id表示车辆在节点1和2
# self.vehicle_position_id = [
#     [1],  # batch 0的车辆在节点1
#     [2]   # batch 1的车辆在节点2
# ]

# # 3. scatter_后的reserved_loc
# reserved_loc = [
#     [False, True,  False, False, False],  # batch 0
#     [False, False, True,  False, False]   # batch 1
# ]

# # 4. 原始mask
# mask = [
#     [1, 1, 1, 1, 1],  # batch 0
#     [1, 1, 1, 1, 1]   # batch 1
# ]

# # 5. 最终结果 mask *= ~reserved_loc
# mask = [
#     [1, 0, 1, 1, 1],  # batch 0: 节点1不可访问
#     [1, 1, 0, 1, 1]   # batch 1: 节点2不可访问
# ]
    def mask_depot_to_other_depots(self, mask: torch.Tensor, next_node_id: torch.Tensor):
        """
        A mask for removing moving between different two depots
        
        Parameters
        ----------
        mask: torch.LongTesnor [batch_size x num_nodes]
        next_node_id: torch.LongTensor [batch_size]
        """
        at_depot = next_node_id.ge(self.num_locs).unsqueeze(1) # [batch_size x 1]
#         ge(self.num_locs): 判断ID是否大于等于基站数量(充电站ID从num_locs开始)
# 返回布尔张量，True表示当前在充电站
        # other_depot = self.node_arange_idx.ne(next_node_id.unsqueeze(1)) # [batch_size x num_nodes]
        # other_depot[:, :self.num_locs] = False # ignore locations here
        #之后改 原本的代码太严格 这个逻辑不仅禁止了从充电站到其他充电站的移动，还禁止了从充电站到任何位置节点的移动！这就是为什么启用后车辆会停留在原地。
        # other_depot = self.node_arange_idx.ge(self.num_locs)  # 所有充电站
        # other_depot = other_depot & self.node_arange_idx.ne(next_node_id.unsqueeze(1))  # 不是当前节点的充电站
        #  # 忽略基站部分 之后改 其实应mask的是loc 
        all_depots = self.node_arange_idx.ge(self.num_locs)  # 所有充电站
        # 当车辆在充电站时，禁止移动到任何充电站（包括当前所在的充电站）
        mask *= ~(at_depot & all_depots)
        # 之后改 看了maskcalculation 发现司机留在自己的电站是可行的 试试可不可以不能留在电站
        # 找出当前在充电站且目标是其他充电站的情况
        # ~(...): 取反，将这些情况标记为不可行

    def remove_small_depots(self, mask: torch.Tensor, next_node_id: torch.Tensor):
        """
        A mask for removing small depots, which have low discharge_rate
        """
        unavail_depots = self.get_unavail_depots(next_node_id) # [batch_size x num_depots]
        unavail_nodes = torch.cat((torch.full((self.batch_size, self.num_locs), False).to(self.device), unavail_depots), -1) # [batch_size x num_nodes]
        mask *= ~unavail_nodes
        return mask
    
    def get_unavail_depots(self, next_node_id: torch.Tensor):
        # 1. 获取当前正在访问的充电站掩码
        stayed_depots = self.get_depot_mask(next_node_id) 
        #创建一个掩码，标识当前车辆所在的充电站
        # 2. 返回不可用充电站的掩码
        # ~stayed_depots: 未被访问的充电站
        # self.small_depots: 放电率低的充电站
        return ~stayed_depots & self.small_depots 
    #~stayed_depots: 取反，标识未被访问的充电站
# self.small_depots: 放电率低于阈值的充电站
# &: 逻辑与，找出既未被访问且放电率低的充电站
#之后改 这个逻辑可以学习

    def get_unavail_depots2(self, next_node_id: torch.Tensor):
        return self.small_depots
    
    def mask_skipped_episodes(self, mask: torch.Tensor, next_node_id: torch.Tensor):
        # 1. 创建当前节点掩码
        current_node = self.node_arange_idx.eq(next_node_id.unsqueeze(1)).int()
#         标识每个批次中车辆当前所在的节点
# 返回一个只有当前节点为1，其他节点为0的掩码
        # 2. 对已跳过的批次更新掩码
        mask[self.skip] = current_node[self.skip]
#         仅对已跳过的批次（self.skip为True）应用更新
# 使车辆只能停留在当前位置
# 处理已完成或需要跳过的批次
# 确保跳过批次中的车辆：
# 停留在当前位置
# 不能移动到其他节点
# 维护模拟的一致性
# 已完成任务的批次不会继续移动
# 异常情况下的批次能够安全停止

# # 假设场景:
# batch_size = 2
# num_nodes = 4
# next_node_id = [1, 2]  # 当前节点ID
# self.skip = [True, False]  # 批次0被跳过，批次1正常

# # 1. 创建当前节点掩码
# node_arange_idx = [
#     [0, 1, 2, 3],  # 批次0的节点索引
#     [0, 1, 2, 3]   # 批次1的节点索引
# ]

# current_node = [
#     [0, 1, 0, 0],  # 批次0：只有节点1可访问
#     [0, 0, 1, 0]   # 批次1：只有节点2可访问
# ]

# # 2. 原始掩码
# mask = [
#     [1, 1, 1, 1],  # 批次0
#     [1, 1, 1, 1]   # 批次1
# ]

# # 3. 更新后的掩码
# mask = [
#     [0, 1, 0, 0],  # 批次0(跳过)：只能待在当前节点
#     [1, 1, 1, 1]   # 批次1(未跳过)：保持不变
# ]

    def get_inputs(self):
        """
        Returns
        -------
        node_feats: torch.tensor [batch_size x num_nodes x node_feat_dim]
            input features of nodes
        vehicle_feats: torch.tensor [batch_size x num_vehicles x vechicle_feat_dim]
            input features of vehicles
        """
        visit_mask = torch.full((self.batch_size, self.num_nodes), 0.0, device=self.device)
        visit_mask.scatter_(1, self.vehicle_position_id, 1.0)
        # for locations (loc_dim = 1+2+1+1+1+1 = 7)
        loc_feats = torch.cat((
            visit_mask[:, :self.num_locs, None],     # 是否被车辆访问 [0/1]
            self.loc_coords,                          # 基站坐标 [x,y]
            # self.loc_cap.unsqueeze(-1) / self.max_cap,           # 基站容量(归一化)
            # # 之后改 这里的数据归一了
            # self.loc_consump_rate.unsqueeze(-1) / self.max_cap,  # 基站耗电率(归一化)
            # self.loc_curr_battery.unsqueeze(-1) / self.max_cap,  # 当前电量(归一化)
            # (self.loc_curr_battery / self.loc_consump_rate).unsqueeze(-1)  # 预计耗尽时间
        ), -1)
        # 之后改 纬度减少
        # Update loc_dim accordingly (from 7 to 3)

        # for depots (depot_dim = 1+2+1 = 4)
        depot_feats = torch.cat((
            visit_mask[:, self.num_locs:, None],     # 是否被车辆访问 [0/1]
            self.depot_coords,                        # 充电站坐标 [x,y]
            self.depot_discharge_rate.unsqueeze(-1) / self.max_cap  # 充电速率(归一化)
        ), -1)
        # for vehicles (vehicle_dim = 1+2+1+1++4+1+1 = 11)
        curr_vehicle_coords = self.coords.gather(1, self.vehicle_position_id.unsqueeze(-1).expand(self.batch_size, self.num_vehicles, self.coord_dim)) # [batch_size x num_vehicles x coord_dim]
        vehicle_phase_time = torch.concat((
            self.vehicle_move_time.unsqueeze(-1),
            self.vehicle_pre_time.unsqueeze(-1),
            self.vehicle_work_time.unsqueeze(-1),
            self.vehicle_post_time.unsqueeze(-1)
        ), -1) # [batch_size x num_vehicles x 4]
        vehicle_phase_time.scatter_(-1, self.vehicle_phase.unsqueeze(-1), self.vehicle_unavail_time.unsqueeze(-1)) # [batch_size x num_vehicles x 4]
        vehicle_phase_time_sum = vehicle_phase_time.sum(-1, keepdim=True)
        vehicle_feats = torch.cat((
            self.vehicle_cap.unsqueeze(-1) / self.max_cap,  # 电池容量(归一化)
            curr_vehicle_coords,                             # 当前坐标 [x,y]
            self.is_depot(self.vehicle_position_id).unsqueeze(-1).to(torch.float),                             # 是否在充电站 [0/1]
            self.vehicle_phase.unsqueeze(-1) / self.phase_id_max,  # 当前阶段
            vehicle_phase_time,                             # 各阶段剩余时间 [4维]
            vehicle_phase_time_sum,                         # 总剩余时间
            self.vehicle_curr_battery.unsqueeze(-1) / self.max_cap # 当前电量(归一化)
        ), -1)
        return loc_feats, depot_feats, vehicle_feats

    def get_mask(self):
        """
        Returns
        -------
        mask: torch.tensor [batch_size x num_nodes]
        """
        return self.mask

    def get_selected_vehicle_id(self):
        """
        Returns 
        -------
        next_vehicle_id [batch_size]
        """
        return self.next_vehicle_id.to(torch.int64)

    def all_finished(self):
        """
        Returns
        -------
        end: torch.tensor [batch_size]
        """
        return self.end.all()

    def get_rewards(self):
        # 被注释掉的惩罚计算逻辑：
        # # 1. 计算剩余时间
        # remaining_time = self.time_horizon - self.current_time
        
        # # 2. 更新基站电量
        # self.loc_curr_battery -= self.loc_consump_rate * remaining_time.unsqueeze(-1)
        
        # # 3. 识别电量耗尽的基站
        # down_locs = (self.loc_curr_battery - self.loc_min_battery) < SMALL_VALUE
        
        # # 4. 计算电量耗尽时间
        # num_empty_locs = ((self.loc_curr_battery - self.loc_min_battery)[down_locs] / 
        #                   self.loc_consump_rate[down_locs]).sum(-1)
        
        # # 5. 忽略跳过的批次
        # num_empty_locs[self.skip] = 0
        
        # # 6. 更新惩罚值
        # self.penalty_empty_locs += num_empty_locs / self.num_locs
        # #累积的基站电量耗尽惩罚
# 除以时间范围进行归一化

#         总行驶距离除以车辆数量
# 得到每辆车的平均行驶距离
        # penalty = self.penalty_empty_locs / self.time_horizon
        conflict_cost = self.accumulated_conflict_cost / self.time_horizon
    # 路径长度保持不变
        tour_length = self.tour_length / self.num_vehicles
        return {
            "tour_length": tour_length,  # 归一化的路径长度
            "conflict_cost": conflict_cost  # 归一化的冲突成本（替代原有的 penalty）
        }
    #     # 假设场景：
    # time_horizon = 10.0        # 总时间范围
    # num_vehicles = 3          # 3辆车
    # tour_length = 30.0       # 总行驶距离
    # penalty_empty_locs = 5.0  # 累积惩罚值
    
    # # 计算奖励：
    # rewards = {
    #     "tour_length": 30.0 / 3 = 10.0,    # 每辆车平均行驶距离
    #     "penalty": 5.0 / 10.0 = 0.5        # 归一化惩罚值
    # }
#     这些奖励值将用于：

# 评估当前策略的好坏
# 指导强化学习的策略更新
# 比较不同解决方案的性能
    def get_final_metrics(self):
        """Gets detailed evaluation metrics
        
        Returns
        -------
        dict
            Dictionary containing detailed evaluation metrics
        """
        # Get base metrics
        base_metrics = self.get_rewards()
        
        # Calculate service coverage
        percent_served = (self.loc_visited.float().sum(dim=1) / self.num_locs) * 100.0
        
        # Calculate average first response time (only consider visited locations)
        visited_mask = self.loc_first_visit_time >= 0
        avg_first_response = torch.zeros(self.batch_size, device=self.device)
        for b in range(self.batch_size):
            if visited_mask[b].any():
                avg_first_response[b] = self.loc_first_visit_time[b][visited_mask[b]].mean()
        
        # Calculate vehicle utilization
        total_active_time = self.total_travel_time + self.total_service_time + self.total_charge_time
        vehicle_utilization = total_active_time / (self.num_vehicles * self.time_horizon)
        
        # Calculate average travel time
        avg_travel_time = self.total_travel_time / self.num_vehicles
        
        # Calculate average wait time
        avg_wait_time = self.total_wait_time / self.num_vehicles
        
        # Calculate charging station average utilization
        avg_station_utilization = self.depot_busy_time.mean(dim=1) / self.time_horizon
        
        # Calculate average queue length
        avg_queue_length = torch.zeros(self.batch_size, device=self.device)
        mask = self.current_time > 0
        avg_queue_length[mask] = self.sum_queue_length_time[mask] / self.current_time[mask]
        
        # Calculate average queuing time
        avg_queue_time = self.vehicle_queuing_time.sum(dim=1) / self.num_vehicles
        
        # Calculate maximum queue time
        max_queue_time = self.vehicle_queuing_time.max(dim=1)[0]
        
        # Calculate average charging events
        avg_charge_events = self.num_charge_events.float() / self.num_vehicles
        
        # Calculate average charge per event
        avg_charge_per_event = torch.zeros(self.batch_size, device=self.device)
        mask = self.num_charge_events > 0
        avg_charge_per_event[mask] = self.total_charge_energy[mask] / self.num_charge_events[mask].float()
        
        # Collect all metrics
        metrics = {
            # Service metrics
            "percent_served": percent_served.cpu().item(),
            "avg_first_response": avg_first_response.cpu().item(),
            
            # Utilization metrics
            "vehicle_utilization": vehicle_utilization.cpu().item(),
            "avg_travel_time": avg_travel_time.cpu().item(),
            "avg_wait_time": avg_wait_time.cpu().item(),
            "avg_station_utilization": avg_station_utilization.cpu().item(),
            
            # Queue metrics
            "avg_queue_length": avg_queue_length.cpu().item(),
            "max_queue_length": self.max_queue_length.cpu().item(),
            "avg_queue_time": avg_queue_time.cpu().item(),
            "max_queue_time": max_queue_time.cpu().item(),
            
            # Charging event metrics
            "avg_charge_events": avg_charge_events.cpu().item(),
            "avg_charge_per_event": avg_charge_per_event.cpu().item(),
            
            # Energy metrics
            "total_travel_energy": self.total_travel_energy.cpu().item(),
            # "total_supply_energy": self.total_supply_energy.cpu().item(),
            "total_charge_energy": self.total_charge_energy.cpu().item()
        }
        
        return metrics
        
    def visualize_state_batch(self, visualized_batch: torch.BoolTensor):
        if self.episode_step == 0 or UNEQUAL_INTERVAL:
                # 处理初始状态或不等间隔的情况
    # 直接可视化当前状态，不需要插值
            for batch in range(1):
                if visualized_batch[batch] == False:
                    continue


                            # 新增：之后改 记录掩码历史
                if self.fname is not None:
                    for mask_name in self.mask_names:
                        mask_value = self.get_current_mask_value(mask_name, batch)
                        if mask_value is not None:
                            self.mask_histories[mask_name][batch].append(mask_value)

                self.visualize_state(batch, 
                                    self.current_time[batch].item(), 
                                    self.vehicle_curr_battery[batch])
                                    #self.loc_curr_battery[batch], 
                                   # ((self.loc_curr_battery[batch] - self.loc_min_battery) <= 0.0).sum().item(),
                                    #self.vehicle_unavail_time[batch])
#                 处理初始状态或不等间隔情况
# 直接可视化当前状态，不需要插值
        else:
                # 获取前一个状态的信息
            for batch in range(1):
                if visualized_batch[batch] == False:
                    continue
                prev_time      = self.time_history[batch][-1]
                curr_time      = self.current_time[batch].item()
                    # 复制前一个状态的电量历史
                prev_veh_batt  = copy.deepcopy(self.vehicle_batt_history[batch])
                #prev_loc_batt  = copy.deepcopy(self.loc_batt_history[batch])
                #prev_down_locs = self.down_history[batch][-1]
                            # 之后改 append 添加记录 新增：记录当前掩码状态
                if self.fname is not None:
                    for mask_name in self.mask_names:
                        mask_value = self.get_current_mask_value(mask_name, batch)
                        if mask_value is not None:
                            self.mask_histories[mask_name][batch].append(mask_value)
                 # 常规状态更新（带插值）
                curr_veh_unavail_time = self.vehicle_unavail_time[batch].clamp(0.0).detach().clone()
                time_interval  = curr_time - prev_time
                # 计算时间间隔
                # 之后改 这里改了下面4行避开0
                # 如果 time_interval 太小（接近 0），可以做跳过，或者给出一个缺省处理
                # 处理极小时间间隔的情况
                if abs(time_interval) < 1e-9:
                    # 在这里跳过当前循环，或者将 time_interval 设置为一个非常小的值，防止除零
                    time_interval = 1e-9  # 给 time_interval 一个默认值，避免后面除零  # 设置最小时间间隔
                    return  # 时间间隔太小则直接返回
                
                #                 # 假设两个时间点非常接近
                # curr_time = 1.0000001
                # prev_time = 1.0000000
                
                # # 计算时间间隔
                # time_interval = curr_time - prev_time  # = 1e-7
                
                # # 如果不处理，后续的插值计算中可能出现问题：
                # # ratio = dt / time_interval  # 可能得到非常大的数字或导致除零错误
                
                # # 使用保护机制：
                # if abs(time_interval) < 1e-9:
                #     time_interval = 1e-9  # 确保最小间隔
                #     return               # 跳过这次更新


                dts = np.arange(OUTPUT_INTERVAL, time_interval, OUTPUT_INTERVAL).tolist()

         # 之后改 这里改了下面4行避开0
                for dt in dts:
                # 计算时确保不会除以零
                    if time_interval != 0:  # 确保不发生除零
                        ratio = dt / time_interval
                    else:
                        ratio = 0  # 或者设置为默认值

                if len(dts) != 0:
                    if time_interval - dts[-1] <= OUTPUT_INTERVAL / 4:
                        dts[-1] = time_interval
                    else:
                        dts.append(time_interval)
                else:
                    dts.append(time_interval)
#                     计算需要插值的时间点
# 确保平滑的动画效果
                for dt in dts:
                    ratio = dt / time_interval
                        # 计算车辆电量插值
                    curr_veh_batt = torch.tensor([
                        interpolate_line(prev_veh_batt[vehicle_id][-1], self.vehicle_curr_battery[batch][vehicle_id].item(), ratio)
                        for vehicle_id in range(self.num_vehicles)
                    ]) # [num_vehicles]
                        # 计算基站电量插值
                    curr_loc_batt = torch.zeros(self.num_locs, device=self.device)  # 创建零张量代替
                    # curr_loc_batt = torch.tensor([
                    #     interpolate_line(prev_loc_batt[loc_id][-1], self.loc_curr_battery[batch][loc_id].item(), ratio) 
                    #     for loc_id in range(self.num_locs)
                    # ]) # [num_locs]
                    curr_down_locs = 0  # 简单设置为0，因为没有loc_curr_battery
                    #curr_down_locs = interpolate_line(prev_down_locs, ((self.loc_curr_battery[batch] - self.loc_min_battery) <= 0.0).sum().item(), ratio) # [1]
                    veh_unavail_time = curr_veh_unavail_time + (time_interval - dt)
                    self.visualize_state(batch, prev_time + dt, curr_veh_batt, None, curr_down_locs, veh_unavail_time)
# # 假设场景：
# prev_time = 1.0          # 上一时刻
# curr_time = 2.0          # 当前时刻
# OUTPUT_INTERVAL = 0.2    # 输出间隔

# # 生成时间点
# time_interval = 1.0      # 时间间隔
# dts = [0.2, 0.4, 0.6, 0.8, 1.0]  # 插值时间点

# # 对于某个车辆：
# prev_battery = 80.0      # 上一状态电量
# curr_battery = 60.0      # 当前状态电量

# # 插值计算：
# for dt in dts:
#     ratio = dt / time_interval
#     interpolated_battery = 80.0 + ratio * (60.0 - 80.0)
#     # 生成 [80, 76, 72, 68, 64, 60] 的平滑过渡
#     # 这里的插值计算是线性的
#     # 通过插值生成的电量值用于可视化
#     # 车辆在每个时间点的电量变化
    def visualize_state(self, batch: int, curr_time: float, curr_veh_batt: torch.FloatTensor,
                        curr_loc_batt: torch.FloatTensor = None, curr_down_locs: float = 0,
                        veh_unavail_time: torch.FloatTensor = None) -> None:
        #-----------------
        # battery history
        #-----------------
        if curr_loc_batt is None:
            # 如果未提供，使用空张量
            curr_loc_batt = torch.zeros(self.num_locs, device=self.device)
            
        if veh_unavail_time is None:
            # 如果未提供，使用零张量
            veh_unavail_time = torch.zeros_like(self.vehicle_unavail_time[batch])
        # 记录队列成本数据
        if self.fname is not None and hasattr(self, 'queue_cost_log'):
                        # 只记录batch=0的数据
            if batch == 0:
                current_queue_snapshot = self.charge_queue[batch].clone().detach().cpu()
                # 只记录状态实际发生变化的时间点，避免重复记录相同状态
                if not self.queue_cost_log or not torch.allclose(self.queue_cost_log[-1][1], current_queue_snapshot):
                    self.queue_cost_log.append((curr_time, current_queue_snapshot))
    
        self.time_history[batch].append(curr_time)
#         将当前时间点添加到时间历史记录中
# 按批次(batch)存储
        for vehicle_id in range(self.num_vehicles):
            self.vehicle_batt_history[batch][vehicle_id].append(curr_veh_batt[vehicle_id].item())
#             记录每辆车在当前时刻的电量
# 按批次和车辆ID分别存储
#         for loc_id in range(self.num_locs):
#             self.loc_batt_history[batch][loc_id].append(curr_loc_batt[loc_id].item())
# #             记录每个基站在当前时刻的电量
# # 按批次和基站ID分别存储
#         self.down_history[batch].append(curr_down_locs)

        if self.fname is not None:
            for action_name in self.action_names:
                action_value = self.get_current_action_value(action_name, batch)
                if action_value is not None:
                    self.action_histories[action_name][batch].append(action_value)
            # 记录队列相关的历史
            # 计算当前排队和充电的车辆
            _charge_phase = (self.vehicle_phase[batch] == self.phase_id["charge"])
            _at_depot = self.is_depot(self.vehicle_position_id[batch])
            _queued = (self.charge_queue[batch].sum(0) > 1)  # [num_vehicles]
            _charging = _charge_phase & _at_depot & ~_queued
            
            # 存储到历史记录中
            self.queue_histories["charge_queue"][batch].append(self.charge_queue[batch].clone().detach().cpu())
            self.queue_histories["queued_vehicles"][batch].append(_queued.clone().detach().cpu())
            self.queue_histories["vehicle_unavail_time"][batch].append(self.vehicle_unavail_time[batch].clone().detach().cpu())
            self.queue_histories["charging_vehicles"][batch].append(_charging.clone().detach().cpu())
#记录当前时刻电量耗尽的基站数量

# 假设场景：
# batch = 0
# num_vehicles = 2
# num_locs = 3
# curr_time = 1.5
# curr_veh_batt = [80.0, 70.0]      # 两辆车的当前电量
# curr_loc_batt = [90.0, 85.0, 75.0] # 三个基站的当前电量
# curr_down_locs = 1                 # 一个基站电量耗尽

# # 数据存储结构：
# self.time_history = [
#     [0.0, 0.5, 1.0, 1.5],  # batch 0的时间点
#     [...],                  # batch 1的时间点
# ]

# self.vehicle_batt_history = [
#     [  # batch 0
#         [100, 95, 90, 80],  # 车辆0的电量历史
#         [100, 90, 80, 70]   # 车辆1的电量历史
#     ],
#     [...],  # batch 1
# ]

# self.loc_batt_history = [
#     [  # batch 0
#         [100, 95, 92, 90],  # 基站0的电量历史
#         [100, 93, 89, 85],  # 基站1的电量历史
#         [100, 90, 82, 75]   # 基站2的电量历史
#     ],
#     [...],  # batch 1
# ]

# self.down_history = [
#     [0, 0, 0, 1],  # batch 0的电量耗尽基站数量历史
#     [...],         # batch 1
# ]
        #---------------
        # visualziation
        #---------------
        if SAVE_PICTURE:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(ncols=2, nrows=3, width_ratios=[1, 1.5])
            ax = fig.add_subplot(gs[:, 1])
#             创建一个大小为20x12的图形
# 设置2列3行的网格布局
# 右侧区域用于显示主要场景
            # current state
            # 转换数据为numpy格式
            #loc_battery = torch2numpy(curr_loc_batt)         # 基站电量
            vehicle_battery = torch2numpy(curr_veh_batt)     # 车辆电量
            #loc_cap = torch2numpy(self.loc_cap[batch])       # 基站容量
            loc_coords = torch2numpy(self.loc_coords[batch])  # 基站坐标
            depot_coords = torch2numpy(self.depot_coords[batch]) # 充电站坐标
            coords = np.concatenate((loc_coords, depot_coords), 0)          # [num_nodes x coord_dim]
            vehicle_cap = torch2numpy(self.vehicle_cap[batch])              # [num_vehicles]
            x_loc = loc_coords[:, 0]; y_loc = loc_coords[:, 1]
            x_depot = depot_coords[:, 0]; y_depot = depot_coords[:, 1]

        # 获取已访问客户点的状态
            visited_locations = self.loc_visited[batch].cpu().numpy()
            # visualize nodes
            # 绘制节点时传入节点ID
            for id in range(self.num_locs):
                # ratio = loc_battery[id] / loc_cap[id]
                # add_base(x_loc[id], y_loc[id], ratio, ax, node_id=id)  # 添加node_id参数
                add_base(x_loc[id], y_loc[id], 1.0, ax, node_id=id)  # Always show full battery
                # 为已访问的客户点添加对勾标记
                if visited_locations[id]:
                    # 在客户点旁边添加对勾标记
                    ax.plot(x_loc[id] + 0.02, y_loc[id] + 0.02, marker='v', color='green', 
                            markersize=15, markeredgecolor='black', zorder=10)
            

            # 绘制充电站
            for id in range(len(x_depot)):
                depot_id = id + self.num_locs  # 充电站ID从num_locs开始
                ax.scatter(x_depot[id], y_depot[id], marker="*", c="black", s=200, zorder=3)
                ax.text(x_depot[id], y_depot[id], f"{depot_id}", 
                        fontsize=8, ha='right', va='bottom', color='black')

            # 绘制车辆时传入车辆ID
            cmap = get_cmap(self.num_vehicles)
            for vehicle_id in range(self.num_vehicles):
                ratio = vehicle_battery[vehicle_id] / vehicle_cap[vehicle_id]
                vehicle_phase = self.vehicle_phase[batch][vehicle_id]
                vehicle_position_id = self.vehicle_position_id[batch][vehicle_id]
                
                if vehicle_phase != self.phase_id["move"]:
                    vehicle_x = coords[vehicle_position_id, 0]
                    vehicle_y = coords[vehicle_position_id, 1]
                    add_vehicle(vehicle_x, vehicle_y, ratio, 
                            vehicle_battery[vehicle_id], 
                            cmap(vehicle_id), ax, 
                            vehicle_id=vehicle_id)  # 添加vehicle_id参数
                else:
                    vehicle_position_id_prev = self.vehicle_position_id_prev[batch][vehicle_id]
                    speed = self.speed[batch]
                    start = coords[vehicle_position_id_prev, :]
                    end = coords[vehicle_position_id, :]
                    distance = np.linalg.norm(start - end)
                    curr_position = interpolate_line(start, end, 
                                    (1.0 - speed * veh_unavail_time[vehicle_id] / distance).item())
                    
                    # 绘制路径和车辆
                    ax.plot([start[0], curr_position[0]], [start[1], curr_position[1]], 
                            zorder=0, linestyle="-", color=cmap(vehicle_id))
                    ax.plot([curr_position[0], end[0]], [curr_position[1], end[1]], 
                            zorder=0, alpha=0.5, linestyle="--", color=cmap(vehicle_id))
                    add_vehicle(curr_position[0], curr_position[1], ratio,
                            vehicle_battery[vehicle_id], 
                            cmap(vehicle_id), ax,
                            vehicle_id=vehicle_id)  # 添加vehicle_id参数
            # 之后改        # Add phase indicator
            all_locations_visited = visited_locations.all()
            phase_text = "Final Phase (Moving Only)" if all_locations_visited else "Service Phase"
            ax.annotate(phase_text, xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=14, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
                        ha='left', va='top')
            
            ax.set_title(f"current time = {curr_time:.3f} h", y=-0.05, fontsize=18)
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
            ax.set_aspect(1)

            ax.set_title(f"current time = {curr_time:.3f} h", y=-0.05, fontsize=18)
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
            ax.set_aspect(1)

            #----------------------------
            # add history plot until now
            #----------------------------
            time_horizon = self.time_horizon.cpu().item()
            max_veh_batt = torch.ceil(self.vehicle_cap[batch].max() / 10).cpu().item() * 10
           # max_loc_batt = torch.ceil(self.loc_cap[batch].max() / 10).cpu().item() * 10
           # max_num_locs = math.ceil(self.num_locs / 10) * 10
#             获取时间范围上限
# 计算车辆和基站电量的最大值（向上取整到10的倍数）
# 计算基站数量上限
            # EV battery history
            ax_ev = fig.add_subplot(gs[0, 0])
            for vehicle_id in range(self.num_vehicles):
                ax_ev.plot(self.time_history[batch], list(self.vehicle_batt_history[batch][vehicle_id]), alpha=0.7, color=cmap(vehicle_id))
#                 显示每辆车的电量变化曲线
# 设置坐标轴范围和标签
#车辆电量历史图表（上图）
            ax_ev.set_xlim(0, time_horizon)
            ax_ev.set_ylim(0, max_veh_batt)
            ax_ev.get_xaxis().set_visible(False)
            ax_ev.axvline(x=self.time_history[batch][-1], ymin=-1.2, ymax=1, c="black", lw=1.5, zorder=0, clip_on=False)
            ax_ev.set_ylabel("EV battery (kWh)", fontsize=18)
            # 基站电量历史图表（中图）
            # 显示每个基站的电量变化曲线
            # Base station battery history
            # ax_base = fig.add_subplot(gs[1, 0])
            # for loc_id in range(self.num_locs):
            #     ax_base.plot(self.time_history[batch], list(self.loc_batt_history[batch][loc_id]), alpha=0.7)
            # ax_base.set_xlim(0, time_horizon)
            # ax_base.set_ylim(0, max_loc_batt)
            # ax_base.get_xaxis().set_visible(False)
            # ax_base.axvline(x=self.time_history[batch][-1], ymin=-1.2, ymax=1, c="black", lw=1.5, zorder=0, clip_on=False)
            # ax_base.set_ylabel("Base station battery (kWh)", fontsize=18)
            # # 电量耗尽基站数量图表（下图）
            # # Num. of downed base stations
            # ax_down = fig.add_subplot(gs[2, 0])
            # ax_down.plot(self.time_history[batch], self.down_history[batch])
            # ax_down.set_xlim(0, time_horizon)
            # ax_down.set_ylim(0, max_num_locs)
            # ax.axvline(
            #     x=self.time_history[batch][-1],   # 当前时间点
            #     ymin=-1.2, ymax=1,                # 线段范围
            #     c="black",                        # 黑色
            #     lw=1.5,                          # 线宽
            #     zorder=0,                        # 绘制层级
            #     clip_on=False                     # 允许超出边界
            # )
            # ax_down.set_xlabel("Time (h)", fontsize=18)
            # ax_down.set_ylabel("# downed base stations", fontsize=18)

            #---------------
            # save an image
            #---------------
            # 1. 调整图形布局
            fig.subplots_adjust(
                left=0.03,    # 左边界距离
                right=1,      # 右边界距离
                bottom=0.05,  # 底部边界距离
                top=0.98,     # 顶部边界距离
                wspace=0.05   # 子图之间的水平间距
            )
            fname = f"{self.fname}-{batch}/png/tour_state{self.episode_step}.png"
            os.makedirs(f"{self.fname}-{batch}/png", exist_ok=True)
            plt.savefig(fname, dpi=DPI)
            plt.close()

# 之后改 我怎么感觉多个地方都有save png的代码 
        self.episode_step += 1

    def output_gif(self):
        for batch in range(1):
            anim_type = "mp4"
            out_fname = f"{self.fname}-{batch}/EVRoute.{anim_type}"
            seq_fname = f"{self.fname}-{batch}/png/tour_state%d.png"
            output_animation(out_fname, seq_fname, anim_type)
    # def output_gif(self):
    #     for batch in range(1):
    #         anim_type = "mp4"
    #         out_fname = f"{self.fname}-{batch}/EVRoute.{anim_type}"
    #         # 之后改 这里把这个frame去掉 但是要出视频必须有图 那可以生成图之后再删掉 图
    #         png_dir = f"{self.fname}-{batch}/png"
    #         seq_fname = f"{png_dir}-{batch}/png/tour_state%d.png"
    #     try:
    #         # 生成动画
    #         output_animation(out_fname, seq_fname, anim_type)
            
    #         # 删除临时PNG文件
    #         import os
    #         import glob
            
    #         # 获取所有匹配的PNG文件
    #         png_files = glob.glob(f"{png_dir}/tour_state*.png")
            
    #         # 删除每个PNG文件
    #         for png_file in png_files:
    #             if os.path.exists(png_file):
    #                 os.remove(png_file)
            
    #         print(f"临时PNG文件已删除")
            
        # except Exception as e:
        #     print(f"发生错误: {str(e)}")


    def output_mask_history(self):
        """输出掩码历史数据，包括状态掩码和时序掩码的变化"""
        batch = 0  # 处理第一个批次
        output_dir = f"{self.fname}-sample{batch}"  # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
        
        # 准备数据结构
        mask_history_data = {
            "time": self.time_history[batch],     # 时间序列
            "masks": {}                           # 掩码历史数据
        }
        
        # 2. 收集每个掩码的历史数据
        # 遍历所有掩码类型，收集其历史数据
        for mask_name in self.mask_names:
            if len(self.mask_histories[mask_name][batch]) > 0:
                mask_history_data["masks"][mask_name] = self.mask_histories[mask_name][batch]

        # 3. 保存PKL文件
        pkl_path = f"{output_dir}/mask_history_data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(mask_history_data, f)
        
        # 生成可读的TXT文件
        txt_path = f"{output_dir}/mask_history_data_readable.txt"
        with open(txt_path, "w") as f:
            # 写入时间历史
            f.write("===== Time Points =====\n")
            f.write(f"State points: {[f'state_{i} ({t:.3f})' for i, t in enumerate(self.time_history[batch])]}\n\n")
            
            # 写入各掩码历史
            f.write("===== Mask History =====\n")
            for mask_name, mask_hist in mask_history_data["masks"].items():
                f.write(f"\n=== {mask_name} ===\n")
                
                # 根据掩码类型分类处理
                if mask_name in ["vehicle_position_id", "vehicle_phase"]:
                    # 车辆相关的多维数据
                    for time_idx, values in enumerate(mask_hist):
                        f.write(f"State_{time_idx} ({self.time_history[batch][time_idx]:.3f}):\n")
                        f.write(f"  Values for each vehicle: {values}\n")
                
                # 处理布尔型掩码
                elif mask_name in ["loc_is_down", "loc_is_full", "loc_is_normal",
                                "wait_vehicle", "at_depot", "at_loc",
                                "queued_vehicles", "charging_vehicles", "small_depots"]:
                    if mask_hist and isinstance(mask_hist[0], torch.Tensor):
                        num_items = mask_hist[0].numel()
                        item_label = ("车辆" if "vehicle" in mask_name or "charge" in mask_name or "queued" in mask_name 
                                    else "基站" if "loc" in mask_name 
                                    else "充电站" if "depot" in mask_name 
                                    else "项目")
                        for time_idx, values_tensor in enumerate(mask_hist):
                            values_list = values_tensor.tolist()
                            f.write(f"State_{time_idx} ({self.time_history[batch][time_idx]:.3f}):\n")
                            status_str = ", ".join([f"{item_label}{i}={val}" for i, val in enumerate(values_list)])
                            f.write(f"  状态: [{status_str}]\n")
                            
                            # 添加统计信息
                            if mask_name == "loc_is_down":
                                down_count = sum(1 for x in values_list if x)
                                f.write(f"  总断电基站数: {down_count}\n")
                            elif mask_name == "queued_vehicles":
                                queued_count = sum(1 for x in values_list if x)
                                f.write(f"  总排队车辆数: {queued_count}\n")
                            elif mask_name == "charging_vehicles":
                                charging_count = sum(1 for x in values_list if x)
                                f.write(f"  总充电车辆数: {charging_count}\n")
            
                elif mask_name in ["charge_queue"]:
                    # 充电队列数据
                    for time_idx, queue_tensor in enumerate(mask_hist):
                        f.write(f"State_{time_idx} ({self.time_history[batch][time_idx]:.3f}):\n")
                        queue_list = queue_tensor.tolist()
                        f.write(f"  Queue status (充电站 x 车辆):\n")
                        for depot_idx, depot_queue in enumerate(queue_list):
                            f.write(f"    充电站 {depot_idx}: {depot_queue}\n")
                
                elif mask_name in ["skip", "end", "next_vehicle_mask"]:
                    # 简单布尔值或单值掩码
                    for time_idx, value in enumerate(mask_hist):
                        f.write(f"State_{time_idx} ({self.time_history[batch][time_idx]:.3f}): {value}\n")
                
                else:
                    # 其他掩码类型
                    for time_idx, values in enumerate(mask_hist):
                        f.write(f"State_{time_idx} ({self.time_history[batch][time_idx]:.3f}): {values}\n")
                
                f.write("\n")
                    

        # 5. 生成可视化图表
        if SAVE_PICTURE:
            self._visualize_mask_history(batch, mask_history_data)

            #             # 输出数据示例
            # {
            #     "time": [0.0, 0.5, 1.0, 1.5],  # 时间点
            #     "masks": {
            #         "vehicle_position_id": [  # 车辆位置历史
            #             [0, 1, 2],  # t=0.0时刻
            #             [1, 2, 3],  # t=0.5时刻
            #             [2, 3, 4]   # t=1.0时刻
            #         ],
            #         "loc_is_down": [  # 基站断电状态历史
            #             [False, False, True],
            #             [False, True, True],
            #             [True, True, True]
            #         ]
            #     }
            # }

    def _visualize_mask_history(self, batch, mask_history_data):
        """生成掩码历史的可视化图表"""
        time_points = mask_history_data["time"]
        masks = mask_history_data["masks"]
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10)) # 创建15x10大小的图表
        
        # 计算需要的子图数量
        num_plots = len([m for m in masks.keys() if m in ["loc_is_down", "vehicle_phase", "charge_queue"]])
        if num_plots == 0:
            plt.close()
            return
            
        # 创建子图
        plot_idx = 1
        
        # 1. 电量耗尽基站数量变化
        if "loc_is_down" in masks:
                # 统计每个时间点电量耗尽的基站数量
            ax = plt.subplot(num_plots, 1, plot_idx)
            down_counts = [sum(1 for x in state if x) for state in masks["loc_is_down"]]
            
            # 确保数据长度匹配
            min_len = min(len(time_points), len(down_counts))
            time_points_plot = time_points[:min_len]
            down_counts_plot = down_counts[:min_len]
                # 绘制红色折线图显示变化趋势
            ax.plot(time_points_plot, down_counts_plot, 'r-', label='Down Locations')
            ax.set_ylabel("Count")
            ax.set_title("Number of Down Locations Over Time")
            ax.grid(True)
            plot_idx += 1
        
        # 2. 车辆阶段变化
        if "vehicle_phase" in masks:
            ax = plt.subplot(num_plots, 1, plot_idx)
            phase_data = np.array(masks["vehicle_phase"])
            
            # 确保数据长度匹配
            min_len = min(len(time_points), phase_data.shape[0])
            time_points_plot = time_points[:min_len]
            phase_data_plot = phase_data[:min_len]
                # 为每辆车绘制一条线，显示其阶段变化
            for vehicle_id in range(phase_data_plot.shape[1]):
                ax.plot(time_points_plot, phase_data_plot[:, vehicle_id], 
                       label=f'Vehicle {vehicle_id}')
            ax.set_ylabel("Phase")
            ax.set_title("Vehicle Phases Over Time")
            ax.legend()
            ax.grid(True)
            plot_idx += 1
        
        # 3. 充电队列长度变化
        if "charge_queue" in masks:
            ax = plt.subplot(num_plots, 1, plot_idx)
                # 统计每个时间点的充电队列长度
            # Fix: Convert tensors to numpy arrays and use element-wise comparison
            queue_lengths = []
            for q in masks["charge_queue"]:
                if isinstance(q, torch.Tensor):
                    # Convert tensor to numpy and count non-zero elements
                    q_np = q.numpy()
                    queue_lengths.append(np.sum(q_np > 0))
                else:
                    # If already numpy array or list
                    queue_lengths.append(len([x for x in q if x > 0]))
            
            # 确保数据长度匹配
            min_len = min(len(time_points), len(queue_lengths))
            time_points_plot = time_points[:min_len]
            queue_lengths_plot = queue_lengths[:min_len]
            
            ax.plot(time_points_plot, queue_lengths_plot, 'b-', label='Queue Length')
            ax.set_xlabel("Time")
            ax.set_ylabel("Length")
            ax.set_title("Charging Queue Length Over Time")
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.fname}-sample{batch}/mask_history_plots.png", dpi=DPI)
        plt.close()
        

        #         # 输入数据示例
        # mask_history_data = {
        #     "time": [0.0, 0.5, 1.0, 1.5],
        #     "masks": {
        #         "loc_is_down": [
        #             [True, False, False],  # t=0.0时刻
        #             [True, True, False],   # t=0.5时刻
        #             [True, True, True]     # t=1.0时刻
        #         ],
        #         "vehicle_phase": [
        #             [0, 1, 2],  # t=0.0时刻
        #             [1, 2, 0],  # t=0.5时刻
        #             [2, 0, 1]   # t=1.0时刻
        #         ]
        #     }
        # }
        
        # 生成的图表将显示：
        # 1. 电量耗尽基站数量：1->2->3
        # 2. 每辆车的阶段变化
        # 3. 充电队列变化趋势

    def output_action_history(self):
        """输出动作和系统状态历史数据，提供完整的决策轨迹"""
        batch = 0  # 处理第一个批次
        output_dir = f"{self.fname}-sample{batch}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据结构
        action_history_data = {
            "time": self.time_history[batch],
            "actions": {}
        }
        
        # 收集每个动作的历史数据
        for action_name in self.action_names:
            if batch < len(self.action_histories[action_name]) and self.action_histories[action_name][batch]:
                action_history_data["actions"][action_name] = self.action_histories[action_name][batch]
        
        # 保存PKL文件
        pkl_path = f"{output_dir}/action_history_data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(action_history_data, f)
        
        # 生成可读的TXT文件
        txt_path = f"{output_dir}/action_history_data_readable.txt"
        with open(txt_path, "w") as f:
            # 写入时间历史
            f.write("===== Decision Timeline =====\n\n")
            
            # 检查数据长度
            time_points = action_history_data["time"]
            
            # 写入每个时间点的详细行动
            for t_idx, t in enumerate(time_points):
                f.write(f"===== State {t_idx} (Time: {t:.3f} h) =====\n")
                
                # 第一部分：动作信息
                f.write("--- Action Information ---\n")
                
                # 获取当前时间点的行动信息
                vehicle_id = None
                node_id = None
                is_wait = None
                
                if "curr_vehicle_id" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["curr_vehicle_id"]):
                    vehicle_id = action_history_data["actions"]["curr_vehicle_id"][t_idx]
                    f.write(f"Acting Vehicle: {vehicle_id}\n")
                
                if "next_node_id" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["next_node_id"]):
                    node_id = action_history_data["actions"]["next_node_id"][t_idx]
                    node_type = "充电站" if node_id >= self.num_locs else "基站"
                    node_local_id = node_id - self.num_locs if node_id >= self.num_locs else node_id
                    f.write(f"Target Node: {node_id} ({node_type} {node_local_id})\n")
                
                if "do_wait" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["do_wait"]):
                    is_wait = action_history_data["actions"]["do_wait"][t_idx]
                    f.write(f"Wait Action: {'是' if is_wait else '否'}\n")
                
                # 通过训练 (train.py) 学习选择能够带来更高长期奖励（或更低成本）的节点。成本函数 (state.get_rewards() 计算，包含 tour_length 和 pena
                # lty_empty_locs）会惩罚过长的路线和基站断电。因此，模型会学习到：如果一直停留在充电站而不去服务基站，会导致 penalty_empty_locs 增加，从而降
                # 低奖励。为了获得更好的奖励，模型倾向于选择移动到需要服务的基站（如果可行）。
                                                               
                if "travel_time" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["travel_time"]):
                    travel_t = action_history_data["actions"]["travel_time"][t_idx]
                    f.write(f"Planned Travel Time: {travel_t:.3f} h\n")
                
                if "charge_time" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["charge_time"]):
                    charge_t = action_history_data["actions"]["charge_time"][t_idx]
                    f.write(f"Planned Charge/Supply Time: {charge_t:.3f} h\n")
                
                if "elapsed_time" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["elapsed_time"]):
                    elapsed_t = action_history_data["actions"]["elapsed_time"][t_idx]
                    f.write(f"Actual Elapsed Time: {elapsed_t:.3f} h\n")
                
                # 第二部分：系统状态
                f.write("\n--- System State ---\n")
                
                if "tour_length" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["tour_length"]):
                    tour_len = action_history_data["actions"]["tour_length"][t_idx]
                    f.write(f"Total Tour Length: {tour_len:.3f}\n")
                
                if "penalty_empty_locs" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["penalty_empty_locs"]):
                    penalty = action_history_data["actions"]["penalty_empty_locs"][t_idx]
                    f.write(f"Accumulated Penalty: {penalty:.5f}\n")
                
                # 第三部分：派生状态
                f.write("\n--- Derived State ---\n")
                
                if "down_locs_count" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["down_locs_count"]):
                    down_count = action_history_data["actions"]["down_locs_count"][t_idx]
                    f.write(f"Down Locations: {down_count}\n")
                
                if "queued_vehicles_count" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["queued_vehicles_count"]):
                    queue_count = action_history_data["actions"]["queued_vehicles_count"][t_idx]
                    f.write(f"Queued Vehicles: {queue_count}\n")
                
                if "charging_vehicles_count" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["charging_vehicles_count"]):
                    charging_count = action_history_data["actions"]["charging_vehicles_count"][t_idx]
                    f.write(f"Charging Vehicles: {charging_count}\n")
                
                if "supplying_vehicles_count" in action_history_data["actions"] and t_idx < len(action_history_data["actions"]["supplying_vehicles_count"]):
                    supplying_count = action_history_data["actions"]["supplying_vehicles_count"][t_idx]
                    f.write(f"Supplying Vehicles: {supplying_count}\n")
                
                f.write("\n")
        
        # 生成可视化图表
        if SAVE_PICTURE:
            self._visualize_action_history(batch, action_history_data)
# A. 动作信息 (Action Information)
#                         - Acting Vehicle (执行动作的车辆)
#             - Target Node (目标节点)
#             - Wait Action (是否等待)
#             - Planned Travel Time (计划行驶时间)
#             - Planned Charge/Supply Time (计划充电/供电时间)
#             - Actual Elapsed Time (实际耗时)
#             B. 系统状态 (System State)
# - Total Tour Length (总行驶距离)
# - Accumulated Penalty (累积惩罚值)
# C. 派生状态 (Derived State)
# - Down Locations (断电基站数量)
# - Queued Vehicles (排队车辆数量)
# - Charging Vehicles (充电中车辆数量)
# - Supplying Vehicles (供电中车辆数量)

# # 某个时间点的输出示例：
# ===== State 10 (Time: 1.500 h) =====
# --- Action Information ---
# Acting Vehicle: 2
# Target Node: 5 (基站 5)
# Wait Action: 否
# Planned Travel Time: 0.300 h
# Planned Charge/Supply Time: 0.500 h
# Actual Elapsed Time: 0.300 h

# --- System State ---
# Total Tour Length: 25.500
# Accumulated Penalty: 0.02500

# --- Derived State ---
# Down Locations: 1
# Queued Vehicles: 2
# Charging Vehicles: 1
# Supplying Vehicles: 3

# 决策追踪：记录系统每个决策点的完整信息
# 性能分析：通过历史数据分析系统表现
# 调试支持：帮助定位和解决问题

    def output_mask_calc_history(self):
        """将详细的掩码计算过程输出到文件"""
        batch = 0  # 处理第一个批次
        
        # 检查是否有日志数据
        if not hasattr(self, 'mask_calc_log') or not self.mask_calc_log[batch]:
            return
        
        # 创建输出目录
        output_dir = f"{self.fname}-sample{batch}"
        os.makedirs(output_dir, exist_ok=True)
        txt_path = f"{output_dir}/mask_calculation_process.txt"
        
        # 掩码张量格式化函数
        def format_mask(mask_tensor, num_locs):
            if mask_tensor is None:
                return "N/A"
            
            mask_list = mask_tensor.int().tolist()
            
            # 区分基站和充电站部分
            loc_part = mask_list[:num_locs]
            depot_part = mask_list[num_locs:]
            
            loc_str = ", ".join([f"基站{i}:{v}" for i, v in enumerate(loc_part)])
            depot_str = ", ".join([f"充电站{i}:{v}" for i, v in enumerate(depot_part)])
            
            return (f"基站:\n"
                    f"  [{loc_str}]\n" 
                    f"充电站:\n"
                    f"  [{depot_str}]")
        
        # 写入文件
        with open(txt_path, "w") as f:
            f.write("===== 掩码计算过程详细记录 =====\n\n")
            
            for log_entry in self.mask_calc_log[batch]:
                # 写入步骤基本信息
                f.write(f"步骤 {log_entry['step']} (时间: {log_entry['time']:.3f})\n")
                f.write(f"决策车辆: {log_entry['acting_vehicle']}\n")
                f.write(f"当前位置: {log_entry['current_node']}\n\n")
                
                # 写入初始掩码
                f.write("初始掩码 (所有节点可访问):\n")
                f.write(f"  {format_mask(log_entry['rule_masks'].get('initial'), self.num_locs)}\n\n")
                
                # 规则1: 放电限制
                f.write("规则1: 电量达到放电下限时必须返回充电站\n")
                f.write(f"  是否触发: {log_entry['intermediate_results'].get('discharge_limit_triggered', False)}\n")
                f.write(f"  应用后: {format_mask(log_entry['rule_masks'].get('after_discharge_limit'), self.num_locs)}\n\n")
                
                # 规则2: 无法返回节点
                f.write("规则2: 屏蔽无法返回的节点 (电量/时间不足)\n")
                f.write(f"  应用后: {format_mask(log_entry['rule_masks'].get('after_unreturnable'), self.num_locs)}\n\n")
                
                # 规则3: 已访问节点
                f.write("规则3: 屏蔽已被其他车辆访问的节点\n")
                
                # 显示原始车辆位置信息（包括充电站）
                reserved_nodes = log_entry['intermediate_results'].get('reserved_nodes')
                if reserved_nodes is not None:
                    f.write("  原始车辆位置 (包括充电站):\n")
                    f.write(f"  {format_mask(reserved_nodes, self.num_locs)}\n")
                
                # 显示重置充电站后的位置信息
                reset_nodes = log_entry['intermediate_results'].get('reserved_nodes_after_reset')
                if reset_nodes is not None:
                    f.write("  重置充电站后的车辆位置:\n")
                    f.write(f"  {format_mask(reset_nodes, self.num_locs)}\n")
                
                # 显示规则应用前的掩码状态
                before_mask = log_entry['rule_masks'].get('before_visited_locs')
                if before_mask is not None:
                    f.write("  应用规则前的掩码:\n")
                    f.write(f"  {format_mask(before_mask, self.num_locs)}\n")
                
                # 显示应用规则后的结果
                after_mask = log_entry['rule_masks'].get('after_visited_locs')
                if after_mask is not None:
                    f.write("  应用规则后的掩码:\n")
                    f.write(f"  {format_mask(after_mask, self.num_locs)}\n\n")
                
                # 新增: 规则3.5: 屏蔽已访问过的客户点
                f.write("规则3.5: 屏蔽已访问过的客户点\n")
                
                # 显示已访问客户点信息
                visited_locations = log_entry['intermediate_results'].get('visited_locations')
                if visited_locations is not None:
                    f.write("  已访问客户点:\n")
                    visited_str = ", ".join([f"基站{i}:{v}" for i, v in enumerate(visited_locations.tolist())])
                    f.write(f"  [{visited_str}]\n")
                
                # 显示扩展后的掩码
                padded_mask = log_entry['intermediate_results'].get('padded_visited_mask')
                if padded_mask is not None:
                    f.write("  扩展后的已访问掩码:\n")
                    f.write(f"  {format_mask(padded_mask, self.num_locs)}\n")
                
                # 显示规则应用前的掩码
                before_rule35_mask = log_entry['rule_masks'].get('before_rule3_5')
                if before_rule35_mask is not None:
                    f.write("  应用规则3.5前的掩码:\n")
                    f.write(f"  {format_mask(before_rule35_mask, self.num_locs)}\n")
                
                # 显示规则应用后的掩码
                after_rule35_mask = log_entry['rule_masks'].get('after_rule3_5')
                if after_rule35_mask is not None:
                    f.write("  应用规则3.5后的掩码:\n")
                    f.write(f"  {format_mask(after_rule35_mask, self.num_locs)}\n\n")
                
                # 规则4: 充电站到充电站
                f.write("规则4: 禁止从充电站直接到另一个充电站\n")
                f.write(f"  当前在充电站: {log_entry['intermediate_results'].get('at_depot', False)}\n")
                f.write(f"  应用后: {format_mask(log_entry['rule_masks'].get('after_depot_to_depot'), self.num_locs)}\n\n")
                
                # 规则5: 低功率充电站
                f.write("规则5: 屏蔽低功率充电站\n")
                small_depots = log_entry['intermediate_results'].get('small_depots')
                if small_depots is not None:
                    small_depots_indices = [i for i, x in enumerate(small_depots.tolist()) if x]
                    f.write(f"  低功率站索引: {small_depots_indices}\n")
                f.write(f"  应用后: {format_mask(log_entry['rule_masks'].get('after_small_depots'), self.num_locs)}\n\n")
                
                # 规则6: 跳过批次
                f.write("规则6: 跳过的批次仅允许留在原地\n")
                f.write(f"  是否跳过: {log_entry['intermediate_results'].get('is_skipped', False)}\n")
                f.write(f"  应用后: {format_mask(log_entry['rule_masks'].get('after_skipped'), self.num_locs)}\n\n")
                
                # 最终掩码
                f.write("最终掩码 (1=可访问, 0=禁止访问):\n")
                f.write(f"  {format_mask(log_entry['final_mask'], self.num_locs)}\n")
                f.write("-" * 80 + "\n\n")
                
    def _visualize_action_history(self, batch, action_history_data):
        """生成动作历史的可视化图表"""
        time_points = action_history_data["time"]
        actions = action_history_data["actions"]
        
        # 计算需要的子图数量
        key_metrics = ["tour_length", "penalty_empty_locs", "down_locs_count", 
                    "queued_vehicles_count", "charging_vehicles_count", "supplying_vehicles_count"]
        available_metrics = [k for k in key_metrics if k in actions and len(actions[k]) > 0]
        
        if not available_metrics:
            return
        
        # 创建图表
        fig = plt.figure(figsize=(15, 3*len(available_metrics)))
        plt.suptitle("System Performance Metrics Over Time", fontsize=16)
        
        # 为每个指标创建子图
        for i, metric in enumerate(available_metrics):
            ax = plt.subplot(len(available_metrics), 1, i+1)
            
            # 获取数据
            if metric in actions and len(actions[metric]) > 0:
                data = actions[metric]
                min_len = min(len(time_points), len(data))
                
                # 选择合适的绘图颜色和样式
                if metric == "tour_length":
                    color = "blue"
                    label = "Tour Length"
                elif metric == "penalty_empty_locs":
                    color = "red"
                    label = "Penalty"
                elif metric == "down_locs_count":
                    color = "purple"
                    label = "Down Locations"
                elif metric == "queued_vehicles_count":
                    color = "orange"
                    label = "Queued Vehicles"
                elif metric == "charging_vehicles_count":
                    color = "green"
                    label = "Charging Vehicles"
                elif metric == "supplying_vehicles_count":
                    color = "teal"
                    label = "Supplying Vehicles"
                else:
                    color = "gray"
                    label = metric
                
                # 绘制数据
                ax.plot(time_points[:min_len], data[:min_len], color=color, marker='o', label=label)
                ax.set_xlabel("Time (h)")
                ax.set_ylabel(label)
                ax.grid(True)
                ax.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f"{self.fname}-sample{batch}/action_history_plots.png", dpi=DPI)
        plt.close()
# key_metrics = [
#     "tour_length",            # 总行驶距离
#     "penalty_empty_locs",     # 基站断电惩罚值
#     "down_locs_count",        # 断电基站数量
#     "queued_vehicles_count",  # 排队车辆数量
#     "charging_vehicles_count",# 充电中车辆数量
#     "supplying_vehicles_count"# 供电中车辆数量
# ]
    def output_batt_history(self):
        """输出电池电量、断电基站数量的历史数据到 PKL 和可读 TXT 文件"""
        batch = 0  # 仅处理第一个批次 (因为可视化通常只针对一个样本)
        if not hasattr(self, 'time_history') or not self.time_history[batch]:
            print("警告: 历史数据为空，无法输出电池历史。")
            return

        output_dir = f"{self.fname}-sample{batch}"
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

        # --- 1. 准备要保存的数据 ---
        history_data = {
            "time": self.time_history[batch],
            "veh_batt": self.vehicle_batt_history[batch],
            # "loc_batt": self.loc_batt_history[batch],
            # "down_loc": self.down_history[batch]
        }

        # --- 2. 保存 PKL 文件 ---
        pkl_path = f"{output_dir}/history_data.pkl"
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(history_data, f)
        except Exception as e:
            print(f"保存 history_data.pkl 时出错: {e}")

        # --- 3. 生成可读的 TXT 文件 ---
        txt_path = f"{output_dir}/battery_history_readable.txt"  # 使用新文件名
        try:
            with open(txt_path, "w") as f:
                f.write("===== 电量历史详细记录 =====\n\n")

                time_points = history_data.get("time", [])
                veh_batt_history = history_data.get("veh_batt", [])
                loc_batt_history = history_data.get("loc_batt", [])
                down_loc_history = history_data.get("down_loc", [])

                # 检查历史记录长度是否一致 (理论上应该一致)
                num_steps = len(time_points)
                if not (len(veh_batt_history) == self.num_vehicles and
                        all(len(hist) == num_steps for hist in veh_batt_history) and
                        len(loc_batt_history) == self.num_locs and
                        all(len(hist) == num_steps for hist in loc_batt_history) and
                        len(down_loc_history) == num_steps):
                    f.write("错误：历史记录长度不一致，无法完整输出。\n")
                    print(f"警告: 历史记录长度不一致 (时间点: {num_steps})，TXT 输出可能不完整。")
                    # 打印长度信息帮助调试
                    print(f"  车辆电量历史长度: {[len(h) for h in veh_batt_history]}")
                    print(f"  基站电量历史长度: {[len(h) for h in loc_batt_history]}")
                    print(f"  断电基站历史长度: {len(down_loc_history)}")
                    # 截断到最短长度以尝试输出
                    min_len = num_steps
                    if veh_batt_history and self.num_vehicles > 0:
                        min_len = min(min_len, min(len(h) for h in veh_batt_history))
                    if loc_batt_history and self.num_locs > 0:
                        min_len = min(min_len, min(len(h) for h in loc_batt_history))
                    if down_loc_history:
                        min_len = min(min_len, len(down_loc_history))
                    num_steps = min_len  # 更新步数以安全迭代

                # 逐个时间步输出状态
                for step_idx in range(num_steps):
                    current_t = time_points[step_idx]
                    f.write(f"===== 步骤 {step_idx} (时间: {current_t:.3f} h) =====\n")

                    # 输出车辆电量
                    f.write("--- 车辆电量 ---\n")
                    if veh_batt_history and self.num_vehicles > 0:
                        veh_batt_str = ", ".join([f"车辆{i}: {veh_batt_history[i][step_idx]:.2f}"
                                                for i in range(self.num_vehicles) if step_idx < len(veh_batt_history[i])])
                        f.write(f"  [{veh_batt_str}]\n")
                    else:
                        f.write("  N/A\n")

                    # 输出基站电量
                    f.write("--- 基站电量 ---\n")
                    if loc_batt_history and self.num_locs > 0:
                        loc_batt_str = ", ".join([f"基站{i}: {loc_batt_history[i][step_idx]:.2f}"
                                                for i in range(self.num_locs) if step_idx < len(loc_batt_history[i])])
                        f.write(f"  [{loc_batt_str}]\n")
                    else:
                        f.write("  N/A\n")

                    # 输出断电基站数量
                    f.write("--- 断电基站 ---\n")
                    if down_loc_history and step_idx < len(down_loc_history):
                        down_count = down_loc_history[step_idx]
                        f.write(f"  数量: {int(down_count)}\n")  # 确保是整数
                    else:
                        f.write("  N/A\n")

                    f.write("\n")  # 每个时间步后加空行

        except Exception as e:
            print(f"写入 battery_history_readable.txt 时出错: {e}")

        # --- 4. 调用其他历史输出函数 ---
        if hasattr(self, 'mask_names') and hasattr(self, 'mask_histories'):
            self.output_mask_history()
        if hasattr(self, 'action_names') and hasattr(self, 'action_histories'):
            self.output_action_history()
        if hasattr(self, 'mask_calc_log'):
            self.output_mask_calc_history()
        if hasattr(self, 'queue_related_names') and hasattr(self, 'queue_histories'):
            self.output_queue_history()
            # 添加队列计算过程输出
        if hasattr(self, 'queue_calc_log'):
            self.output_queue_calc_history()
        if hasattr(self, 'queue_cost_log') and self.queue_cost_log:
            batch = 0  # 只处理第一个批次
            output_dir = f"{self.fname}-sample{batch}"
            self.plot_queue_conflict_cost(output_dir)

        # --- 5. 绘制电池历史图 ---
        if SAVE_PICTURE:
            fig = plt.figure(figsize=(10, 30))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            
            # 确保在绘图前检查历史记录是否存在且不为空
            if self.time_history and self.time_history[batch]:
                # 绘制车辆电量历史
                for vehicle_id in range(self.num_vehicles):
                    if vehicle_id < len(self.vehicle_batt_history[batch]) and self.vehicle_batt_history[batch][vehicle_id]:
                        min_len = min(len(self.time_history[batch]), len(self.vehicle_batt_history[batch][vehicle_id]))
                        ax1.plot(self.time_history[batch][:min_len], list(self.vehicle_batt_history[batch][vehicle_id][:min_len]))
                
                # # 绘制基站电量历史
                # for loc_id in range(self.num_locs):
                #     if loc_id < len(self.loc_batt_history[batch]) and self.loc_batt_history[batch][loc_id]:
                #         min_len = min(len(self.time_history[batch]), len(self.loc_batt_history[batch][loc_id]))
                #         ax2.plot(self.time_history[batch][:min_len], list(self.loc_batt_history[batch][loc_id][:min_len]))
                
                # # 绘制断电基站数量历史
                # if self.down_history and self.down_history[batch]:
                #     min_len = min(len(self.time_history[batch]), len(self.down_history[batch]))
                #     ax3.plot(self.time_history[batch][:min_len], self.down_history[batch][:min_len])

            # 设置标签等
            ax1.set_xlabel("Time (h)")
            ax1.set_ylabel("EVs' battery (KW)")
            ax2.set_xlabel("Time (h)")
            ax2.set_ylabel("Base stations' battery (KW)")
            ax3.set_xlabel("Time (h)")
            ax3.set_ylabel("Number of downed base stations")
            plt.tight_layout()  # 调整布局防止重叠
            plt.savefig(f"{output_dir}/batt_history.png", dpi=DPI)
            plt.close()

    def output_queue_history(self):
        """输出队列相关的历史数据"""
        batch = 0  # 只处理第一个批次
        if not hasattr(self, 'queue_histories'):
            print("警告：queue_histories未初始化，无法输出队列历史。")
            return

        output_dir = f"{self.fname}-sample{batch}"
        os.makedirs(output_dir, exist_ok=True)

        # 准备数据
        queue_history_data = {
            "time": self.time_history[batch],
            "queue_related": {}
        }
        for name in self.queue_related_names:
            if batch < len(self.queue_histories[name]) and self.queue_histories[name][batch]:
                # 将张量转换为列表，便于序列化和可读性
                data_list = []
                for item in self.queue_histories[name][batch]:
                    if isinstance(item, torch.Tensor):
                        data_list.append(item.tolist())
                    else:
                        data_list.append(item)  # 如果已经是列表或其他类型
                queue_history_data["queue_related"][name] = data_list

        # 保存PKL文件
        pkl_path = f"{output_dir}/queue_history_data.pkl"
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(queue_history_data, f)
        except Exception as e:
            print(f"保存queue_history_data.pkl时出错: {e}")
            print("数据内容:")
            # 打印部分数据用于调试
            for key, value in queue_history_data.items():
                if key == 'queue_related':
                    print(f"  {key}:")
                    for name, hist in value.items():
                        print(f"    {name}: (长度 {len(hist)}) {str(hist)[:200]}...")
                else:
                    print(f"  {key}: (长度 {len(value)}) {str(value)[:200]}...")

        # 生成可读的TXT文件
        txt_path = f"{output_dir}/queue_history_data_readable.txt"
        with open(txt_path, "w") as f:
            f.write("===== 队列相关历史 =====\n\n")
            time_points = queue_history_data.get("time", [])
            queue_data = queue_history_data.get("queue_related", {})

            for t_idx, t in enumerate(time_points):
                f.write(f"===== 状态 {t_idx} (时间: {t:.3f} 小时) =====\n")

                # charge_queue
                if "charge_queue" in queue_data and t_idx < len(queue_data["charge_queue"]):
                    cq = queue_data["charge_queue"][t_idx]
                    f.write("--- 充电队列 (充电站 x 车辆: 队列号，0表示不在队列) ---\n")
                    for depot_idx, depot_queue in enumerate(cq):
                        f.write(f"  充电站 {depot_idx}: {depot_queue}\n")
                else:
                    f.write("--- 充电队列: 无数据 ---\n")

                # queued_vehicles
                if "queued_vehicles" in queue_data and t_idx < len(queue_data["queued_vehicles"]):
                    qv = queue_data["queued_vehicles"][t_idx]
                    qv_str = ", ".join([f"车辆{i}={v}" for i, v in enumerate(qv)])
                    f.write(f"--- 排队车辆 (是否在排队): [{qv_str}]\n")
                else:
                    f.write("--- 排队车辆: 无数据 ---\n")

                # vehicle_unavail_time
                if "vehicle_unavail_time" in queue_data and t_idx < len(queue_data["vehicle_unavail_time"]):
                    vut = queue_data["vehicle_unavail_time"][t_idx]
                    vut_str = ", ".join([f"车辆{i}={v:.3f}" for i, v in enumerate(vut)])
                    f.write(f"--- 车辆不可用时间: [{vut_str}]\n")
                else:
                    f.write("--- 车辆不可用时间: 无数据 ---\n")

                # charging_vehicles
                if "charging_vehicles" in queue_data and t_idx < len(queue_data["charging_vehicles"]):
                    cv = queue_data["charging_vehicles"][t_idx]
                    cv_str = ", ".join([f"车辆{i}={v}" for i, v in enumerate(cv)])
                    f.write(f"--- 充电中车辆 (是否正在充电): [{cv_str}]\n")
                else:
                    f.write("--- 充电中车辆: 无数据 ---\n")

                f.write("\n")

    def plot_queue_conflict_cost(self, output_dir):
        """Plot charging station queue conflict cost charts"""
        # Check if log data exists
        if not hasattr(self, 'queue_cost_log') or not self.queue_cost_log:
            print("No queue cost log data available for plotting.")
            return
        
        times = []
        # Create a list to store cost history for each charging station 
        station_costs = [[] for _ in range(self.num_depots)]
        total_costs = []  # Store system total cost history
        
        # New: Cumulative count data
        cumulative_conflicts = 0  # Initialize cumulative conflict count
        cumulative_conflicts_history = []  # Store cumulative conflict count history
        
        # Record queue state at previous time point for each charging station for comparison
        last_queue_state = None
        
        # Iterate through log data
        for t, queue_state in self.queue_cost_log:
            times.append(t)
            
            current_total_cost = 0
            # Calculate cost for each charging station
            for depot_idx in range(self.num_depots):
                # Calculate number of vehicles with queue number > 1 at this station
                cost = (queue_state[depot_idx] > 1).sum().item()
                station_costs[depot_idx].append(cost)
                current_total_cost += cost
                
                # New: Calculate cumulative conflict count
                if last_queue_state is not None:
                    # Compare current and previous queue numbers
                    # Count new numbers >= 2
                    new_conflicts = ((queue_state[depot_idx] >= 2) & 
                                    (queue_state[depot_idx] > last_queue_state[depot_idx])).sum().item()
                    cumulative_conflicts += new_conflicts
            
            total_costs.append(current_total_cost)
            cumulative_conflicts_history.append(cumulative_conflicts)
            
            # Update previous state
            last_queue_state = queue_state.clone()
        
        # --- Start plotting ---
        plt.figure(figsize=(14, 15))  # Increase figure height to accommodate three subplots
        
        # Figure 1: Conflict costs for each charging station
        ax1 = plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
        cmap = get_cmap(self.num_depots)  # Get color map
        for depot_idx in range(self.num_depots):
            # Ensure data lengths match
            if len(station_costs[depot_idx]) == len(times):
                # Plot cost curve for each charging station
                plt.plot(times, station_costs[depot_idx],
                        label=f'Station {depot_idx} (Node {self.num_locs + depot_idx})',
                        color=cmap(depot_idx), alpha=0.8)
            else:
                print(f"Warning: Data length mismatch for station {depot_idx}. Time points: {len(times)}, Cost points: {len(station_costs[depot_idx])}")
        
        plt.title('Queue Conflict Cost per Charging Station (Vehicles with Queue Number > 1)')
        plt.ylabel('Number of Queued Vehicles')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend on the right
        plt.grid(True)
        # Hide x-axis labels as they are shared with plot below
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        # Figure 2: System total conflict cost
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)  # 3 rows, 1 column, second subplot, sharing x-axis
        # Ensure data lengths match
        if len(total_costs) == len(times):
            plt.plot(times, total_costs, label='Total System Cost', color='black', linewidth=2)
        else:
            print(f"Warning: Total cost data length mismatch. Time points: {len(times)}, Cost points: {len(total_costs)}")
        
        plt.title('Total System Queue Conflict Cost')
        plt.ylabel('Total Queued Vehicles')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend on the right
        plt.grid(True)
        # Hide x-axis labels as they are shared with plot below
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        # Figure 3: Cumulative conflict count
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)  # 3 rows, 1 column, third subplot, sharing x-axis
        if len(cumulative_conflicts_history) == len(times):
            plt.plot(times, cumulative_conflicts_history, label='Cumulative Conflicts', 
                    color='red', linewidth=2)
            
            # Add annotations every 10 units
            interval = max(1, len(times) // 10)  # Ensure at least some annotation points
            for i in range(0, len(times), interval):
                if i < len(cumulative_conflicts_history):
                    plt.annotate(f'{cumulative_conflicts_history[i]}', 
                                (times[i], cumulative_conflicts_history[i]),
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center')
        else:
            print(f"Warning: Cumulative conflicts data length mismatch. Time points: {len(times)}, Cost points: {len(cumulative_conflicts_history)}")
        
        plt.title('System Cumulative Conflict Count (Counts Each New Queue Number ≥2)')
        plt.xlabel('Time (h)')
        plt.ylabel('Cumulative Conflict Count')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend on the right
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
        # Check if output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/queue_conflict_cost.png", dpi=DPI)  # Save figure
        plt.close()  # Close figure to free memory
        
        print(f"Queue conflict cost plots saved to: {output_dir}/queue_conflict_cost.png")
    def output_queue_calc_history(self):
        """将详细的队列计算过程输出到文件"""
        batch = 0  # 处理第一个批次
        
        # 检查是否有日志数据
        if not hasattr(self, 'queue_calc_log') or not self.queue_calc_log[batch]:
            print("警告: queue_calc_log 未初始化或为空，无法输出队列计算过程。")
            return
        
        # 创建输出目录
        output_dir = f"{self.fname}-sample{batch}"
        os.makedirs(output_dir, exist_ok=True)
        txt_path = f"{output_dir}/queue_calculation_process.txt"
        
        # 写入文件
        with open(txt_path, "w") as f:
            f.write("===== 充电队列计算过程详细记录 =====\n\n")
            
            for log_entry in self.queue_calc_log[batch]:
                # 写入步骤基本信息
                f.write(f"步骤 {log_entry['step']} (时间: {log_entry['time']:.3f}) - {'阶段对齐' if log_entry['align_phase'] else '状态更新'}\n")
                f.write(f"选择车辆: {log_entry['selected_vehicle']}\n")
                f.write(f"当前位置: {log_entry['vehicle_position']}\n")
                f.write(f"目标节点: {log_entry['next_node']}\n")
                f.write(f"经过时间: {log_entry['elapsed_time']:.3f}\n\n")
                
                # 写入更新前的状态
                f.write("更新前状态:\n")
                f.write(f"  车辆阶段: {log_entry['queue_state_before']['vehicle_phase']}\n")
                f.write(f"  车辆位置: {log_entry['queue_state_before']['vehicle_position_id']}\n")
                f.write(f"  车辆不可用时间: {[f'{t:.3f}' for t in log_entry['queue_state_before']['vehicle_unavail_time']]}\n")
                
                # 格式化充电队列以便更易读
                f.write(f"  充电队列 (充电站 x 车辆):\n")
                for d_idx, depot_queue in enumerate(log_entry['queue_state_before']['charge_queue']):
                    f.write(f"    充电站 {d_idx}: {depot_queue}\n")
                    
                f.write("\n")
                
                # 写入计算过程
                f.write("充电队列计算过程:\n")
                for calc in log_entry.get("calculations", []):
                    f.write(f"  充电站 {calc['depot_id']} (索引 {calc['depot_idx']}):\n")
                    f.write(f"    在站车辆: {calc.get('vehicles_at_depot', [])}\n")
                    
                    if "queue_numbers" in calc:
                        f.write(f"    队列号: {calc['queue_numbers']}\n")
                    
                    if "valid_vehicles" in calc:
                        f.write(f"    有效车辆: {calc['valid_vehicles']}\n")
                        f.write(f"    有效队列号: {calc['valid_queue_nums']}\n")
                    
                    if "min_vehicle" in calc:
                        f.write(f"    最小队列车辆: {calc['min_vehicle']} (队列号: {calc['min_queue']})\n")
                        f.write(f"    充电车辆: {calc['charging_vehicle']}\n")
                        f.write(f"    排队车辆: {calc['queued_vehicles']}\n")
                    
                    f.write("\n")
                
                # 写入最终结果
                f.write("计算结果:\n")
                f.write(f"  充电车辆掩码: {log_entry.get('charging_vehicles_mask', [])}\n")
                f.write(f"  排队车辆掩码: {log_entry.get('queued_vehicles_mask', [])}\n")
                
                # 写入时间更新信息
                if "time_update" in log_entry:
                    f.write(f"  时间更新:\n")
                    f.write(f"    更新车辆掩码: {log_entry['time_update']['update_vehicles_mask']}\n")
                    f.write(f"    经过时间: {log_entry['time_update']['elapsed_time']:.3f}\n")
                f.write("\n")
                
                # 写入队列更新信息 (入队/出队)
                if "queue_update_align" in log_entry:
                    update_info = log_entry["queue_update_align"]
                    f.write(f"  队列操作 ({update_info['operation']}):\n")
                    for detail in update_info.get("details", []):
                        f.write(f"    充电站 {detail['depot_id']}:\n")
                        f.write(f"      到达车辆: {detail['arriving_vehicle']}\n")
                        f.write(f"      是否等待: {detail['waiting']}\n")
                        f.write(f"      当前最大队列号: {detail['current_max_queue']}\n")
                        if "assigned_queue" in detail:
                            f.write(f"      分配队列号: {detail['assigned_queue']}\n")
                    f.write("\n")
                
                if "queue_update_not_align" in log_entry:
                    update_info = log_entry["queue_update_not_align"]
                    f.write(f"  队列操作 ({update_info['operation']}):\n")
                    for detail in update_info.get("details", []):
                        f.write(f"    充电站 {detail['depot_id']}:\n")
                        f.write(f"      离开车辆: {detail['leaving_vehicle']}\n")
                        f.write(f"      之前队列号: {detail['previous_queue']}\n")
                    f.write("\n")
                
                # 写入更新后的状态
                f.write("更新后状态:\n")
                f.write(f"  车辆阶段: {log_entry['queue_state_after']['vehicle_phase']}\n")
                f.write(f"  车辆位置: {log_entry['queue_state_after']['vehicle_position_id']}\n")
                f.write(f"  车辆不可用时间: {[f'{t:.3f}' for t in log_entry['queue_state_after']['vehicle_unavail_time']]}\n")
                
                # 格式化充电队列以便更易读
                f.write(f"  充电队列 (充电站 x 车辆):\n")
                for d_idx, depot_queue in enumerate(log_entry['queue_state_after']['charge_queue']):
                    f.write(f"    充电站 {d_idx}: {depot_queue}\n")
                    
                f.write("\n")
                f.write("-" * 80 + "\n\n")

    def get_current_mask_value(self, mask_name: str, batch_index: int):
        """获取当前指定掩码的值
        
        Parameters
        ----------
        mask_name : str
            掩码名称
        batch_index : int
            批次索引
            
        Returns
        -------
        torch.Tensor
            掩码值的CPU张量副本
        """
        
        # if mask_name == "loc_is_down":
        #     return (self.loc_curr_battery[batch_index] <= self.loc_min_battery).clone().detach().cpu()
        # # 判断基站电量是否低于等于最小电量阈值
        # if mask_name == "loc_is_full":
        #     return (self.loc_curr_battery[batch_index] >= self.loc_cap[batch_index]).clone().detach().cpu()
        # # 判断基站电量是否大于等于其容量上限
        # elif mask_name == "loc_is_normal":
        #     return ((self.loc_curr_battery[batch_index] > self.loc_min_battery) & 
        #         (self.loc_curr_battery[batch_index] < self.loc_cap[batch_index])).clone().detach().cpu()
       # 判断基站电量是否在最小电量和容量上限之间
        if mask_name == "vehicle_position_id":
            return self.vehicle_position_id[batch_index].clone().detach().cpu()
        elif mask_name == "small_depots":
            return self.small_depots[batch_index].clone().detach().cpu()
        elif mask_name == "charge_queue":
            return self.charge_queue[batch_index].clone().detach().cpu()
        elif mask_name == "vehicle_phase":
            return self.vehicle_phase[batch_index].clone().detach().cpu()
        elif mask_name == "wait_vehicle":
            return self.wait_vehicle[batch_index].clone().detach().cpu()
        elif mask_name == "at_depot":
            return self.is_depot(self.vehicle_position_id[batch_index]).clone().detach().cpu()
        elif mask_name == "at_loc":
            return (~self.is_depot(self.vehicle_position_id[batch_index])).clone().detach().cpu()
        elif mask_name == "next_vehicle_mask":
            return self.get_vehicle_mask(self.next_vehicle_id[batch_index]).clone().detach().cpu()
        elif mask_name == "skip":
            return self.skip[batch_index].clone().detach().cpu()
        elif mask_name == "end":
            return self.end[batch_index].clone().detach().cpu()
            # --- 添加的 CASES ---
        elif mask_name == "queued_vehicles":
            # 基于特定批次的当前充电队列状态重新计算
            # 如果车辆在 *任何* 充电站的到达顺序号 > 1，则认为它在排队
            # 对车辆的队列号在所有充电站维度上求和；如果 > 1，则表示它在某处排队（或发生错误）
            # 注意: self.charge_queue[batch_index] 的形状是 [num_depots, num_vehicles]
            queued = (self.charge_queue[batch_index].sum(0) > 1) # 沿 depot 维度 (dim 0) 求和 -> 形状 [num_vehicles]
            return queued.clone().detach().cpu()

        elif mask_name == "charging_vehicles":
            # 基于特定批次的当前阶段、位置和队列状态重新计算
            charge_phase = (self.vehicle_phase[batch_index] == self.phase_id["charge"]) # [num_vehicles]
            at_depot = self.is_depot(self.vehicle_position_id[batch_index]) # [num_vehicles]
            # 为此快照重新计算排队状态
            queued = (self.charge_queue[batch_index].sum(0) > 1) # [num_vehicles]
            # 车辆只有在处于充电阶段、在充电站、并且没有排队时，才算正在充电
            charging = (charge_phase & at_depot & ~queued)
            return charging.clone().detach().cpu()
        # --- 添加的 CASES 结束 ---
        else:
            return None  # 未知掩码名称

    def get_current_action_value(self, action_name: str, batch_index: int):
        """获取当前动作和系统状态信息

        Parameters
        ----------
        action_name : str
            动作名称
        batch_index : int
            批次索引

        Returns
        -------
        Any
            根据动作名称返回对应的状态值
        """
        # 从上一步 update 和 update_state 中获取的动作信息
        if action_name == "curr_vehicle_id" and self.last_action_info["curr_vehicle_id"] is not None:
            return self.last_action_info["curr_vehicle_id"][batch_index].item()
        elif action_name == "next_node_id" and self.last_action_info["next_node_id"] is not None:
            return self.last_action_info["next_node_id"][batch_index].item()
        elif action_name == "do_wait" and self.last_action_info["do_wait"] is not None:
            return self.last_action_info["do_wait"][batch_index].item()
        elif action_name == "travel_time" and self.last_action_info["travel_time"] is not None:
            return self.last_action_info["travel_time"][batch_index].item()
        elif action_name == "charge_time" and self.last_action_info["charge_time"] is not None:
            return self.last_action_info["charge_time"][batch_index].item()
        elif action_name == "elapsed_time" and self.last_action_info["elapsed_time"] is not None:
            return self.last_action_info["elapsed_time"][batch_index].item()
        elif action_name == "accumulated_conflict_cost":
            return self.accumulated_conflict_cost[batch_index].item()
        
        # 系统状态指标
        elif action_name == "tour_length":
            return self.tour_length[batch_index].item()
        # elif action_name == "penalty_empty_locs" or action_name == "accumulated_conflict_cost":
        #     return self.accumulated_conflict_cost[batch_index].item()
        
        # # 派生状态信息
        # elif action_name == "down_locs_count":
        #     return ((self.loc_curr_battery[batch_index] - self.loc_min_battery) <= 0.0).sum().item()
        elif action_name == "queued_vehicles_count":
            return (self.charge_queue[batch_index].sum(0) > 1).sum().item()
        elif action_name == "charging_vehicles_count":
            charge_phase = (self.vehicle_phase[batch_index] == self.phase_id["charge"])
            at_depot = self.is_depot(self.vehicle_position_id[batch_index])
            queued = (self.charge_queue[batch_index].sum(0) > 1)
            return (charge_phase & at_depot & ~queued).sum().item()
        # elif action_name == "supplying_vehicles_count":
        #     charge_phase = (self.vehicle_phase[batch_index] == self.phase_id["charge"])
        #     at_loc = ~self.is_depot(self.vehicle_position_id[batch_index])
        #     return (charge_phase & at_loc).sum().item()
        
        return None
# 从 self.last_action_info 字典中获取最近一次动作的信息：
# # 动作相关信息
# "curr_vehicle_id"  # 当前执行动作的车辆ID
# "next_node_id"     # 目标节点ID
# "do_wait"          # 是否执行等待动作
# "travel_time"      # 预计行驶时间
# "charge_time"      # 预计充电/供电时间
# "elapsed_time"     # 实际耗时

# # 系统状态指标
# "tour_length"         # 总行驶距离
# "penalty_empty_locs"  # 基站断电惩罚值累计

# # 断电基站数量
# "down_locs_count" = (电池电量 <= 最小电量).sum()

# # 排队车辆数量
# "queued_vehicles_count" = (充电队列长度 > 1).sum()

# # 正在充电的车辆数量
# "charging_vehicles_count" = (
#     处于充电阶段 & 
#     在充电站 & 
#     ~在排队
# ).sum()

# # 正在供电的车辆数量
# "supplying_vehicles_count" = (
#     处于充电阶段 & 
#     在基站
# ).sum()

# # 获取当前执行动作的车辆ID
# vehicle_id = state.get_current_action_value("curr_vehicle_id", batch=0)

# # 获取系统中断电基站的数量
# down_count = state.get_current_action_value("down_locs_count", batch=0)

# # 获取正在充电的车辆数量
# charging_count = state.get_current_action_value("charging_vehicles_count", batch=0)


def torch2numpy(tensor: torch.Tensor):
    return tensor.cpu().detach().numpy().copy()

def add_base(x, y, ratio, ax, node_id=None):
    width = 0.01 * 1.5
    height = 0.015 * 1.5
    battery_color = "limegreen"  # 始终为绿色

    frame = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height, fill=False, ec="black")
    battery = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height, facecolor=battery_color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame) 
    
    if node_id is not None:
        text_x_offset = width
        text_y_offset = 0
        ax.text(x + text_x_offset, y + text_y_offset, 
                f"{node_id}",
                fontsize=8,
                ha='left',
                va='center',
                color='black',
                zorder=5)
        
    # if ratio > 0.5:
    #     battery_color = "limegreen"
    # elif ratio > 0.3:
    #     battery_color = "gold"
    # else:
    #     battery_color = "red"
    # ec = "red" if ratio < 1e-9 else "black"

    # frame = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height, fill=False, ec=ec)
    # battery = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height_mod, facecolor=battery_color, linewidth=.5, ec="black")
    # ax.add_patch(battery)
    # ax.add_patch(frame) 
    
    # 添加节点编号
    if node_id is not None:
        text_x_offset = width  # 水平偏移量
        text_y_offset = 0      # 垂直偏移量
        ax.text(x + text_x_offset, y + text_y_offset, 
                f"{node_id}",
                fontsize=8,
                ha='left',
                va='center',
                color='black',
                zorder=5)

def add_vehicle(x, y, ratio, batt, color, ax, vehicle_id=None):
    offst = 0.03
    BATT_OFFSET = 0.025
    # vehicle_battery
    width = 0.015 * 1.2
    height = 0.01 * 1.2
    width_mod = ratio * width
    ec = "red" if ratio < 1e-9 else "black"
    frame = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width, height=height, fill=False, ec=ec)
    battery = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width_mod, height=height, facecolor=color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame)

    # add remaining battery
    ax.text(x-0.02, y+0.07, f"{batt:.1f}", fontsize=10)
    
    # 添加车辆编号
    if vehicle_id is not None:
        text_x_offset = 0.03  # 水平偏移量
        text_y_offset = offst # 垂直偏移量
        ax.text(x + text_x_offset, y + text_y_offset, 
                f"V{vehicle_id}",
                fontsize=8,
                ha='left',
                va='center',
                color=color,
                fontweight='bold',
                zorder=5)
    
    # vehicle
    original_img = plt.imread("images/ev_image.png")
    vehicle_img = np.where(original_img == (1., 1., 1., 1.), (color[0], color[1], color[2], color[3]), original_img)
    vehicle_img = OffsetImage(vehicle_img, zoom=0.25)
    ab = AnnotationBbox(vehicle_img, (x, y+offst), xycoords='data', frameon=False)
    ax.add_artist(ab)

def get_cmap(num_colors: int):
    if num_colors <= 10:
        cm_name = "tab10"
    elif num_colors <= 20:
        cm_name = "tab20"
    else:
        assert False
    return cm.get_cmap(cm_name)

def output_animation(out_fname, seq_fname, type="gif"):
    if type == "gif":
        cmd = f"ffmpeg -r {FPS} -i {seq_fname} -r {FPS} {out_fname}"
    else:
        cmd = f"ffmpeg -r {FPS} -i {seq_fname} -vcodec libx264 -pix_fmt yuv420p -r {FPS} {out_fname}"
    subprocess.call(cmd, shell=True)

def interpolate_line(start, end, ratio):
    return ratio * (end - start) + start