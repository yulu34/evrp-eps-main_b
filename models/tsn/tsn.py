LARGE_VALUE = int(1e+6)      # 100万，用于数值计算的大数
BIT_LARGE_VALUE = int(1e+4)  # 1万，用于位运算的大数

import os               # 操作系统接口
import matplotlib.pyplot as plt  # 绘图库
import collections     # 提供额外的数据结构
import math           # 数学运算
import numpy as np    # 数值计算库
import torch         # 深度学习框架
from scipy.spatial import distance  # 空间距离计算
# 这些用于：

# 矩阵运算
# 距离计算
# 张量操作
from ortools.sat.python import cp_model  # Google的约束规划求解器
from typing import List, Any, Tuple, Union

INFINITY = int(1e+14)

class TimeSpaceNetwork():
    def __init__(self, 
                 num_nodes: int,
                 T: int,
                 dt: float,
                 veh_speed: int,
                 distance_matrix: int,
                 max_traversal_step: int = 1) -> None:
        self.num_nodes = num_nodes
        self.T = T
        self.veh_speed = veh_speed
        self.distance_matrix = distance_matrix
        # 0 -> invalid, 1 -> valid
        self.valid_nodes = np.ones((num_nodes, T))  # 所有节点初始都是有效的
        # 二维 全1
        self.valid_arcs = np.zeros((num_nodes, num_nodes, T, T))  # 所有弧初始都是无效的
        # 四维 全0
        # remove arcs that cannot be reached within the time (traversal_steps * dt)
        for t1 in range(T):
            #第一次清理：基于距离的可达性
            for t2 in range(t1+1, T):
                self.valid_arcs[:, :, t1, t2] = (distance_matrix <= veh_speed * (t2 - t1) * dt)
#                 检查在给定时间内是否能到达
# 如果距离大于车速×时间，则该弧不可行
# 例如：如果两点距离100km，车速40km/h，时间差1h，则不可达
        # remove arcs that traversal more than (max_traversal_step+1) steps:
        # EVs alway reach a node at the earliest
        for t1 in range(T):
            # 第二次清理：最大步长限制
            for t2 in range(t1+1, T-max_traversal_step):
                for t3 in range(t2+max_traversal_step, T):
                    self.valid_arcs[:, :, t1, t3] = np.maximum(self.valid_arcs[:, :, t1, t3] - self.valid_arcs[:, :, t1, t2], 0)
#                     移除超过最大步长的弧
# 强制车辆选择最短路径
# 例如：如果A->C可以经过B，就不允许直接从A到C

        # remove duplicated stay arcs:
        # stay arcs that traverse more than 2 steps is not needed as it can be represented by two stay arcs that traverse only 1 step instead
        
        for t1 in range(T):
             # 第三次清理：重复停留弧
            for t2 in range(t1+2, T):
                np.fill_diagonal(self.valid_arcs[:, :, t1, t2], 0)
#                 移除多步停留弧
# 如果需要停留多步，可以用多个单步停留组合
# 例如：停留3步可以用3个1步停留表示

# 优化计算效率：减少不必要的弧，降低后续求解复杂度
# 保证可行性：确保所有路径都是物理上可行的
# 简化表示：用最简单的方式表达所有可能的路径
        # store valid nodes & arcs
        self.nodes = [(node_id, t) for node_id, t in zip(*np.where(self.valid_nodes))]

        #         # 假设有一个3x4的valid_nodes矩阵：
        # valid_nodes = [
        #     [1, 1, 0, 1],  # 节点0
        #     [1, 0, 1, 1],  # 节点1
        #     [0, 1, 1, 0]   # 节点2
        # ]
        
        # # np.where(valid_nodes)会返回：
        # # ([0,0,0, 1,1,1, 2,2], [0,1,3, 0,2,3, 1,2])
        
        # # 最终self.nodes为：
        # # [(0,0), (0,1), (0,3), (1,0), (1,2), (1,3), (2,1), (2,2)]

        self.arcs = [(from_node_id, to_node_id, from_time, to_time) for from_node_id, to_node_id, from_time, to_time in zip(*np.where(self.valid_arcs))]


# np.where的作用:
# 返回所有非零(有效弧)元素的坐标
# 返回4个数组，分别对应四个维度的索引
# *解包操作符将np.where的结果展开
# zip将四个索引数组打包成元组
# # 假设valid_arcs是2×2×2×2的4维数组
# valid_arcs = [
#     [[[0,1],
#       [0,0]],
#      [[0,0],
#       [1,0]]],
#     [[[0,0],
#       [0,1]],
#      [[0,0],
#       [0,0]]]
# ]

# # np.where(valid_arcs)会返回:
# # ([0,0,1], [1,0,0], [1,1,1], [0,1,1])

# # zip后得到:
# # [(0,1,1,0), (0,0,1,1), (1,0,1,1)]

# # 最终self.arcs为:
# # [(0,1,1,0), (0,0,1,1), (1,0,1,1)]
    def inflow_arcs(self, to_node_id: int, to_t: int) -> List[Tuple[int, int, int, int]]:
        # 1. 找出所有可能的入流节点和时间
        from_nodes = np.where(self.valid_arcs[:, to_node_id, :, to_t])
        
        # 2. 构建入流弧列表
        inflows = [(from_node_id, to_node_id, from_t, to_t) 
                  for from_node_id, from_t in zip(*from_nodes)]
        
        return inflows

# 例如: [(1,3,2,4), (2,3,3,4)] 表示:
# 从节点1在时间2出发到节点3在时间4到达
# 从节点2在时间3出发到节点3在时间4到达
    def outflow_arcs(self, from_node_id: int, from_t: int):
        # 找出从特定节点和时间点出发的所有可能目标节点
        to_nodes = np.where(self.valid_arcs[from_node_id, :, from_t, :])
        # 构建出流弧列表: (起点,终点,出发时间,到达时间)
        outflows = [(from_node_id, to_node_id, from_t, to_t) for to_node_id, to_t in zip(*to_nodes)]
        return outflows
# 功能: 获取从指定节点在指定时间可以出发的所有弧
# 输入: 起始节点ID和出发时间
# 输出: 可行的出流弧列表

    def stay_arcs(self, to_node_id: int, to_t: int) -> Union[Tuple[int, int, int, int], None]:
        if to_t > 0:
            return (to_node_id, to_node_id, to_t-1, to_t)
        else:
            return None
# 功能: 获取节点的停留弧(原地不动)
# 输入: 节点ID和时间
# 输出: 停留弧(同一节点,相邻时间)或None

    def arriving_arcs(self, arriving_time: int):
        # 找出在特定时间到达的所有弧
        from_nodes = np.where(self.valid_arcs[:, :, :, arriving_time])
        # 构建到达弧列表
        arrivings = [(from_id, to_id, from_time, arriving_time) for from_id, to_id, from_time in zip(*from_nodes)]
        return arrivings
# 功能: 获取在指定时间到达的所有弧
# 输入: 到达时间
# 输出: 在该时间到达的弧列表


    def time_slice(self, t: int) -> List[Tuple[int, int]]:
        return [(id[0], t) for id in zip(*np.where(self.valid_nodes[:, t]))]

# 功能: 获取特定时间点的有效节点
# 输入: 时间t
# 输出: 该时间点的所有有效节点列表

    def arc_distance(self, arc: Tuple[int, int, int, int]) -> int:
        return self.distance_matrix[arc[0]][arc[1]]
# 功能: 计算弧的距离
# 输入: 弧(起点,终点,出发时间,到达时间)
# 输出: 两点间的距离
    # def disp(self):
    #     print(self.nodes)
    #     print(self.distance_matrix)
    #     #print(self.arcs_array)
    #     block = np.blo ck([[self.valid_arcs[t1, t2] for t2 in self.time_index] for t1 in self.time_index])
    #     print(block)

    def visualize(self, outputdir: str) -> None:
        pass
        os.makedirs(outputdir, exist_ok=True)
        # DEBUG用 arc, nodeの可視化
        self.plot_node_arc(self.nodes, self.arcs, outputdir)
        #self.plot_node_arc(self.time_slice(1), [])
        #self.plot_node_arc([], self.inflow_arcs(2, 1))
        #self.plot_node_arc([], self.outflow_arcs(1, 1))
        #self.plot_node_arc([], self.stay_arcs(2,1))

    def plot_node_arc(self, 
                      nodes: List[Tuple[int, int]], 
                      arcs: List[Tuple[int, int, int, int]], 
                      outputdir: str = None) -> None:
        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot()
        for node in nodes:
            ax.scatter(node[1], node[0], s=128, color='red')
        for arc in arcs:
            ax.annotate("", 
                xy=[arc[3], arc[1]],        # 终点坐标(到达时间,终点ID)
                xytext=[arc[2], arc[0]],    # 起点坐标(出发时间,起点ID)
                arrowprops=dict(
                    shrink=0,               # 箭头不缩短
                    width=0.2,             # 箭头宽度
                    headwidth=8,           # 箭头头部宽度
                    headlength=6,          # 箭头头部长度
                    connectionstyle='arc3', # 弧线样式
                    facecolor='gray',      # 填充颜色
                    edgecolor='gray',      # 边框颜色
                    alpha=0.4              # 透明度
                ))
            
        # ax.set_xticks(self.T)
        # ax.set_yticks(range(0, len(self.distance_matrix)))
        # ax.set_xlim(self.time_index[0], self.time_index[-1])
        # ax.set_ylim(-len(self.distance_matrix)*0.05, (len(self.distance_matrix)-1)*1.05)
        if outputdir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(outputdir, 'TSN.png'))

def flatten_list(l: list) -> List[Any]:
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten_list(el)
        else:
            yield el

# nested_list = [1, [2, 3, [4, 5]], 6]
# flattened = list(flatten_list(nested_list))
# # 结果: [1, 2, 3, 4, 5, 6]

class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, limit) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables    # 变量列表
        self.__solution_count = 0       # 解决方案计数器
        self.__solution_limit = limit   # 最大解决方案数限制
# 这是一个用于约束规划(CP)求解器的回调类，限制求解方案数量：

    def on_solution_callback(self):
        self.__solution_count += 1
        # for v in self.__variables:
        #     print(f'{v}={self.Value(v)}')
        # print()
        # 达到限制时停止搜索
        if self.__solution_count >= self.__solution_limit:
            print(f'Stop search after {self.__solution_limit} solutions')
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count
   # 计数方法：

#    控制求解器找到指定数量的解后停止
# 避免无限搜索所有可能的解
# 提高求解效率
# 跟踪已找到的解决方案数量

class CP4TSN():
    def __init__(self,
                 time_horizon: int = 12,
                 dt: float = 0.5,
                 vehicle_speed: float = 41,
                 loss_coef: int = 1000,
                 loc_pre_time: float = 0.5,
                 loc_post_time: float = 0.5,
                 depot_pre_time: float = 0.17,
                 depot_post_time: float = 0.17,
                 ensure_minimum_charge: bool = True,
                 ensure_minimum_supply: bool = True,
                 random_seed: int = 1234,
                 num_search_workers: int = 4,
                 log_search_progress: bool = False,
                 limit_type: str = None,
                 time_limit: float = 60.0,
                 solution_limit: int = 10):
        """
        Parameters
        ----------

        """
        self.time_horizon = time_horizon  # 规划时间范围，默认12小时
        self.dt = dt                      # 时间步长，默认0.5小时 # 之后改这里可能不一定0.5
        self.T = int(time_horizon / dt) + 1  # 总时间步数 # 时间范围
        self.vehicle_speed = vehicle_speed  # 车辆速度，默认41km/h
        self.loss_coef = loss_coef        # 损失函数系数，默认1000
        # 基站操作时间
        self.loc_pre_time = loc_pre_time    # 基站准备时间，默认0.5h
        self.loc_post_time = loc_post_time  # 基站完成时间，默认0.5h
        
        # 充电站操作时间
        self.depot_pre_time = depot_pre_time    # 充电站准备时间，默认0.17h
        self.depot_post_time = depot_post_time  # 充电站完成时间，默认0.17h

        self.ensure_minimum_charge = ensure_minimum_charge  # 确保最小充电量
        self.ensure_minimum_supply = ensure_minimum_supply  # 确保最小供应量

        # 计算操作时间余量
        self.loc_surplus_pre_time = loc_pre_time - dt * (math.ceil(loc_pre_time / dt) - 1)
        self.loc_surplus_post_time = loc_post_time - dt * (math.ceil(loc_post_time / dt) - 1)
        self.depot_surplus_pre_time = depot_pre_time - dt * (math.ceil(depot_pre_time / dt) - 1)
        self.depot_surplus_post_time = depot_post_time - dt * (math.ceil(depot_post_time / dt) - 1)

        self.random_seed = random_seed            # 随机种子，确保结果可复现
        self.num_search_workers = num_search_workers  # 搜索工作线程数
        self.log_search_progress = log_search_progress  # 是否记录搜索进度
        self.limit_type = limit_type              # 限制类型
        self.time_limit = time_limit              # 时间限制（60秒）
        self.solution_limit = solution_limit       # 解决方案数量限制（10个）
        # NOTE: time_limit could make the results unrepreducible even if random_seed is set 
        # because the calculation time could differ in each run, resulting in different numbers of found solution

    def solve(self, input: dict, log_fname: str = None):
        """
        Paramters
        ---------

        Returns
        -------

        """
        # convert input feature
        self.set_input(input)

        # deffine Time Space Network
        # 创建时空网络
        self.tsn = TimeSpaceNetwork(self.num_nodes, self.T, self.dt, self.normalized_veh_speed, self.distance_matrix)

        # define a model
        print("defining a model...", end="")
        model = cp_model.CpModel()
        variables = self.add_variables(model)# 添加变量
        self.add_constraints(model, variables)# 添加约束
        self.add_objectives(model, variables, self.loss_coef) # 添加目标函数
        print("done")

        # validate the model
        validate_res = model.Validate()
        if validate_res != "":
            print(validate_res)

        # solve TSN with the CP-SAT solver
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = self.random_seed
        solver.parameters.num_search_workers = self.num_search_workers
        solver.log_search_progress = self.log_search_progress
        # 根据限制类型选择求解方式
        if self.limit_type == "time":
             # 时间限制模式
            solver.parameters.max_time_in_seconds = self.time_limit
            status = solver.Solve(model)
        elif self.limit_type == "solution_count":
                # 解决方案数量限制模式
            solver.parameters.num_search_workers = 1 # enumerating all solutions does not work in parallel
            solver.parameters.enumerate_all_solutions = True
            variable_list = []
                # 收集所有变量
            for variable in variables.values():
                if isinstance(variable, list):
                    variable_list += list(flatten_list(variable))
                elif isinstance(variable, dict):
                    variable_list += list(variable.values())
                else:
                    variable_list += [variable]
            solution_printer = VarArraySolutionPrinterWithLimit(variable_list, self.solution_limit)
            status = solver.Solve(model, solution_printer)
        else:
            status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            print("The optimal solution found!!!")
        elif status == cp_model.FEASIBLE:
            print("A feasible solution found!")
        else:
            print("No solution found :(")

        # if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        #     print([solver.Value(variables["num_down_loc"][t]) for t in range(self.T)])
        #     print([solver.Value(variables["travel_distance"][veh_id]) for veh_id in range(self.num_vehicles)])
        #     for veh_id in range(self.num_vehicles):
        #         print(f"EV{veh_id}")
        #         route = [arc for arc in self.tsn.arcs if solver.Value(variables["x"][veh_id, arc]) == 1]
        #         route.sort(key=lambda a: a[2])
        #         for arc in route:
        #             print(f"(time:{arc[2]}->{arc[3]}) station:{arc[0]}-> {arc[1]}")
        #     print(solver.Value(variables["loss1"]) / (LARGE_VALUE * BIT_LARGE_VALUE), solver.Value(variables["loss2"]) / (LARGE_VALUE * BIT_LARGE_VALUE))
        # print(f"Status: {solver.StatusName(status)}")
        # print(solver.SolutionInfo())
        # print(solver.ResponseStats())
        
        # 构建路径信息
        route = [[] for _ in range(self.num_vehicles)]
        for veh_id in range(self.num_vehicles):
            # 获取车辆使用的所有弧
            arcs = [arc for arc in self.tsn.arcs if solver.Value(variables["x"][veh_id, arc]) == 1]
            arcs.sort(key=lambda a: a[2])  # 按时间排序
            for arc in arcs:
                route[veh_id].append((arc[2], arc[3], arc[0], arc[1]))  # (起始时间,结束时间,起点,终点)
        
        # 返回结果字典
        return {
            "route": route,  # 每辆车的路径
            "total_route_length": sum(solver.Value(variables["travel_distance"][veh_id]) 
                                    for veh_id in range(self.num_vehicles)) / LARGE_VALUE,  # 总路程
            "num_down_locs": [solver.Value(variables["num_down_loc"][t]) 
                             for t in range(self.T)],  # 每个时间点的断电基站数
            "objective_value": solver.ObjectiveValue() / (LARGE_VALUE * BIT_LARGE_VALUE)  # 目标函数值
        }
    
    #     {
    #     "route": [
    #         [(0,1,2,3), (1,2,3,4)],  # 车辆0的路径
    #         [(0,2,1,4), (2,3,4,2)]   # 车辆1的路径
    #     ],
    #     "total_route_length": 150.5,  # 总路程
    #     "num_down_locs": [0,1,0,2,1], # 各时间点断电基站数
    #     "objective_value": 235.6      # 目标函数值
    # }
    def set_input(self, input: dict):
        # locations
        self.loc_cap = (input["loc_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_locs]
        self.loc_consump_rate = (input["loc_consump_rate"] * LARGE_VALUE).to(torch.long).tolist() # [num_locs]
        self.loc_init_batt = (input["loc_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_locs]
        self.loc_min_batt = 0

        # depots
        self.depot_discharge_rate = (input["depot_discharge_rate"] * LARGE_VALUE).to(torch.long).tolist() # [num_depots]
        
        # EVs
        self.veh_cap = (input["vehicle_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_vehicles]
        self.veh_init_batt = (input["vehicle_cap"] * LARGE_VALUE).to(torch.long).tolist() # [num_vehicles]
        self.veh_discharge_rate = (input["vehicle_discharge_rate"] * LARGE_VALUE).to(torch.long).tolist()
        self.veh_consump_rate = input["vehicle_consump_rate"].tolist()
        self.veh_init_position_id = input["vehicle_initial_position_id"].tolist() # [num_vehicles]
        self.veh_min_batt = 0

        # distance_matrix
        self.loc_coords = input["loc_coords"].detach().numpy().copy() # [num_locs, coord_dim]
        self.depot_coords = input["depot_coords"].detach().numpy().copy() # [num_depots, coord_dim]
        self.node_coords = np.concatenate((self.loc_coords, self.depot_coords), 0) # [num_nodes, coord_dim]
        self.distance_matrix = (distance.cdist(self.node_coords, self.node_coords) * LARGE_VALUE).astype(np.long)

        # parameters
        self.num_locs = len(self.loc_cap)
        self.num_depots = len(self.depot_discharge_rate)
        self.num_nodes = self.num_locs + self.num_depots
        self.num_vehicles = len(self.veh_cap)
        self.grid_scale = input["grid_scale"]
        self.normalized_veh_speed = int(self.vehicle_speed / self.grid_scale * LARGE_VALUE)

    def add_variables(self, model):
        """
        Parameters
        ----------

        Returns
        -------
        """
        variables = {}
        self.add_batt_variables(model, variables)
        self.add_route_variables(model, variables)
        self.add_objective_variables(model, variables)
        return variables

    def add_batt_variables(self, model, var):
        # for locations
        var["loc_batt"] = [[model.NewIntVar(0, self.loc_cap[i], f"loc{i}_t{t}_batt") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_slack"] = [[model.NewIntVar(0, self.loc_min_batt+1, f"loc{i}_t{t}_slack") for t in range(self.T)] for i in range(self.num_locs)]
        var["enable_slack"] = [[model.NewBoolVar(f"loc{i}_t{t}_enable_slack") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_is_down"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_down") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_is_full"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_full") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_is_normal"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_normal") for t in range(self.T)] for i in range(self.num_locs)]
        var["loc_charge_amount"] = [[model.NewIntVar(0, self.loc_cap[i], f"loc{i}_t{t}_charge_amount") for t in range(self.T)] for i in range(self.num_locs)]
        # for EVs
        var["veh_batt"]           = [[model.NewIntVar(0, self.veh_cap[k], f"veh{k}_t{t}_batt") for t in range(self.T)] for k in range(self.num_vehicles)]
        var["veh_charge_amount"]  = [[model.NewIntVar(0, self.veh_cap[k], f"veh{k}_t{t}_charge_amount") for t in range(self.T)] for k in range(self.num_vehicles)]
        var["veh_is_discharging"] = [[model.NewBoolVar(f"veh{k}_t{t}_is_charging") for t in range(self.T)] for k in range(self.num_vehicles)]

    def add_route_variables(self, model, var):
        var["x"] = {(k, arc): model.NewBoolVar(f"x_veh{k}_arc{arc}") 
                    for k in range(self.num_vehicles) 
                    for arc in self.tsn.arcs}
#         表示车辆k是否使用某条弧
# 二元变量(0/1)：1表示使用该弧，0表示不使用
        var["z"] = [[[model.NewBoolVar(f"z_veh{k}_node{n}_t{t}") 
                      for t in range(self.T)] 
                      for n in range(self.num_nodes)] 
                      for k in range(self.num_vehicles)]
#         表示车辆k在时间t是否在节点n
# 三维数组：[车辆][节点][时间]

        var["loc_is_down2"] = [[model.NewBoolVar(f"loc{i}_t{t}_is_down2") 
                                for t in range(self.T)] 
                                for i in range(self.num_locs)]
#         表示基站i在时间t是否断电
# 二维数组：[基站][时间]
        var["veh_prepare_at_loc"]  = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_prepare_at_loc") 
                                        for t in range(self.T)] 
                                        for i in range(self.num_locs)] 
                                        for k in range(self.num_vehicles)]
        var["veh_prepare_at_loc2"] = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_prepare_at_loc2") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)] 
        var["veh_cleanup_at_loc"]  = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_cleanup_at_loc") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
        var["veh_cleanup_at_loc2"] = [[[model.NewBoolVar(f"veh{k}_loc{i}_t{t}_cleanup_at_loc2") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
        var["veh_prepare_at_depot"] = [[[model.NewBoolVar(f"veh{k}_depot{j}_t{t}_prepare_at_depot") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]
        var["veh_cleanup_at_depot"] = [[[model.NewBoolVar(f"veh{k}_depot{j}_t{t}_cleanup_at_depot") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]
        if self.ensure_minimum_supply:
            var["loc_supply_notenough"] = [[[model.NewBoolVar(f"not_enough_veh{k}_loc{i}_t{t}") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
            var["loc_suuply_notenough_notcleanup"] = [[[model.NewBoolVar(f"notenough_notcleanup_veh{k}_loc{i}_t{t}") for t in range(self.T)] for i in range(self.num_locs)] for k in range(self.num_vehicles)]
        if self.ensure_minimum_charge:
            var["veh_charge_notenough"] = [[[model.NewBoolVar(f"not_enough_veh{k}_depot{j}_t{t}") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]
            var["veh_charge_notenough_notcleanup"] = [[[model.NewBoolVar(f"notenough_notcleanup_veh{k}_depot{j}_t{t}") for t in range(self.T)] for j in range(self.num_depots)] for k in range(self.num_vehicles)]
#跟踪供电和充电不足的情况
    def add_objective_variables(self, model, var):
        var["num_down_loc"] = [model.NewIntVar(0, self.num_locs, f"num_down_loc_t{t}") for t in range(self.T)]
        var["travel_distance"] = [model.NewIntVar(0, INFINITY, f"veh{k}_travel_distance") for k in range(self.num_vehicles)]
        var["loss1"] = model.NewIntVar(0, INFINITY, "loss1")
        var["loss2"] = model.NewIntVar(0, INFINITY, "loss2")

    def add_constraints(self, model, var):
        self.battery_init_lowerbound(model, var)
        self.ensure_route_continuity(model, var)
        self.forbit_multi_veh_at_same_node(model, var)
        self.define_batt_behavior(model, var)
        self.add_objective_constraints(model, var)

    def battery_init_lowerbound(self, model, var):
        # for EVs
        for veh_id in range(self.num_vehicles):
            model.Add(var["veh_batt"][veh_id][0] == self.veh_init_batt[veh_id])
            for t in range(self.T):
                model.Add(var["veh_is_discharging"][veh_id][t] == sum(var["z"][veh_id][loc_id][t] for loc_id in range(self.num_locs)))
                model.Add(var["veh_batt"][veh_id][t] >= self.veh_min_batt).OnlyEnforceIf(var["veh_is_discharging"][veh_id][t])
        # for locations
        for loc_id in range(self.num_locs):
            model.Add(var["loc_batt"][loc_id][0] == self.loc_init_batt[loc_id])
            for t in range(self.T):
                # implement enable_slack
                model.Add(var["loc_batt"][loc_id][t] < self.loc_min_batt).OnlyEnforceIf(var["enable_slack"][loc_id][t])
                model.Add(var["loc_batt"][loc_id][t] >= self.loc_min_batt).OnlyEnforceIf(var["enable_slack"][loc_id][t].Not())
                # implement loc_slack
                model.Add(var["loc_slack"][loc_id][t] >= 1).OnlyEnforceIf(var["enable_slack"][loc_id][t])
                model.Add(var["loc_slack"][loc_id][t] == 0).OnlyEnforceIf(var["enable_slack"][loc_id][t].Not())
                # add a constraint
                model.Add(var["loc_batt"][loc_id][t] + var["loc_slack"][loc_id][t] >= self.loc_min_batt)

    def ensure_route_continuity(self, model, var):
        """
        Parameters
        ----------

        """
        for veh_id in range(self.num_vehicles):
            # set initial position 
            # outflow arcs of the first node for a vehcile is 1
            outflow_arcs_from_init_depot = self.tsn.outflow_arcs(self.veh_init_position_id[veh_id], 0)
            model.Add(sum(var["x"][veh_id, arc] for arc in outflow_arcs_from_init_depot) == 1)
#             确保每辆车从其初始位置出发
# 每辆车必须且只能选择一条从初始位置出发的路径
            # 
            for node_id in range(self.num_nodes):
                if node_id in self.veh_init_position_id:
                    continue
                outflow_arcs = self.tsn.outflow_arcs(node_id, 0)
                for arc in outflow_arcs:
                    model.Add(var["x"][veh_id, arc] == 0)
# 禁止车辆在t=0时从非初始位置出发
# 确保车辆只能从指定的初始位置开始
            # route continuity: the number outflow arcs shold equals to the number of inflow arcs in a node
            # To handle sparsified TSN, we use get_xxx_arcs function
            for t in range(1, self.T-1):
                for n in range(self.num_nodes):
                    inflow_arcs  = self.tsn.inflow_arcs(n, t)  # get valid inflow arcs
                    outflow_arcs = self.tsn.outflow_arcs(n, t) # get valid outflow arcs
                    model.Add(sum(var["x"][veh_id, arc] for arc in inflow_arcs) == sum(var["x"][veh_id, arc] for arc in outflow_arcs))
# 确保流量守恒：进入节点的弧数等于离开节点的弧数
# 保证路径的连续性
            # 
            for t in range(0, self.T):
                for n in range(self.num_nodes):
                    if t == 0:
                        model.Add(var["z"][veh_id][n][t] == 0)
                    else:
                        stay_arc = self.tsn.stay_arcs(n, t) # get stay arc
                        if stay_arc is not None:
                            model.Add(var["z"][veh_id][n][t] <= var["x"][veh_id, stay_arc])

# z[veh_id][n][t]表示车辆是否在时间t停留在节点n
# t=0时刻，所有节点的停留状态为0
# 其他时刻，只有当存在停留弧时才可能停留

# # 假设场景：
# vehicles = 2      # 2辆车
# nodes = 3         # 3个节点
# time_steps = 4    # 4个时间步

# # 初始位置约束
# vehicle_0_init = [(0,1,0,1)]  # 车0从节点0到节点1的弧
# model.Add(var["x"][0, vehicle_0_init[0]] == 1)  # 必须使用这条弧

# # 路径连续性约束
# # 如果车0在t=2时进入节点1：
# inflow = [(0,1,1,2)]   # 从节点0到节点1的入流弧
# outflow = [(1,2,2,3)]  # 从节点1到节点2的出流弧
# model.Add(sum(var["x"][0,arc] for arc in inflow) == 
#          sum(var["x"][0,arc] for arc in outflow))

    def forbit_multi_veh_at_same_node(self, model, var):
        for t in range(1, self.T):
            for n in range(self.num_nodes):
                inflow_arcs = self.tsn.inflow_arcs(n, t)
                model.Add(sum(var["x"][veh_id, arc] for arc in inflow_arcs for veh_id in range(self.num_vehicles)) <= 1)
#   是禁止多车同时访问同一节点的约束：

# 遍历每个时间点和节点
# 计算所有车辆到达该节点的总和
# 确保在任意时刻最多只有一辆车在一个节点

    def define_batt_behavior(self, model, var):
        prepare_t = math.ceil(self.loc_pre_time / self.dt)
        for loc_id in range(self.num_locs):
            for veh_id in range(self.num_vehicles):
                for p in range(prepare_t):
                    model.Add(var["veh_prepare_at_loc"][veh_id][loc_id][p] == 0)
                    model.Add(var["veh_cleanup_at_loc"][veh_id][loc_id][self.T-p-1] == 0)
                model.Add(var["veh_prepare_at_loc2"][veh_id][loc_id][self.T-1] == 0)
                model.Add(var["veh_cleanup_at_loc2"][veh_id][loc_id][0] == 0)

# 这是定义车辆在基站的准备和清理行为约束：

# prepare_t: 计算准备时间需要的时间步数
# 禁止在时间范围开始时进行准备操作
# 禁止在时间范围结束时进行清理操作

# # 假设场景：
# T = 10          # 总时间步数
# dt = 0.5        # 时间步长
# loc_pre_time = 1.0  # 准备时间
# prepare_t = math.ceil(1.0/0.5) = 2  # 需要2个时间步

# # 约束示例：
# # 1. 多车访问约束
# # 在t=5时刻，节点2：
# inflow_arcs = [(1,2,4,5), (3,2,4,5)]  # 到达节点2的弧
# # 保证 x[0,arc1] + x[1,arc1] + x[0,arc2] + x[1,arc2] <= 1

# # 2. 电池行为约束
# # 对于基站0和车辆1：
# # 前prepare_t时间步不能准备：
# veh_prepare_at_loc[1][0][0] = 0
# veh_prepare_at_loc[1][0][1] = 0
# # 最后prepare_t时间步不能清理：
# veh_cleanup_at_loc[1][0][9] = 0
# veh_cleanup_at_loc[1][0][8] = 0
        for t in range(self.T-1):
            prev_t = t
            curr_t = t + 1
            # for EVs
            arriving_arcs = self.tsn.arriving_arcs(curr_t)
            for veh_id in range(self.num_vehicles):
                # charging: depot -> vehcile
                # 充电站给车辆充电
                model.Add(var["veh_charge_amount"][veh_id][curr_t] == sum([int(self.depot_discharge_rate[depot_offst_id] * self.dt) * var["z"][veh_id][depot_id][curr_t] for depot_offst_id, depot_id in enumerate(range(self.num_locs, self.num_nodes))])
                          - sum(int(self.veh_discharge_rate[veh_id] * self.depot_surplus_pre_time) * var["veh_prepare_at_depot"][veh_id][depot_offst_id][curr_t] for depot_offst_id in range(self.num_depots))   # TODO
                          - sum(int(self.veh_discharge_rate[veh_id] * self.depot_surplus_post_time) * var["veh_cleanup_at_depot"][veh_id][depot_offst_id][curr_t] for depot_offst_id in range(self.num_depots))) # TODO
                 # 从充电站获得的电量 # 减去准备阶段消耗  # 减去清理阶段消耗 
                # EV's battery change
                model.Add(var["veh_batt"][veh_id][curr_t] == var["veh_batt"][veh_id][prev_t] 
                          - sum([int(self.veh_discharge_rate[veh_id] * self.dt) * var["z"][veh_id][i][curr_t] for i in range(self.num_locs)]) # discharge consumption 
                          - sum([int(self.veh_consump_rate[veh_id] * self.tsn.arc_distance(arc)) * var["x"][veh_id, arc] for arc in arriving_arcs])              # travel consumption
                          + var["veh_charge_amount"][veh_id][curr_t])                                                     # power charge form a charge station 
            # 车辆电池状态变化 # 上一时刻电量  # 给基站供电消耗  # 行驶消耗 # 充电获得 
            # for locations
            for loc_id in range(self.num_locs):
                for veh_id in range(self.num_vehicles):
                    if prev_t + prepare_t < self.T: # if prepare time does not exceed the time horizon
                        # implement veh_prepare_at_loc & veh_cleanup_at_loc
                        # 基站准备和清理状态转换
                        diff_z_loc = var["z"][veh_id][loc_id][prev_t+prepare_t] - var["z"][veh_id][loc_id][prev_t]
                        # 基站充电量计算
                        model.Add(diff_z_loc == 1).OnlyEnforceIf(var["veh_prepare_at_loc"][veh_id][loc_id][prev_t+prepare_t])
                        model.Add(diff_z_loc <= 0).OnlyEnforceIf(var["veh_prepare_at_loc"][veh_id][loc_id][prev_t+prepare_t].Not())
                        model.Add(-diff_z_loc == 1).OnlyEnforceIf(var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t])
                        model.Add(-diff_z_loc <= 0).OnlyEnforceIf(var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t].Not())
                    # implement veh_prepare_at_loc & veh_cleanup_at_loc
                    diff_veh_prepare_at_loc = var["veh_prepare_at_loc"][veh_id][loc_id][prev_t] - var["veh_prepare_at_loc"][veh_id][loc_id][curr_t]
                    diff_veh_cleanup_at_loc = var["veh_cleanup_at_loc"][veh_id][loc_id][curr_t] - var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t]
                    model.Add(diff_veh_prepare_at_loc == 1).OnlyEnforceIf(var["veh_prepare_at_loc2"][veh_id][loc_id][prev_t])
                    model.Add(diff_veh_prepare_at_loc <= 0).OnlyEnforceIf(var["veh_prepare_at_loc2"][veh_id][loc_id][prev_t].Not())
                    model.Add(diff_veh_cleanup_at_loc == 1).OnlyEnforceIf(var["veh_cleanup_at_loc2"][veh_id][loc_id][curr_t])
                    model.Add(diff_veh_cleanup_at_loc <= 0).OnlyEnforceIf(var["veh_cleanup_at_loc2"][veh_id][loc_id][curr_t].Not())
                if self.ensure_minimum_supply:
                    for veh_id in range(self.num_vehicles):
                        # implement loc_supply_not_enough
                        model.Add(self.loc_cap[loc_id] - var["loc_batt"][loc_id][prev_t] >  0).OnlyEnforceIf(var["loc_supply_notenough"][veh_id][loc_id][prev_t])
                        model.Add(self.loc_cap[loc_id] - var["loc_batt"][loc_id][prev_t] <= 0).OnlyEnforceIf(var["loc_supply_notenough"][veh_id][loc_id][prev_t].Not())
                        # implement loc_suuply_notenough_notcleanup
                        model.AddImplication(var["loc_suuply_notenough_notcleanup"][veh_id][loc_id][prev_t], var["loc_supply_notenough"][veh_id][loc_id][prev_t])
                        model.AddImplication(var["loc_suuply_notenough_notcleanup"][veh_id][loc_id][prev_t], var["veh_cleanup_at_loc"][veh_id][loc_id][prev_t].Not())
                        # add a constraint
                        model.Add(var["z"][veh_id][loc_id][curr_t] >= var["z"][veh_id][loc_id][prev_t]).OnlyEnforceIf(var["loc_suuply_notenough_notcleanup"][veh_id][loc_id][prev_t])

                # supplying: 
                model.Add(var["loc_charge_amount"][loc_id][curr_t] == int(self.veh_discharge_rate[veh_id] * self.dt) * sum([var["z"][veh_id_][loc_id][curr_t] - var["veh_prepare_at_loc"][veh_id_][loc_id][curr_t] - var["veh_cleanup_at_loc"][veh_id_][loc_id][curr_t] for veh_id_ in range(self.num_vehicles)])
                                                                      + int(self.veh_discharge_rate[veh_id] * (self.dt - self.loc_surplus_pre_time)) * sum(var["veh_prepare_at_loc2"][veh_id_][loc_id][curr_t] for veh_id_ in range(self.num_vehicles))
                                                                      + int(self.veh_discharge_rate[veh_id] * (self.dt - self.loc_surplus_post_time)) * sum(var["veh_cleanup_at_loc2"][veh_id_][loc_id][curr_t] for veh_id_ in range(self.num_vehicles)))
                # location's battery change
                model.Add(var["loc_batt"][loc_id][curr_t] == var["loc_batt"][loc_id][prev_t]
                            - (1 - var["loc_is_down"][loc_id][curr_t]) * int(self.loc_consump_rate[loc_id] * self.dt)
                            + var["loc_charge_amount"][loc_id][curr_t]
                            ).OnlyEnforceIf(var["loc_is_normal"][loc_id][curr_t])
                
                # clippling battery
                # implement loc_is_down & loc_is_full
                loc_curr_batt = var["loc_batt"][loc_id][prev_t] - int(self.loc_consump_rate[loc_id] * self.dt) + var["loc_charge_amount"][loc_id][curr_t]
                model.Add(loc_curr_batt <= 0).OnlyEnforceIf(var["loc_is_down"][loc_id][curr_t])
                model.Add(loc_curr_batt >  0).OnlyEnforceIf(var["loc_is_down"][loc_id][curr_t].Not())
                model.Add(loc_curr_batt >= self.loc_cap[loc_id]).OnlyEnforceIf(var["loc_is_full"][loc_id][curr_t])
                model.Add(loc_curr_batt <  self.loc_cap[loc_id]).OnlyEnforceIf(var["loc_is_full"][loc_id][curr_t].Not())
                # clip battery
                model.Add(var["loc_batt"][loc_id][curr_t] == 0).OnlyEnforceIf(var["loc_is_down"][loc_id][curr_t])
                model.Add(var["loc_batt"][loc_id][curr_t] == self.loc_cap[loc_id]).OnlyEnforceIf(var["loc_is_full"][loc_id][curr_t])

            # for depots
            for depot_offst_id, depot_id in enumerate(range(self.num_locs, self.num_nodes)):
                for veh_id in range(self.num_vehicles):
                    # implement veh_prepare_at_depot & veh_cleanup_at_depot
                    diff_z_depot = var["z"][veh_id][depot_id][curr_t] - var["z"][veh_id][depot_id][prev_t]
                    model.Add(diff_z_depot == 1).OnlyEnforceIf(var["veh_prepare_at_depot"][veh_id][depot_offst_id][curr_t])
                    model.Add(diff_z_depot <= 0).OnlyEnforceIf(var["veh_prepare_at_depot"][veh_id][depot_offst_id][curr_t].Not())
                    model.Add(-diff_z_depot == 1).OnlyEnforceIf(var["veh_cleanup_at_depot"][veh_id][depot_offst_id][prev_t])
                    model.Add(-diff_z_depot <= 0).OnlyEnforceIf(var["veh_cleanup_at_depot"][veh_id][depot_offst_id][prev_t].Not())
                if self.ensure_minimum_charge:
                    for veh_id in range(self.num_vehicles):
                        # implement veh_charge_notenough
                        model.Add(self.veh_cap[veh_id] - var["veh_batt"][veh_id][prev_t] >  0).OnlyEnforceIf(var["veh_charge_notenough"][veh_id][depot_offst_id][prev_t])
                        model.Add(self.veh_cap[veh_id] - var["veh_batt"][veh_id][prev_t] <= 0).OnlyEnforceIf(var["veh_charge_notenough"][veh_id][depot_offst_id][prev_t].Not())
                        # implement 
                        model.AddImplication(var["veh_charge_notenough_notcleanup"][veh_id][depot_offst_id][prev_t], var["veh_charge_notenough"][veh_id][depot_offst_id][prev_t])
                        model.AddImplication(var["veh_charge_notenough_notcleanup"][veh_id][depot_offst_id][prev_t], var["veh_cleanup_at_depot"][veh_id][depot_offst_id][prev_t].Not())
                        # add a constraint
                        model.Add(var["z"][veh_id][depot_id][curr_t] >= var["z"][veh_id][depot_id][prev_t]).OnlyEnforceIf(var["veh_charge_notenough_notcleanup"][veh_id][depot_offst_id][prev_t])
# 追踪和更新车辆和基站的电量状态
# 确保电量变化符合物理约束
# 管理充放电过程的各个阶段
# 处理异常状态(如断电、满电等)
       
        for t in range(self.T):
            for loc_id in range(self.num_locs):
                model.AddBoolOr([var["loc_is_down"][loc_id][t], var["loc_is_full"][loc_id][t], var["loc_is_normal"][loc_id][t]]) # the sate of a location is down or ful or normal
                model.Add(var["loc_batt"][loc_id][t] <= 0).OnlyEnforceIf(var["loc_is_down2"][loc_id][t])
                model.Add(var["loc_batt"][loc_id][t] >= 1).OnlyEnforceIf(var["loc_is_down2"][loc_id][t].Not())

    def add_objective_constraints(self, model, var):
        # 计算每个时间步的断电基站数量
        for t in range(self.T):
            model.Add(var["num_down_loc"][t] == 
                     sum(var["loc_is_down2"][loc_id_][t] for loc_id_ in range(self.num_locs)))
        
        # 计算每辆车的总行驶距离
        for veh_id in range(self.num_vehicles):
            model.Add(var["travel_distance"][veh_id] == 
                     sum(var["x"][veh_id, arc] * self.tsn.arc_distance(arc) 
                         for arc in self.tsn.arcs))
#             对每个时间步t，计算该时刻断电的基站数量
# var["loc_is_down2"][loc_id][t]是布尔值，表示基站是否断电
# 通过求和得到总断电基站数

    def add_objectives(self, model, var, loss_coef: int):
        # 计算平均行驶距离
        avg_travel_distance = sum(var["travel_distance"][veh_id] 
                                for veh_id in range(self.num_vehicles)) * \
                             int(BIT_LARGE_VALUE / self.num_vehicles)
        
        # 计算断电率
        down_rate = sum(var["num_down_loc"][t] 
                       for t in range(self.T)) * \
                    int(LARGE_VALUE / (self.num_locs * self.T)) * BIT_LARGE_VALUE
        
        # 设置损失函数值
        model.Add(var["loss1"] == avg_travel_distance)
        model.Add(var["loss2"] == loss_coef * down_rate)
        
        # 定义最小化目标
        model.Minimize(avg_travel_distance + loss_coef * down_rate)


#         使用LARGE_VALUE和BIT_LARGE_VALUE进行数值缩放
# 通过loss_coef调节目标函数各部分的权重
# 目标函数考虑了效率(距离)和可靠性(断电)的平衡
# 优化目的：
# 减少车辆行驶距离，提高效率
# 减少基站断电情况，保证供电可靠性
# 在两个目标之间寻找平衡点