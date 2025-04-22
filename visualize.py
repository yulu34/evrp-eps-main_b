from utils.util import load_dataset  # 加载数据集的工具函数
import torch                         # PyTorch深度学习框架
import numpy as np                   # 数值计算库
from copy import copy               # 用于对象复制
from PIL import Image               # 图像处理库
from torchvision import transforms  # PyTorch图像变换工具
import matplotlib.pyplot as plt      # 主要绘图库
import matplotlib.cm as cm          # 颜色映射模块
import matplotlib.animation as animation # 动画模块
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  
# OffsetImage: 在图上添加图像
# AnnotationBbox: 为图像添加标注框

from matplotlib import patches      # 绘制各种形状(矩形、圆形等)

DPI = 150  # 之后改 
# 数值越大，图像质量越高，文件也越大 之后改 小一点 能看就行
# 150是一个平衡值，提供清晰度的同时保持合理的文件大小
VIS_OFFSET = 0.02  # 可视化偏移量、
# 用于调整车辆图标在图上的垂直位置
# 避免车辆图标和其他元素(如路线、基站)重叠
# 提供更清晰的视觉效果
BATT_OFFSET = 0.2  # 电池图标偏移量
# 控制电池状态指示器相对于车辆图标的位置偏移
# 使电池状态显示在适当位置
# 避免与车辆图标重叠


def add_base(x, y, ratio, ax, node_id=None):
    # Always use green color regardless of ratio
    width = 0.01 * 1.5
    height = 0.015 * 1.5
    battery_color = "limegreen"  # Always green
    
    frame = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height, fill=False, ec="black")
    battery = patches.Rectangle(xy=(x-width/2, y-height/2), width=width, height=height, facecolor=battery_color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame) 
    
    # Add node ID if provided
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

    

def add_vehicle(x, y, ratio, color, ax, offst=0.0):
    # 绘制车辆图标
    # - x,y: 车辆位置坐标
    # - ratio: 车辆电量比例(当前电量/最大容量)
    # - color: 车辆颜色
    # - ax: matplotlib轴对象
    # - offst: 车辆图标的垂直偏移量
    # 车辆图标的宽度和高度
    # vehicle_battery
    # ratio = 0.4
    width = 0.015
    height = 0.01
    width_mod = ratio * width
    # 根据电量决定边框颜色
    if ratio < 1e-9:   # 电量耗尽
        ec = "red"     # 红色边框警告
    else:
        ec = "black"   # 正常黑色边框
    frame = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width, height=height, fill=False, ec=ec)
    battery = patches.Rectangle(xy=(x-width/2, y-height/2+offst+BATT_OFFSET), width=width_mod, height=height, facecolor=color, linewidth=.5, ec="black")
    ax.add_patch(battery)
    ax.add_patch(frame)

    # vehicle
    original_img = plt.imread("images/ev_image.png")
    # 加载车辆图片
    original_img = plt.imread("images/ev_image.png")
    
    # 设置车辆颜色
    vehicle_img = np.where(
        original_img == (1., 1., 1., 1.),  # 找到白色像素
        (color[0], color[1], color[2], color[3]),  # 替换为指定颜色
        original_img  # 保持其他像素不变
    )
    
    # 创建车辆图像对象
    vehicle_img = OffsetImage(vehicle_img, zoom=0.1)  # 缩放图像
    
    # 将图像添加到图表
    ab = AnnotationBbox(
        vehicle_img, 
        (x, y+offst),  # 位置
        xycoords='data',
        frameon=False   # 不显示边框
    )


def visualize_tour(dataset_path, tour_path, save_dir, instance, anim_type):
    dataset = load_dataset(dataset_path)  # 加载数据集
    tours = load_dataset(tour_path)       # 加载路线信息
    data = dataset[instance]              # 获取特定实例的数据
    tour = tours[instance]                # 获取特定实例的路线

# # 假设某辆车的路线可能是:
# tour[0] = [
#     [0, 0.5, 1.0],    # 访问节点0, 行驶0.5小时, 充电1.0小时
#     [5, 0.8, 0.5],    # 访问节点5, 行驶0.8小时, 充电0.5小时
#     [2, 0.6, 0.7]     # 访问节点2, 行驶0.6小时, 充电0.7小时
# ]

    loc_coords = data["loc_coords"]       # 基站坐标 [num_locs x coord_dim]
    depot_coords = data["depot_coords"]   # 充电站坐标 [num_depots x coord_dim]
    coords = torch.cat((loc_coords, depot_coords), 0)  # 合并所有坐标
    x_loc = loc_coords[:, 0]; y_loc = loc_coords[:, 1]        # 基站x,y坐标
    x_depot = depot_coords[:, 0]; y_depot = depot_coords[:, 1] # 充电站x,y坐标

    num_locs = len(x_loc)        # 基站数量
    num_depots = len(x_depot)    # 充电站数量
    num_vehicles = len(tour)     # 车辆数量
    vehicle_steps = [0] * num_vehicles  # 每辆车当前步骤
#     # 创建一个长度为num_vehicles的列表，初始值都是0
#     每个元素对应一辆车的当前步骤数
# 初始值全部设为0，表示所有车辆都在起点
    
    # 时间相关参数
    vehicle_travel_time = np.zeros(num_vehicles)    # 行驶时间 
    vehicle_charge_time = np.zeros(num_vehicles)    # 充电时间  #可以输出吗 #真的影响吗
    vehicle_unavail_time = np.zeros(num_vehicles)   # 不可用时间  #可以输出吗 #真的影响吗
    estimated_unavail_time = np.zeros(num_vehicles) # 预估不可用时间  #可以输出吗 #真的影响吗
    vehicle_phase = ["move" for _ in range(num_vehicles)]  # 车辆状态("move"/"charge")  #可以输出吗 #真的影响吗 
    finished = ["end" for _ in range(num_vehicles)]        # 完成状态标记  #可以输出吗 #真的影响吗 
    vehicle_visit = np.zeros((num_vehicles, 2, 2))        # 车辆访问点坐标 
    vehicle_max_steps = [len(tour[vehicle_id]) for vehicle_id in range(num_vehicles)]  # 最大步数
    # loc_battery = data["loc_initial_battery"] # [num_locs] 
    # 基站初始电量
    # loc_cap = data["loc_cap"] # [num_locs]
    # 基站最大电量
    # loc_consump_rate = data["loc_consump_rate"] # [num_locs]  #可以输出吗 #真的影响吗
    # 基站电量消耗速率
    vehicle_discharge_rate = data["vehicle_discharge_rate"] # [num_vehicles]  #可以输出吗 #真的影响吗
    # 车辆电量消耗速率
    vehicle_position = np.zeros(num_vehicles).astype(int) # [num_vehicles]
    # 车辆当前位置
    vehicle_battery = data["vehicle_cap"].clone() # [num_vehicles]
    # 车辆初始电量
    vehicle_cap = data["vehicle_cap"].clone() # [num_vehicles]  #可以输出吗 #真的影响吗
    # 车辆最大电量
    depot_discharge_rate = data["depot_discharge_rate"] #最好统一
     # 充电站充电率
    curr_time = 0.0
     # 当前时间
    
    # 根据车辆数量选择合适的颜色映射
    if num_vehicles <= 10:
        cm_name = "tab10"    # 10种不同的颜色
    elif num_vehicles <= 20:
        cm_name = "tab20"    # 20种不同的颜色
    else:
        assert False         # 超过20辆车时报错
    
    # 获取颜色映射对象
    cmap = cm.get_cmap(cm_name)
    #之后改 一般不要超过20辆车
    #规模不要太大 训练成本也高

    #----------------------------
    # visualize the initial step
    #----------------------------
    # initialize a fig instance
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
#     这个 111 是一个三位数字的缩写，实际上代表了三个数字：

# 第一个 1: 表示图表的行数
# 第二个 1: 表示图表的列数
# 第三个 1: 表示当前子图的索引号

    # add locations & depots
    for id in range(num_locs):
        ratio = loc_battery[id] / loc_cap[id]
        add_base(x_loc[id], y_loc[id], ratio, ax)
    ax.scatter(x_depot, y_depot,    # 充电站的x,y坐标
              marker="*",           # 使用星形标记
              c="black",           # 标记颜色为黑色
              s=100,               # 标记大小为100
              zorder=3)            # 绘制层级为3(值越大越在上层)
    #充电站用黑色星星标记

    # initial vehicle assignment
    NODE_ID = 0; TRAVEL_TIME = 1; CHARGE_TIME = 2
#     NODE_ID: 节点ID的索引位置
# TRAVEL_TIME: 行驶时间的索引位置
# CHARGE_TIME: 充电时间的索引位置
    for i in range(num_vehicles):
        vehicle_steps[i] += 1  # 更新车辆步数
        vehicle_position[i] = tour[i][vehicle_steps[i]][NODE_ID]  # 更新车辆位置
        # 存储当前位置和下一个位置的坐标
        vehicle_visit[i, 0, 0] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 0]  # 当前x坐标
        # vehicle_visit 是一个形状为 [num_vehicles, 2, 2] 的数组
# i: 车辆编号
# : : 所有点(当前点和下一个点)
# 0/1: 0表示x坐标，1表示y坐标
# vehicle_visit[i, :, 0]  # 车辆i的所有点的x坐标
# vehicle_visit[i, :, 1]  # 车辆i的所有点的y坐标
        vehicle_visit[i, 0, 1] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 1]  # 当前y坐标
        vehicle_visit[i, 1, 0] = coords[tour[i][vehicle_steps[i]][NODE_ID], 0]    # 下一个x坐标
        vehicle_visit[i, 1, 1] = coords[tour[i][vehicle_steps[i]][NODE_ID], 1]    # 下一个y坐标
        ax.plot(vehicle_visit[i, :, 0], vehicle_visit[i, :, 1], 
                zorder=0,       # 绘制层级
                alpha=0.5,      # 透明度
                linestyle="--", # 虚线样式
                color=cmap(i))  # 使用颜色映射为每辆车分配不同颜色
        
        vehicle_travel_time[i] = tour[i][vehicle_steps[i]][TRAVEL_TIME]  #可以输出吗 #真的影响吗
        # 行驶时间初始化为当前步骤的行驶时间
        vehicle_charge_time[i] = tour[i][vehicle_steps[i]][CHARGE_TIME]  #可以输出吗 #真的影响吗
    
#     # tour 的结构是一个三维列表:
# tour = [  # 车辆列表
#    [  # 每辆车的路线
#       [node_id, travel_time, charge_time],  # 每个步骤的信息
#       [node_id, travel_time, charge_time],
#       ...
#    ],
#    [...],  # 第二辆车的路线
#    ...
# ]
# # 假设有这样的数据:
# tour = [
#    [  # 第0辆车的路线
#       [1, 0.5, 1.0],  # 节点1，行驶0.5小时，充电1.0小时
#       [2, 0.8, 0.5],  # 节点2，行驶0.8小时，充电0.5小时
#    ],
#    # ... 其他车辆的路线
# ]

# # 如果 i = 0 且 vehicle_steps[0] = 1，那么:
# vehicle_charge_time[0] = tour[0][1][2]  # 结果为 0.5
        # 充电时间初始化为当前步骤的充电时间
        vehicle_unavail_time[i] = vehicle_travel_time[i]  #可以输出吗 #真的影响吗
        # 车辆的不可用时间初始化为行驶时间  它就是行驶时间
        estimated_unavail_time[i] = vehicle_travel_time[i]  #可以输出吗 #真的影响吗
        # 预估不可用时间初始化为行驶时间

        # add a vehicle to image
        ratio = vehicle_battery[i] / vehicle_cap[i] 
        add_vehicle(vehicle_visit[i, 0, 0], vehicle_visit[i, 0, 1], ratio, cmap(i), ax, VIS_OFFSET)

    # plt.savefig(f"{save_dir}/vis/png/tour_test.png")
    # 这里可以要想动画必须绘图
    ax.set_title(f"current_time = {curr_time:.3f}")
    plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05)
    plt.savefig(f"{save_dir}/vis/png/tour_test0.png", dpi=DPI)
    plt.close()

    #-------------------------------
    # visualize the subseqent steps
    #-------------------------------
    total_steps = 1
    all_finished = False 
    while not all_finished:
        # select next vehicle
        next_vehicle_id = np.argmin(vehicle_unavail_time)
        # 选择下一个车辆的ID
        # np.argmin: 返回数组中最小值的索引
        # vehicle_unavail_time: 车辆不可用时间
        # 选择不可用时间最小的车辆 这个不可用时间等于最小剩余行驶时间

        i = next_vehicle_id
        # check if all vehicles are finished
        # 检查所有车辆是否完成
        # 如果所有车辆的不可用时间都为0，表示所有车辆都完成了
        # 这里的 vehicle_unavail_time[i] 是当前车辆的不可用时间
        # 如果所有车辆的不可用时间都为0，表示所有车辆都完成了

        # update time
        # 1. 保存当前选中车辆的不可用时间
        elapsed_time = vehicle_unavail_time[i].copy()  
        
        # 2. 更新全局时间，加上当前车辆的不可用时间
        curr_time += vehicle_unavail_time[i]  
        # 这是右图下方不断变化的时间
        
        # 3. 所有车辆的不可用时间都减去刚经过的时间
        vehicle_unavail_time -= vehicle_unavail_time[i]   #可以输出吗 #真的影响吗
        # 以次来减小所有车辆的不可用时间
        # 这样可以确保所有车辆的不可用时间都能被更新
            #vehicle_unavail_time 是一个数组，长度等于车辆数量，记录每辆车的"不可用时间"：
#             当车辆在移动时：不可用时间 = 行驶时间
# 当车辆在充电时：不可用时间 = 充电时间

        # 这里的 vehicle_unavail_time[i] 是当前车辆的不可用时间
        # 这里的 vehicle_unavail_time 是所有车辆的不可用时间
        #表示特定车辆 i 的不可用时间
                # 例如：vehicle_unavail_time = [2.5, 1.0, 3.0]
        # i = 1
        # vehicle_unavail_time[i]  # 结果为 1.0，表示第2辆车还需要1.0小时才能完成当前任务
                # 例如：vehicle_unavail_time = [2.5, 1.0, 3.0]
        # 表示：
        # - 第1辆车还需要2.5小时
        # - 第2辆车还需要1.0小时
        # - 第3辆车还需要3.0小时
        # 4. 确保所有不可用时间不会小于0
        vehicle_unavail_time = vehicle_unavail_time.clip(0.0)  

        # initialize a fig instance
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # add locations & depots
        charged_loc = []  # 用于记录已被充电的基站  #可以输出吗 #真的影响吗 
        # 之后选择时不会选择已经充电的客户
        # 之后改 这里有点问题
        for vehicle_id in range(num_vehicles):
            loc_id = vehicle_position[vehicle_id]  # 获取车辆当前位置
            at_loc = loc_id < num_locs  # 判断是否在基站(True)还是充电站(False)  
            # loc_id: 车辆当前位置的索引
            # vehicle_position: 车辆当前位置的索引  就是判断id
            charging = vehicle_phase[vehicle_id] == "charge"  # 判断车辆是否在充电状态  #可以输出吗 #真的影响吗
            #一个bool值
            # vehicle_phase: 车辆的状态("move"/"charge")
            # charging: 车辆是否在充电状态

#             if at_loc & charging:  # 如果在基站且处于充电状态
#                 # 基站电量随着unavail_time时间增加
#                 loc_battery[loc_id] += vehicle_discharge_rate[vehicle_id] * elapsed_time #删除
#                 # 车辆电量相应减少
#                 vehicle_battery[vehicle_id] -= vehicle_discharge_rate[vehicle_id] * elapsed_time #删除
#                 # 确保车辆电量不会变成负数
#                 # 论文中对车辆电量有保护：只能（充到beta%*客户电量-车电量）/（车放电量速度-客户充电量速度）
#                 # 或充到车辆底线时停止充电（车电量下限 and 到最近电站的距离 中更大的一个）
#                 # 之后改 这里作图时不知道有没有设计这个逻辑
# # 真正的保护逻辑是在模型训练和状态更新的代码中实现的，而不是在可视化代码中。
# # 因为逻辑不会体现在可视化中
#                 # 这里的 vehicle_battery 是一个数组，长度等于车辆数量
#                 vehicle_battery[vehicle_id] = vehicle_battery[vehicle_id].clip(0.0)
#                 # 计算基站电量比例并更新显示
#                 ratio = loc_battery[loc_id] / loc_cap[loc_id]
#                 add_base(x_loc[loc_id], y_loc[loc_id], ratio, ax)
#                 # 记录已充电的基站
#                 charged_loc.append(loc_id)
                # 这些是论文中的公式
            if (not at_loc) & charging:  #可以输出吗 #真的影响吗
                # 如果在充电站且处于充电状态
                vehicle_battery[vehicle_id] += depot_discharge_rate[loc_id - num_locs] * elapsed_time
                # 车电量相应增加
            
        for id in range(num_locs): 
            if not (id in charged_loc):  # 如果该基站不在已充电列表中
                # loc_battery[id] -= loc_consump_rate[id] * elapsed_time 
                # 基站电量随着unavail_time时间减少
                # loc_battery[id] = loc_battery[id].clamp(0.0) 
                # 确保基站电量不会变成负数
                
                # ratio = loc_battery[id] / loc_cap[id]
                # 计算基站电量比例并更新显示
                add_base(x_loc[id], y_loc[id], ratio, ax)
                # 绘制基站电量
        # ax.scatter(x_loc, y_loc, marker="o", c="black", zorder=3)
        ax.scatter(x_depot, y_depot, marker="*", c="black", s=100, zorder=3)
# zorder=3 表示充电站的标记将被绘制在 zorder 值小于 3 的其他元素之上。
# 通常在代码中可以看到以下顺序:
# 路径线条 zorder=0 (最底层)
# 基站标记 默认值
# 充电站标记 zorder=3 (较上层)
# 车辆图标 (最上层) 可以通过 Plot z-order 设置来改变默认的绘制顺序
        #-----------------------------------
        # visualization of selected vehicle
        #-----------------------------------
        # add the path
        if vehicle_phase[i] == "move":
            ax.plot(vehicle_visit[i, :, 0], vehicle_visit[i, :, 1], zorder=0, linestyle="-", color=cmap(i))
        # add selected vehicle to image
        ratio = vehicle_battery[i] / vehicle_cap[i]
        add_vehicle(vehicle_visit[i, 1, 0], vehicle_visit[i, 1, 1], ratio, cmap(i), ax, VIS_OFFSET)

        #---------------------------------
        # visualization of other vehicles
        #---------------------------------
        for k in range(num_vehicles):
            if k != i:  # 如果不是当前选中的车辆
                x_st  = vehicle_visit[k, 0, 0]; y_st  = vehicle_visit[k, 0, 1]
                # 车辆的起始坐标
                # vehicle_visit 是一个形状为 [num_vehicles, 2, 2] 的数组
                x_end = vehicle_visit[k, 1, 0]; y_end = vehicle_visit[k, 1, 1]
                # 车辆的结束坐标    
                # vehicle_visit 是一个形状为 [num_vehicles, 2, 2] 的数组
                if vehicle_phase[k] == "move":  # 如果车辆在移动
                    # 计算移动进度(0到1之间)
                    progress = 1.0 - (vehicle_unavail_time[k] / estimated_unavail_time[k])
                    
                    # 计算当前位置（线性插值）
                    x_curr = progress * (x_end - x_st) + x_st
                    y_curr = progress * (y_end - y_st) + y_st
                    # 像做预测 不过是一次函数
                    # 绘制车辆的起始位置
                    
                    # 绘制已经走过的路径（实线）
                    ax.plot([x_st, x_curr], [y_st, y_curr], zorder=0, linestyle="-", color=cmap(k))
                    # 绘制剩余路径（虚线）
                    ax.plot([x_curr, x_end], [y_curr, y_end], zorder=0, alpha=0.5, linestyle="--", color=cmap(k))
                    
                    # 设置车辆当前位置
                    x_vehicle = x_curr
                    y_vehicle = y_curr
                    vis_offst = 0.0  # 移动时不需要垂直偏移
                else: # 如果车辆在充电或其他状态
                    x_vehicle = x_end; y_vehicle = y_end
                    vis_offst = VIS_OFFSET# 停止时需要垂直偏移以避免重叠
                # add other vehicles to image
                ratio = vehicle_battery[k] / vehicle_cap[k]
                
                add_vehicle(x_vehicle, y_vehicle, ratio, cmap(k), ax, vis_offst)

        #--------------
        # update state
        #--------------
        if vehicle_phase[i] == "move": 
            #当车辆在移动状态 中更新状态包括 充电时间  
            # 更新为充电时间
            vehicle_unavail_time[i] = vehicle_charge_time[i].copy()
            estimated_unavail_time[i] = vehicle_charge_time[i].copy()
            
            # 检查是否到达路线终点
            if vehicle_steps[next_vehicle_id] >= vehicle_max_steps[next_vehicle_id] - 1:
#                 # 假设一辆车的路线是：
# tour[0] = [
#     [0, 0.5, 1.0],  # step 0
#     [5, 0.8, 0.5],  # step 1
#     [2, 0.6, 0.7]   # step 2 (最后一步)
# ]

# vehicle_max_steps[0] = 3  # 总共3个节点
# # 当 vehicle_steps[0] >= 2 时，表示车辆已经到达最后一个节点
                # 如果到达终点
                vehicle_phase[i] = "end"  # 标记为结束状态
                vehicle_unavail_time[i] = 1e+9  # 设置一个很大的不可用时间
            else:
                vehicle_phase[i] = "charge"  # 转换为充电状态

        elif vehicle_phase[i] == "charge":

                 # 执行的是充电完成后的状态更新，主要包括：
                                   # 更新车辆步数
                                   #充电完成后，准备移动到下一个节点：
                                   # 更新车辆位置为下一个节点
                                   # 设置下一段路程的行驶时间
                                   # 设置下一个节点的充电时间
                                   # 更新不可用时间为行驶时间
                                   #更新车辆的路线坐标：
                                   ## 更新起点坐标（当前位置）
                                   ## 更新终点坐标（下一个目标位置）
                                   # 充电完成后，转换为移动状态 状态转换：

            
            vehicle_steps[next_vehicle_id] += 1  # 增加车辆的步数计数
            vehicle_position[i] = tour[i][vehicle_steps[i]][NODE_ID]  # 更新车辆位置到下一个节点
            # 从路线信息中获取下一段行程的时间参数
            vehicle_travel_time[i] = tour[i][vehicle_steps[i]][TRAVEL_TIME]  # 设置行驶时间
            vehicle_charge_time[i] = tour[i][vehicle_steps[i]][CHARGE_TIME]  # 设置充电时间 
            # 更新不可用时间
            vehicle_unavail_time[i] = vehicle_travel_time[i].copy()  # 设置不可用时间
            estimated_unavail_time[i] = vehicle_travel_time[i].copy()  # 设置预估不可用时间
            # 更新当前位置坐标(起点)
            vehicle_visit[i, 0, 0] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 0]  # 当前x坐标
            vehicle_visit[i, 0, 1] = coords[tour[i][vehicle_steps[i]-1][NODE_ID], 1]  # 当前y坐标
            
            # 更新目标位置坐标(终点)
            vehicle_visit[i, 1, 0] = coords[tour[i][vehicle_steps[i]][NODE_ID], 0]  # 目标x坐标
            vehicle_visit[i, 1, 1] = coords[tour[i][vehicle_steps[i]][NODE_ID], 1]  # 目标y坐标
            vehicle_phase[i] = "move"  # 将车辆状态改为移动状态

        if elapsed_time < 1e-9:  # 如果经过的时间接近0
            plt.close()
            all_finished = not (vehicle_phase != finished)
            continue
        else:  # 保存当前帧
            ax.set_title(f"current_time = {curr_time:.3f}")  # 设置标题显示当前时间
            plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05)    # 设置坐标轴范围
            plt.savefig(f"{save_dir}/vis/png/tour_test{total_steps}.png", dpi=DPI)  # 保存当前帧
            plt.close()
        
        # plt.savefig(f"{save_dir}/vis/png/tour_test.png")

        total_steps += 1
        all_finished = not (vehicle_phase != finished)  # 检查是否所有车辆都完成任务

    # 创建动画
    gif_fig = plt.figure(figsize=(10, 10))
    pic_list = [f"{save_dir}/vis/png/tour_test{i}.png" for i in range(total_steps)]  # 收集所有帧
    ims = []
    for i in range(len(pic_list)):
        im = Image.open(pic_list[i])
        ims.append([plt.imshow(im)])
    plt.axis("off")
    gif_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    gif = animation.ArtistAnimation(gif_fig, ims, interval=500, repeat_delay=5000)  # 创建动画
    
    if anim_type == "gif":
        gif.save(f"{save_dir}/vis/test.gif", writer="pillow")
    else:
        gif.save(f"{save_dir}/vis/test.mp4", writer="ffmpeg")
    
    

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--dataset_path", type=str, required=True)  # 数据集路径
    parser.add_argument("--tour_path", type=str, required=True)     # 路线文件路径
    parser.add_argument("--save_dir", type=str, required=True)      # 保存目录
    parser.add_argument("--instance", type=int, default=0)          # 实例索引
    parser.add_argument("--anim_type", type=str, default="gif")     # 动画类型 # 之后改试试看哪种好
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(f"{args.save_dir}/vis/png", exist_ok=True)
    # 运行可视化函数
    visualize_tour(args.dataset_path, args.tour_path, args.save_dir, args.instance, args.anim_type)
