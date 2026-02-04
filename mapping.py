import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Polygon
import os
import matplotlib
import random

from urdf_reader import URDFReader

# ========== 全局可视化配置 ==========
# 一、箭头参数
# 1. 默认参数
ARROW_COLOR = '#3498db'           # 箭头统一颜色（中间蓝色）
ARROW_ALPHA = 0.7                 # 箭头透明度
ARROW_HEAD_WIDTH = 0.2            # 箭头头部宽度
ARROW_HEAD_LENGTH = 0.3           # 箭头头部长度
ARROW_ZORDER = 6                  # 箭头绘制层级
ARROW_MIN_POINTS = 2              # 绘制箭头的最小轨迹点数
ARROW_STEP_MIN = 100              # 箭头间隔最小值（每多少个点绘制一个箭头）
ARROW_STEP_RATIO = 5              # 箭头间隔比例（不超过总点数的1/ARROW_STEP_RATIO）
# 2. 机器人箭头参数
ARROW_ROBOT_COLOR = 'blue'     # 机器人箭头颜色
ARROW_ROBOT_ALPHA = 0.6        # 机器人箭头透明度
# 3. 障碍物箭头参数
ARROW_OBS_COLOR = 'red'         # 障碍物箭头颜色
ARROW_OBS_ALPHA = 0.6           # 障碍物箭头透明度

# 二、停留圆圈参数
STAY_CIRCLE_COLOR = 'orange'      # 圆圈颜色
STAY_CIRCLE_ALPHA_MIN = 0.1       # 圆圈透明度最小值
STAY_CIRCLE_ALPHA_MAX = 0.5       # 圆圈透明度最大值
STAY_TIME_THRESHOLD = 2.0         # 停留时间阈值（平均值的多少倍时绘制圆圈）
STAY_SIZE_MAX = 5.0               # 圆圈大小最大值系数
STAY_CIRCLE_ZORDER = 2            # 圆圈绘制层级
STAY_MIN_POINTS = 2               # 绘制停留圆圈的最小轨迹点数

# 三、轨迹线参数
# 1. 默认参数
TRAJECTORY_LINEWIDTH_MIN = 1.0    # 轨迹线最小宽度
TRAJECTORY_LINEWIDTH_MAX = 4.0    # 轨迹线最大宽度
TRAJECTORY_ALPHA = 0.7            # 轨迹线透明度
TRAJECTORY_ZORDER = 3             # 轨迹线绘制层级
TRAJECTORY_MIN_POINTS = 2         # 绘制轨迹的最小点数
# 2. 机器人轨迹参数
TRAJ_ROBOT_LINEWIDTH_MIN = 2.0          # 机器人轨迹线最小宽度
TRAJ_ROBOT_LINEWIDTH_MAX = 6.0          # 机器人轨迹线最大宽度
TRAJ_ROBOT_ALPHA = 0.9                  # 机器人轨迹线透明度
TRAJ_ROBOT_ZORDER = 5                   # 机器人轨迹线绘制层级
# 3. 障碍物轨迹参数
TRAJ_OBS_LINEWIDTH_MIN = 1.5
TRAJ_OBS_LINEWIDTH_MAX = 8.0
TRAJ_OBS_ALPHA = 0.7
TRAJ_OBS_ZORDER = 4

# 四、物体参数
# 1. 默认参数
DATA_LABEL_OFFSET_X = 0
DATA_LABEL_OFFSET_Y = -0.5
DATA_LABEL_FONTSIZE = 12
DATA_LABEL_BOXSTYLE = "round"
DATA_LABEL_BOXPAD = 0.3  
DATA_LABEL_BOXCOLOR = 'white'
DATA_LABEL_BOXALPHA = 0.8
DATA_LABEL_ZORDER = 30
ICON_EDGE_COLOR = 'black'
ICON_EDGE_WIDTH = 3
# 2. 机器人始终点参数
ROBOT_POINT_MARKERSIZE = 25
ROBOT_POINT_ZORDER = 10
ROBOT_START_COLOR = '#2ecc71'
ROBOT_END_COLOR = '#e74c3c'
# 3. 障碍物位置参数
OBS_POINT_MARKERSIZE = 20
OBS_POINT_COLOR = '#e74c3c'
OBS_POINT_ZORDER = 6
# 4. 目标点参数
TARGET_POINT_MARKERSIZE = 35
TARGET_POINT_COLOR = '#9b59b6'
TARGET_POINT_ZORDER = 9


# 五、图形显示参数
FIG_XY_FONTSIZE = 14          # 坐标轴字体大小
FIG_X_LABEL_CONTENT = 'X坐标 (米)'  # 坐标轴标签内容
FIG_Y_LABEL_CONTENT = 'Y坐标 (米)'  # 坐标轴标签内容
FIG_TITLE_CONTENT = '机器人导航轨迹图 - 时间渐变可视化' # 图形标题内容
FIG_TITLE_FONTSIZE = 16        # 标题字体大小
FIG_TITLE_PAD = 20            # 标题与图形的距离
GRID_MAJOR_VISIBLE = True    # 主网格可见性
GRID_MAJOR_LINE_ALPHA = 0.4  # 主网格线透明度
GRID_MAJOR_LINE_STYLE = '-'  # 主网格线样式
GRID_MAJOR_LINE_WIDTH = 0.5 # 主网格线宽度
GRID_MINOR_VISIBLE = True   # 次网格可见性
GRID_MINOR_LINE_ALPHA = 0.2 # 次网格线透明度
GRID_MINOR_LINE_STYLE = ':' # 次网格线样式
GRID_MINOR_LINE_WIDTH = 0.5 # 次网格线宽度
FIG_XY_LIM_MIN = 0         # 坐标轴范围最小值
FIG_XY_LIM_MAX = 10        # 坐标轴范围最大值
FIG_XY_TICKS_MIN = 0    # 坐标轴刻度最小值
FIG_XY_TICKS_MAX = 11   # 坐标轴刻度最大值
FIG_XY_TICK = 2          # 坐标轴刻度间隔

# ========== 中文字体设置 ==========
def setup_chinese_font():
    """设置中文字体支持"""

    try:
        # Windows系统字体路径
        if os.name == 'nt':  # Windows
            font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
            if os.path.exists(font_path):
                matplotlib.font_manager.fontManager.addfont(font_path)
                font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                plt.rcParams['font.sans-serif'] = [font_name]
            else:
                # 尝试其他中文字体
                font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
                if os.path.exists(font_path):
                    matplotlib.font_manager.fontManager.addfont(font_path)
                    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                    plt.rcParams['font.sans-serif'] = [font_name]
        
        # Linux/Mac系统字体路径
        else:
            # 常见Linux/Mac中文字体路径
            chinese_fonts = [

                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux文泉驿
            ]
            
            for font_path in chinese_fonts:
                if os.path.exists(font_path):
                    matplotlib.font_manager.fontManager.addfont(font_path)
                    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                    plt.rcParams['font.sans-serif'] = [font_name]
                    break

        # 确保能正常显示负号
        plt.rcParams['axes.unicode_minus'] = False
        
        print("✓ 中文字体设置完成")
        
    except Exception as e:
        print(f"⚠️ 中文字体设置失败: {e}")
        print("将使用默认字体，中文可能显示为方框")

# 在程序开始时调用字体设置
setup_chinese_font()


class SimpleRobotController:
    """简单机器人控制器 - 直接定点移动"""
    def __init__(self, robot_id, target_points, kp=1.5, max_velocity=2.5):
        self.robot_id = robot_id
        self.target_points = target_points  # 目标点列表 [[x1,y1], [x2,y2], ...]
        self.kp = kp      # 比例增益
        self.max_velocity = max_velocity  # 最大速度
        self.current_target_index = 0     # 当前目标点索引
        self.goal_threshold = 0.5         # 到达目标的距离阈值（米）
        
    def compute_velocity(self):
        """计算直接移动到当前目标点的速度指令"""
        if not self.target_points or self.current_target_index >= len(self.target_points):
            return None
        
        # 获取当前目标点
        target = self.target_points[self.current_target_index]
        
        # 获取机器人当前位置
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        current_x, current_y = current_pos[0], current_pos[1]
        
        # 计算到目标的距离和方向
        dx = target[0] - current_x
        dy = target[1] - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 检查是否到达当前目标点
        if distance < self.goal_threshold:
            print(f"✓ 到达目标点 {self.current_target_index}: ({target[0]:.1f}, {target[1]:.1f})")
            self.current_target_index += 1
            if self.current_target_index >= len(self.target_points):
                print("✓ 已完成所有目标点导航！")
                return None
            return self.compute_velocity()  # 计算下一个目标点的速度
        
        # 计算速度指令（简化的比例控制）
        velocity = min(self.kp * distance, self.max_velocity)
        
        # 计算方向角度
        angle = math.atan2(dy, dx)
        
        # 将速度分解为左右轮速度（差分驱动）
        left_velocity = velocity * (1 + 0.5 * math.sin(angle))
        right_velocity = velocity * (1 - 0.5 * math.sin(angle))
        
        return [left_velocity, right_velocity]

class NavigationVisualizer:
    """增强版导航可视化器"""
    def __init__(self, robot_id, obstacles_ids, dynamic_obstacles_info=None):
        self.robot_id = robot_id
        self.obstacles_ids = obstacles_ids
        self.dynamic_obstacles_info = dynamic_obstacles_info or []
        
        # # 创建图表
        # self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # 存储数据
        self.robot_positions = []  # 存储机器人轨迹点
        self.robot_timestamps = [] # 存储机器人时间戳
        self.obstacle_positions = {oid: [] for oid in obstacles_ids}  # 存储障碍物轨迹点
        self.obstacle_timestamps = {oid: [] for oid in obstacles_ids} # 存储障碍物时间戳
        
        # 创建自定义颜色映射
        self.create_custom_colormaps()

        # self.planned_path = None
        
    def create_custom_colormaps(self):
        """创建自定义颜色映射"""
        # 机器人轨迹颜色映射（蓝色渐变）
        self.robot_cmap = LinearSegmentedColormap.from_list(
            'robot_path',
            ['#a5cdf3', '#4c94c9', '#78b0de', '#1f77b4'],  # 浅蓝到深蓝
            N=256
        )
        
        # 障碍物轨迹颜色映射（橙色到红色渐变，表示时间）
        self.obstacle_cmap = LinearSegmentedColormap.from_list(
            'obstacle_path',
            ['#ffe0b2', '#ffcc80', '#ffa726', '#ff6b6b'],  # 浅橙到深红
            N=256
        )
        
        # 方向指示颜色映射（绿色到红色，表示不同方向）（暂时不用）
        self.direction_cmap = LinearSegmentedColormap.from_list(
            'direction_path',
            ['#00ff00', '#ffff00', '#ff0000'],  # 绿->黄->红
            N=256
        )


    def update(self):
        """更新数据，但不绘图"""
        current_time = time.time()
        
        # 获取机器人当前位置
        robot_pos = self.get_2d_position(self.robot_id)
        self.robot_positions.append(robot_pos)
        self.robot_timestamps.append(current_time)
        
        # 获取障碍物当前位置
        for obs_id in self.obstacles_ids:
            obs_pos = self.get_2d_position(obs_id)
            self.obstacle_positions[obs_id].append(obs_pos)
            self.obstacle_timestamps[obs_id].append(current_time)
    
    def get_2d_position(self, body_id):
        """获取2D位置"""
        pos, _ = p.getBasePositionAndOrientation(body_id)
        return [pos[0], pos[1]]  # X-Y平面
    
    def plot_gradient_trajectory(self, ax, positions, timestamps,
                                 cmap_name='robot_path', 
                                 linewidth_range=(TRAJECTORY_LINEWIDTH_MIN, TRAJECTORY_LINEWIDTH_MAX),
                                 alpha=TRAJECTORY_ALPHA, 
                                 zorder=TRAJECTORY_ZORDER, label="轨迹"):
        if len(positions) < TRAJECTORY_MIN_POINTS:
            return

        positions_array = np.array(positions)
        timestamps_array = np.array(timestamps)

        # 归一化时间戳 (0, 1)
        if len(timestamps_array) > 1:
            normalized_time = (timestamps_array - timestamps_array.min()) / (timestamps_array.max() - timestamps_array.min())
        else:
            normalized_time = np.zeros_like(timestamps_array)

        # 创建线段集合
        segments = []
        colors = []
        widths = []

        for i in range(len(positions_array) - 1):
            # 创建线段
            segment = np.array([positions_array[i], positions_array[i + 1]])
            segments.append(segment)

            # 使用中点的时间作为颜色参考
            mid_time = (normalized_time[i] + normalized_time[i + 1]) / 2

            # 选择颜色映射
            if cmap_name == 'robot_path':
                cmap = self.robot_cmap
                color = cmap(mid_time)
            elif cmap_name == 'obstacle_path':
                cmap = self.obstacle_cmap
                color = cmap(mid_time)
            elif cmap_name == 'direction_path':
                cmap = self.direction_cmap
                color = cmap(mid_time)
            else:
                color = 'red'
            
            colors.append(color)

            # 线宽渐变（时间越新，线越粗）
            width = linewidth_range[0] + (linewidth_range[1] - linewidth_range[0]) * mid_time
            widths.append(width)
        


        # 创建LineCollection
        lc = LineCollection(segments, colors=colors, linewidths=widths,
                           alpha=alpha, zorder=zorder, capstyle='round')
        
        ax.add_collection(lc)
        
        # 创建图例代理
        from matplotlib.lines import Line2D
        proxy = Line2D([0], [0], color=colors[-1] if colors else 'red', 
                      linewidth=linewidth_range[1], alpha=alpha)
        
        return proxy, label

    def plot_direction_arrows(self, ax, positions, timestamps, color=ARROW_COLOR, alpha=ARROW_ALPHA):
        """绘制方向箭头（表示运动方向）"""
        if len(positions) < ARROW_MIN_POINTS:
            return
        
        positions_array = np.array(positions)
        
        # 增大箭头绘制间隔，确保稀疏但每个线段都有
        # 使用全局配置参数
        arrow_step = max(1, min(ARROW_STEP_MIN, len(positions_array) // ARROW_STEP_RATIO))
        
        # 使用统一的中间色
        arrow_color = color
        
        for i in range(0, len(positions_array) - 1, arrow_step):
            if i + 1 < len(positions_array):
                # 计算方向
                dx = positions_array[i+1][0] - positions_array[i][0]
                dy = positions_array[i+1][1] - positions_array[i][1]
                
                if abs(dx) > 0.01 or abs(dy) > 0.01:  # 避免零向量
                    # 绘制箭头（使用全局配置参数）
                    ax.arrow(positions_array[i][0], positions_array[i][1],
                            dx, dy, 
                            head_width=ARROW_HEAD_WIDTH, 
                            head_length=ARROW_HEAD_LENGTH,
                            fc=arrow_color, ec=arrow_color, 
                            alpha=alpha,
                            length_includes_head=True, 
                            zorder=ARROW_ZORDER)
    
    def plot_time_gradient_circles(self, ax, positions, timestamps, 
                                 color='red', 
                                 alpha_range=(STAY_CIRCLE_ALPHA_MIN, STAY_CIRCLE_ALPHA_MAX)):
        """绘制时间渐变的圆圈（表示停留时间）"""
        if len(positions) < STAY_MIN_POINTS:
            return
        
        positions_array = np.array(positions)
        
        # 计算每个点的时间密度（停留时间）
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            avg_time_diff = np.mean(time_diffs)
            
            # 找到长时间停留的点
            for i in range(len(time_diffs)):
                if time_diffs[i] > avg_time_diff * STAY_TIME_THRESHOLD:  # 停留时间超过平均值的2倍
                    # 根据停留时间计算圆圈大小和透明度
                    size_factor = min(time_diffs[i] / avg_time_diff, STAY_SIZE_MAX)
                    radius = 0.1 * size_factor
                    alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * (1 - 1/size_factor)
                    
                    circle = plt.Circle(positions_array[i], radius, 
                                       color=color, alpha=alpha, zorder=STAY_CIRCLE_ZORDER,
                                       fill=True, linewidth=0)
                    ax.add_patch(circle)
    


    def save_trajectory_plot(self, filename="trajectory_plot.png", dpi=300, output_dir="./results", prisms_data=None):
        """保存增强版轨迹图"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 设置背景色
        ax.set_facecolor('#f8f9fa')
        
        # 1. 绘制机器人渐变轨迹
        if len(self.robot_positions) > 1:
            robot_proxy, robot_label = self.plot_gradient_trajectory(
                ax, self.robot_positions, self.robot_timestamps,
                cmap_name='robot_path', linewidth_range=(TRAJ_ROBOT_LINEWIDTH_MIN, TRAJ_ROBOT_LINEWIDTH_MAX),
                alpha=TRAJ_ROBOT_ALPHA, zorder=TRAJ_ROBOT_ZORDER, label="机器人轨迹（时间渐变）"
            )
            
            # 绘制机器人方向箭头
            self.plot_direction_arrows(ax, self.robot_positions, 
                                      self.robot_timestamps, color=ARROW_ROBOT_COLOR, alpha=ARROW_ROBOT_ALPHA)
        
        # 2. 绘制机器人起点和终点（特殊标记）
        if self.robot_positions:
            start_pos = self.robot_positions[0]
            end_pos = self.robot_positions[-1]
            
            # 起点（绿色大圆）
            ax.plot(start_pos[0], start_pos[1], 'o', markersize=ROBOT_POINT_MARKERSIZE,
                   color=ROBOT_START_COLOR, markeredgecolor=ICON_EDGE_COLOR, 
                   markeredgewidth=ICON_EDGE_WIDTH, zorder=ROBOT_POINT_ZORDER, label='起点')
            ax.text(start_pos[0]+ DATA_LABEL_OFFSET_X, start_pos[1] + DATA_LABEL_OFFSET_Y, '起点', 
                   fontsize=DATA_LABEL_FONTSIZE, ha='center', va='top', fontweight='bold', zorder = DATA_LABEL_ZORDER,
                   bbox=dict(boxstyle=DATA_LABEL_BOXSTYLE, pad=DATA_LABEL_BOXPAD, facecolor=DATA_LABEL_BOXCOLOR, alpha=DATA_LABEL_BOXALPHA))
            
            # 终点（红色大圆）
            ax.plot(end_pos[0], end_pos[1], 'o', markersize=ROBOT_POINT_MARKERSIZE,
                   color=ROBOT_END_COLOR, markeredgecolor=ICON_EDGE_COLOR,
                   markeredgewidth=ICON_EDGE_WIDTH, zorder=ROBOT_POINT_ZORDER, label='终点')
            ax.text(end_pos[0]+ DATA_LABEL_OFFSET_X, end_pos[1] + DATA_LABEL_OFFSET_Y, '终点',
                   fontsize=DATA_LABEL_FONTSIZE, ha='center', va='top', fontweight='bold', zorder = DATA_LABEL_ZORDER,
                   bbox=dict(boxstyle=DATA_LABEL_BOXSTYLE, pad=DATA_LABEL_BOXPAD, facecolor=DATA_LABEL_BOXCOLOR, alpha=DATA_LABEL_BOXALPHA))
        
        # 3. 绘制障碍物渐变轨迹
        obstacle_proxies = []
        for i, (prism_info, obs_id) in enumerate(prisms_data, self.obstacles_ids):
            if len(self.obstacle_positions[obs_id]) > 1:
                # 计算障碍物运动方向（用于选择颜色映射）
                positions = self.obstacle_positions[obs_id]
                timestamps = self.obstacle_timestamps[obs_id]
                
                # 单向运动使用时间颜色映射
                proxy, label = self.plot_gradient_trajectory(
                    ax, positions, timestamps,
                    cmap_name='obstacle_path', linewidth_range=(TRAJ_OBS_LINEWIDTH_MIN, TRAJ_OBS_LINEWIDTH_MAX),
                    alpha=TRAJ_OBS_ALPHA, zorder=TRAJ_OBS_ZORDER, 
                    label=f"障碍物{i+1}轨迹（时间渐变）"
                )

                obstacle_proxies.append((proxy, label))
                
                # 绘制障碍物方向箭头
                self.plot_direction_arrows(ax, positions, timestamps, 
                                          color=ARROW_OBS_COLOR, alpha=ARROW_OBS_ALPHA)
                
                # 绘制时间渐变圆圈（显示停留时间）
                self.plot_time_gradient_circles(ax, positions, timestamps,
                                              color='orange', alpha_range=(0.1, 0.5))
                
                # 绘制障碍物当前位置
                if positions:
                    current_pos = positions[-1]
                    base_points = prism_info['base_points']
                    angle_z = obs_id.get('z_angle', 0)
                    
                    
                    
                    ax.plot(current_pos[0], current_pos[1], 's', markersize=OBS_POINT_MARKERSIZE,
                           color=OBS_POINT_COLOR, markeredgecolor=ICON_EDGE_COLOR,
                           markeredgewidth=ICON_EDGE_WIDTH, zorder=OBS_POINT_ZORDER, 
                           label=f'障碍物{i+1}当前位置' if i == 0 else "")
        
        # 4. 绘制目标点
        target_points = [
            [2, 2], [8, 2], [8, 8], [2, 8]
        ]
        
        for i, target in enumerate(target_points):
            # 绘制目标点（紫色星星）
            ax.plot(target[0], target[1], '*', markersize=TARGET_POINT_MARKERSIZE,
                   color=TARGET_POINT_COLOR, markeredgecolor=ICON_EDGE_COLOR,
                   markeredgewidth=ICON_EDGE_WIDTH, zorder=TARGET_POINT_ZORDER,
                   label=f'目标点{i+1}' if i == 0 else "")
            
            # 添加目标点标签
            ax.text(target[0] + DATA_LABEL_OFFSET_X, target[1] + DATA_LABEL_OFFSET_Y, f'目标{i+1}',
                   fontsize=DATA_LABEL_FONTSIZE, ha='center', va='bottom', fontweight='bold', zorder = DATA_LABEL_ZORDER,
                   bbox=dict(boxstyle=DATA_LABEL_BOXSTYLE ,pad=DATA_LABEL_BOXPAD, facecolor=DATA_LABEL_BOXCOLOR, 
                            alpha=DATA_LABEL_BOXALPHA))
        
        # 6. 设置图形属性
        ax.set_xlabel(FIG_X_LABEL_CONTENT, fontsize=FIG_XY_FONTSIZE, fontweight='bold')
        ax.set_ylabel(FIG_Y_LABEL_CONTENT, fontsize=FIG_XY_FONTSIZE, fontweight='bold')
        ax.set_title(FIG_TITLE_CONTENT, 
                    fontsize=FIG_TITLE_FONTSIZE, fontweight='bold', pad=FIG_TITLE_PAD)
        
        # ax.grid(GRID_VISIBLE, alpha=GRID_LINE_ALPHA, linestyle=GRID_LINE_STYLE, linewidth=GRID_LINE_WIDTH)
        ax.set_xlim(FIG_XY_LIM_MIN, FIG_XY_LIM_MAX)
        ax.set_ylim(FIG_XY_LIM_MIN, FIG_XY_LIM_MAX)
        ax.set_xticks(range(FIG_XY_TICKS_MIN, FIG_XY_TICKS_MAX, FIG_XY_TICK))
        ax.set_yticks(range(FIG_XY_TICKS_MIN, FIG_XY_TICKS_MAX, FIG_XY_TICK))
        
        # 添加坐标网格
        ax.grid(GRID_MAJOR_VISIBLE, which='major', alpha=GRID_MAJOR_LINE_ALPHA, linestyle=GRID_MAJOR_LINE_STYLE, linewidth=GRID_MAJOR_LINE_WIDTH)
        ax.grid(GRID_MINOR_VISIBLE, which='minor', alpha=GRID_MINOR_LINE_ALPHA, linestyle=GRID_MINOR_LINE_STYLE, linewidth=GRID_MINOR_LINE_WIDTH)
        ax.minorticks_on()
        
        # 7. 创建图例
        legend_handles = []
        legend_labels = []
        
        # 添加机器人轨迹图例
        if len(self.robot_positions) > 1:
            from matplotlib.lines import Line2D
            robot_legend = Line2D([0], [0], color=self.robot_cmap(0.7), 
                                linewidth=4, alpha=0.9)
            legend_handles.append(robot_legend)
            legend_labels.append("机器人轨迹（细→粗：时间推进）")
        
        # 添加起点终点图例
        start_legend = Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='#2ecc71', markersize=10,
                             markeredgecolor='black', markeredgewidth=2)
        legend_handles.append(start_legend)
        legend_labels.append("起点")
        
        end_legend = Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='#e74c3c', markersize=10,
                           markeredgecolor='black', markeredgewidth=2)
        legend_handles.append(end_legend)
        legend_labels.append("终点")
        
        # 添加障碍物图例
        if obstacle_proxies:
            obs_legend = Line2D([0], [0], color=self.obstacle_cmap(0.5),
                              linewidth=3, alpha=0.7)
            legend_handles.append(obs_legend)
            legend_labels.append("障碍物轨迹（浅→深：时间推进）")
            
            dir_legend = Line2D([0], [0], color=self.direction_cmap(0.5),
                              linewidth=3, alpha=0.7)
            legend_handles.append(dir_legend)
            legend_labels.append("往返运动（绿→红：方向变化）")
        
        # 添加目标点图例
        target_legend = Line2D([0], [0], marker='*', color='w',
                             markerfacecolor='#9b59b6', markersize=15,
                             markeredgecolor='black', markeredgewidth=1)
        legend_handles.append(target_legend)
        legend_labels.append("目标点")
        
        # 添加箭头图例
        arrow_legend = Line2D([0], [0], marker='>', color='w',
                            markerfacecolor='blue', markersize=10,
                            markeredgecolor='blue', markeredgewidth=1)
        legend_handles.append(arrow_legend)
        legend_labels.append("运动方向")
        
        # 绘制图例
        ax.legend(legend_handles, legend_labels, loc='upper left',
                 fontsize=11, framealpha=0.95, shadow=True,
                 fancybox=True, borderpad=1)
        
        ax.set_aspect('equal')
        
#         # 8. 添加信息文本框
#         info_text = f"""
# 轨迹信息:
# • 机器人轨迹点数: {len(self.robot_positions)}
# • 轨迹总时长: {self.robot_timestamps[-1] - self.robot_timestamps[0]:.1f}s
# • 障碍物数量: {len(self.obstacles_ids)}
# • 可视化说明:
#   线宽渐变: 细→粗表示时间推进
#   颜色渐变: 浅→深表示时间推进
#   箭头方向: 表示瞬时运动方向
#   圆圈大小: 表示停留时间长短
# """
#         ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
#                fontsize=10, verticalalignment='top',
#                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
#                        alpha=0.9, edgecolor='gray'))
        
        # 保存图片
        plt.tight_layout()
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"✓ 增强版轨迹图已保存为: {filepath}")
        print(f"  时间渐变可视化已启用")



    def get_base_projection(self, base_points, angle_z=0, scale=1.0):
        """获取底面的2D投影（考虑旋转）"""
        # 转换为numpy数组
        points = np.array(base_points)
        
        # 应用Z轴旋转
        if angle_z != 0:
            rot_matrix = np.array([
                [math.cos(angle_z), -math.sin(angle_z)],
                [math.sin(angle_z), math.cos(angle_z)]
            ])
            points = np.dot(points, rot_matrix.T)
        
        # 应用缩放
        points = points * scale
        
        return points
    

    def create_projection_patch(self, base_points, position, angle_z=0, 
                               color=None, alpha=0.7, scale=1.0):
        """创建投影的多边形补丁"""
        # 获取旋转后的底面投影
        projected_points = self.get_base_projection(base_points, angle_z, scale)
        
        # 应用位置偏移
        center = np.array([position[0], position[1]])
        translated_points = projected_points + center
        
        # 创建多边形补丁
        if color is None:
            color = np.random.rand(3)
        
        polygon = Polygon(
            translated_points,
            closed=True,
            edgecolor=color,
            facecolor=color + (1 - np.array(color)) * 0.3,  # 使填充色稍浅
            alpha=alpha,
            linewidth=2
        )
        
        return polygon

class SimulationEnvironment:
    def __init__(self, gui=True, gravity=9.8):
        """"初始化仿真环境"""
        # 连接物理引擎
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # 设置参数
        p.setGravity(0, 0, gravity)
        p.setTimeStep(1/240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)

        # 设置摄像头
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # 创建机器人（使用立方体代替机器人，确保显示正确）
        robot_start_pos = [1, 1, 0.5]
        # 创建简单的立方体作为机器人
        robot_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
        self.robot_id = p.createMultiBody(baseMass=1.0,
                                    baseCollisionShapeIndex=robot_shape,
                                    basePosition=robot_start_pos)
        
        # 设置机器人颜色为蓝色
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[0, 0, 1, 1])


        planeId = p.loadURDF("plane.urdf")
        
        self.prisms = []

    def load_prism(self, urdf_file, position=[0, 0, 0],
                orientation=[0, 0, 0, 1], scale=1.0):
        """"加载棱柱模型"""
        urdf_path = urdf_file

        prism_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scale,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        self.prisms.append({
            'id': prism_id,
            'urdf_file': urdf_file,
            'position': position,
            'orientation': orientation
        })

        return prism_id

    def random_place_prisms(self, prisms_info, num_prisms=None,
                            limits_x=[-5, 5], limits_y=[-5, 5]):
        """随机放置模型"""
        if num_prisms is None:
            num_prisms = len(prisms_info)

        placed_prisms = []
        robot_id = self.robot_id

        for i in range(min(num_prisms, len(prisms_info))):
            info = prisms_info[i]

            # 随机分布位置，后采用泊松分布确保不重叠
            max_attempts = 100
            placed_completed = False

            for attempt in range(max_attempts):
                # 随机位置
                x = random.uniform(limits_x[0], limits_x[1])
                y = random.uniform(limits_y[0], limits_y[1])

                # 随机绕z轴旋转
                angle = random.uniform(0, 2 * math.pi)
                orientation = p.getQuaternionFromEuler([0, 0, angle])

                # 高度
                height = info.get('height', 1.0)
                position = [x, y, height / 2.0+ 0.01]

                # 检查是否有重叠
                overlap = False
                for prism in placed_prisms:
                    dx = prism['position'][0] - x
                    dy = prism['position'][1] - y
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < 2.0:  # 安全距离
                        overlap = True
                        break
                
                if not overlap:
                    # 加载模型
                    prism_id = self.load_prism(
                        info['urdf_file'],
                        position=position,
                        orientation=orientation
                    )

                    placed_prisms.append({
                        'id': prism_id,
                        'info':info,
                        'position': position,
                        'orientation': orientation,
                        'angle_z': angle
                    })


                    placed_completed = True
                    break

            if not placed_completed:
                print(f"⚠️ 未能成功放置模型: {info['name']}，请检查空间限制。")

        return placed_prisms, robot_id
    
    def movement_setup(self, placed_prisms):
        """设置动态障碍物以及运动参数"""
        dynamic_obstacles_info = []
        straight_moving_ids = [3,4]
        circle_moving_ids = [5,6]
        polyline_moving_ids = [7,8]

        # 障碍物1: 简单直线往返
        for ids in  straight_moving_ids:
            obstacle_start = placed_prisms[ids]['position']
            obstacle_end = [obstacle_start[0], obstacle_start[1] + 3, obstacle_start[2]]    # 暂时

            dynamic_obstacles_info.append({
                'id': placed_prisms[ids]['id'],
                'start_pos': obstacle_start,
                'end_pos': obstacle_end,
                'direction': 1,
                'speed': 0.02
            })            

        # 障碍物2: 圆形运动
        for ids in circle_moving_ids:
            obstacle_center = placed_prisms[ids]['position']
            obstacle_radius = 2.0
            dynamic_obstacles_info.append({
                'id': placed_prisms[ids]['id'],
                'type': 'circular',
                'center': obstacle_center,
                'radius': obstacle_radius,
                'angle': 0,
                'speed': 0.02
            })

        # 障碍物3: 复杂折线运动
        for ids in polyline_moving_ids:
            obstacle_waypoints = [
                [placed_prisms[ids]['position'][0], placed_prisms[ids]['position'][1], placed_prisms[ids]['position'][2]],
                [placed_prisms[ids]['position'][0] + 3, placed_prisms[ids]['position'][1], placed_prisms[ids]['position'][2]],
                [placed_prisms[ids]['position'][0] + 3, placed_prisms[ids]['position'][1] + 3, placed_prisms[ids]['position'][2]],
                [placed_prisms[ids]['position'][0], placed_prisms[ids]['position'][1], placed_prisms[ids]['position'][2]]
            ]
            dynamic_obstacles_info.append({
                'id': placed_prisms[ids]['id'],
                'type': 'waypoints',
                'waypoints': obstacle_waypoints,
                'current_waypoint': 0,
                'speed': 0.02
            })

        return dynamic_obstacles_info


def create_simulation_environment():
    """创建仿真环境"""
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 设置物理参数
    p.setPhysicsEngineParameter(fixedTimeStep=1./240.)

    # 加载地面
    planeId = p.loadURDF("plane.urdf")
    
    # 创建机器人（使用立方体代替赛车，确保显示正确）
    robot_start_pos = [1, 1, 0.5]
    # 创建简单的立方体作为机器人
    robot_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
    robot_id = p.createMultiBody(baseMass=1.0,
                                baseCollisionShapeIndex=robot_shape,
                                basePosition=robot_start_pos)
    
    # 设置机器人颜色为蓝色
    p.changeVisualShape(robot_id, -1, rgbaColor=[0, 0, 1, 1])
    
    # 创建障碍物
    obstacles = []
    dynamic_obstacles_info = []
    
    # 静态障碍物（球体）
    static_obstacle_positions = [
        [3, 3, 1.5],
        [7, 3, 1.5],
        [5, 5, 1.5],
        [3, 7, 1.5],
        [7, 7, 1.5]
    ]
    
    for pos in static_obstacle_positions:
        obstacle_id = p.loadURDF("sphere2.urdf", pos, globalScaling=0.5)
        p.changeVisualShape(obstacle_id, -1, rgbaColor=[1, 0, 0, 1])  # 红色
        obstacles.append(obstacle_id)
    

    # 动态障碍物 - 创建不同类型的运动
    # 障碍物1: 简单直线往返
    obstacle1_start = [2, 5, 0.5]
    obstacle1_end = [2, 8, 0.5]
    
    obstacle_id1 = p.loadURDF("cube.urdf", obstacle1_start, globalScaling=0.5)
    p.changeVisualShape(obstacle_id1, -1, rgbaColor=[1, 0.5, 0, 1])
    obstacles.append(obstacle_id1)
    dynamic_obstacles_info.append({
        'id': obstacle_id1,
        'start_pos': obstacle1_start,
        'end_pos': obstacle1_end,
        'direction': 1,
        'speed': 0.02
    })
    
    # 障碍物2: 圆形运动
    obstacle2_center = [5, 5, 0.5]
    obstacle2_radius = 2.0
    obstacle_id2 = p.loadURDF("cube.urdf", [obstacle2_center[0] + obstacle2_radius, 
                                          obstacle2_center[1], 0.5], 
                             globalScaling=0.5)
    p.changeVisualShape(obstacle_id2, -1, rgbaColor=[0.5, 1, 0, 1])
    obstacles.append(obstacle_id2)
    dynamic_obstacles_info.append({
        'id': obstacle_id2,
        'type': 'circular',
        'center': obstacle2_center,
        'radius': obstacle2_radius,
        'angle': 0,
        'speed': 0.02
    })
    
    # 障碍物3: 复杂折线运动
    obstacle3_waypoints = [
        [8, 2, 0.5],
        [8, 8, 0.5],
        [6, 6, 0.5],
        [8, 2, 0.5]
    ]
    obstacle_id3 = p.loadURDF("cube.urdf", obstacle3_waypoints[0], globalScaling=0.5)
    p.changeVisualShape(obstacle_id3, -1, rgbaColor=[1, 0, 0.5, 1])
    obstacles.append(obstacle_id3)
    dynamic_obstacles_info.append({
        'id': obstacle_id3,
        'type': 'waypoints',
        'waypoints': obstacle3_waypoints,
        'current_waypoint': 0,
        'speed': 0.02
    })
    
    # 稳定化步骤
    for _ in range(10):
        p.stepSimulation()
        time.sleep(0.01)
    
    
    return robot_id, obstacles, dynamic_obstacles_info





def main():
    """主函数"""
    # 读取模型信息
    reader = URDFReader()
    prisms_info = reader.read_prisms_from_info_file()
    # 创建仿真环境
    env = SimulationEnvironment(gui=True, gravity=9.8)
    placed_prisms, robot_id = env.random_place_prisms(prisms_info=prisms_info, num_prisms=10,limits_x=[-10,10], limits_y=[-10,10])
    dynamic_obs_info = env.movement_setup(placed_prisms)

    # 创建仿真环境
    # robot_id, obstacles, dynamic_obstacles_info = create_simulation_environment()
    
    # 创建可视化器
    visualizer = NavigationVisualizer(robot_id, placed_prisms, dynamic_obs_info)
    
    # 设置目标点序列
    target_points = [
        [2, 2],      # 第一个目标点
        [8, 2],      # 第二个目标点  
        [8, 8],      # 第三个目标点
        [2, 8]       # 第四个目标点
    ]
    
    # 创建控制器
    robot_controller = SimpleRobotController(robot_id, target_points, kp=0.8, max_velocity=1.2)
    
    print("\n开始导航...")
    print("目标点序列:", target_points)
    print("动态障碍物包含: 直线往返、圆形运动、折线运动\n")
    
    # 添加键盘控制标记
    key_stop = False
    
    # 主仿真循环
    try:
        step_count = 0
        max_steps = 500  # 增加最大步数
        
        while step_count < max_steps and not key_stop:
            p.stepSimulation()
            step_count += 1
            
            # 更新可视化数据
            visualizer.update()
            
            # 控制机器人移动
            velocity_cmd = robot_controller.compute_velocity()
            
            if velocity_cmd:
                # 改进的速度控制 - 使用直接位置控制，避免物理碰撞问题
                current_pos, current_orn = p.getBasePositionAndOrientation(robot_id)
                
                # 获取当前目标点
                target = robot_controller.target_points[robot_controller.current_target_index]
                
                # 计算移动方向
                dx = target[0] - current_pos[0]
                dy = target[1] - current_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0.1:  # 如果距离目标较远
                    # 计算移动速度
                    speed = min(velocity_cmd[0] * 5, 2.5)  # 限制最大速度
                    
                    # 计算新位置（直接设置位置，避免碰撞）
                    step_size = speed * (1./240.)  # 基于时间步长
                    new_x = current_pos[0] + dx/distance * step_size
                    new_y = current_pos[1] + dy/distance * step_size
                    
                    # 保持Z坐标不变，避免掉下去
                    new_z = 0.5  # 固定高度
                    
                    # 直接设置机器人位置（跳过物理引擎）
                    p.resetBasePositionAndOrientation(robot_id, [new_x, new_y, new_z], current_orn)

            else:
                # 已到达所有目标点
                print("✓ 已完成所有目标点导航！")
                break
            
            # 动态障碍物控制
            for obs_info in dynamic_obs_info:
                obs_id = obs_info['id']
                current_pos, current_orn = p.getBasePositionAndOrientation(obs_id)
                
                if obs_info.get('type') == 'circular':
                    # 圆形运动
                    angle = obs_info.get('angle', 0) + obs_info['speed']
                    obs_info['angle'] = angle
                    
                    x = obs_info['center'][0] + obs_info['radius'] * math.cos(angle)
                    y = obs_info['center'][1] + obs_info['radius'] * math.sin(angle)
                    
                    p.resetBasePositionAndOrientation(obs_id, [x, y, 0.5], current_orn)
                    
                elif obs_info.get('type') == 'waypoints':
                    # 折线运动
                    waypoints = obs_info['waypoints']
                    current_idx = obs_info['current_waypoint']
                    target_wp = waypoints[current_idx]
                    
                    dx = target_wp[0] - current_pos[0]
                    dy = target_wp[1] - current_pos[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < 0.1:
                        obs_info['current_waypoint'] = (current_idx + 1) % len(waypoints)
                    else:
                        direction_x = dx / distance if distance > 0 else 0
                        direction_y = dy / distance if distance > 0 else 0
                        
                        new_x = current_pos[0] + direction_x * obs_info['speed']
                        new_y = current_pos[1] + direction_y * obs_info['speed']
                        
                        p.resetBasePositionAndOrientation(obs_id, [new_x, new_y, 0.5], current_orn)
                        
                else:
                    # 直线往返运动
                    if obs_info['direction'] == 1:
                        target_pos = obs_info['end_pos']
                    else:
                        target_pos = obs_info['start_pos']
                    
                    dx = target_pos[0] - current_pos[0]
                    dy = target_pos[1] - current_pos[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < 0.1:
                        obs_info['direction'] *= -1
                    else:
                        direction_x = dx / distance if distance > 0 else 0
                        direction_y = dy / distance if distance > 0 else 0
                        
                        new_x = current_pos[0] + direction_x * obs_info['speed']
                        new_y = current_pos[1] + direction_y * obs_info['speed']
                        
                        p.resetBasePositionAndOrientation(obs_id, [new_x, new_y, 0.5], current_orn)
            
            # 短暂暂停，让仿真可见
            time.sleep(1./60.)
            
            # 检查键盘输入
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("用户按下Q键，停止仿真")
                key_stop = True
        
        # 保存轨迹图
        print("\n正在生成轨迹图...")
        visualizer.save_trajectory_plot("论文用图.png", dpi=600, output_dir="paper/Pic")
        
    except KeyboardInterrupt:
        print("仿真被用户中断")
    except Exception as e:
        print(f"仿真出错: {e}")
    finally:
        p.disconnect()
        plt.close('all')
        print("仿真结束")

if __name__ == "__main__":
    main()