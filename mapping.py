import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation

class PyBullet2DVisualizer:
    def __init__(self, robot_id, obstacles_ids):
        self.robot_id = robot_id
        self.obstacles_ids = obstacles_ids  # 包含静态和动态障碍物ID
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # 存储轨迹历史
        self.robot_positions = []
        self.obstacle_trajectories = {oid: [] for oid in obstacles_ids}
        
        # 颜色映射
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(obstacles_ids)+2))
        
    def get_2d_position(self, body_id):
        """从3D获取2D投影（X-Z平面或X-Y平面）"""
        pos, _ = p.getBasePositionAndOrientation(body_id)
        return [pos[0], pos[2]]  # X-Z平面，适合地面机器人
        # return [pos[0], pos[1]]  # X-Y平面
        
    def update_plot(self, frame):
        self.ax.clear()
        
        # 获取当前位置
        robot_pos = self.get_2d_position(self.robot_id)
        self.robot_positions.append(robot_pos)
        
        # 绘制机器人轨迹
        if len(self.robot_positions) > 1:
            path = np.array(self.robot_positions)
            self.ax.plot(path[:, 0], path[:, 1], 
                        'b-', linewidth=2, label='Robot Path', alpha=0.7)
        
        # 绘制机器人当前位置
        self.ax.plot(robot_pos[0], robot_pos[1], 
                    'bo', markersize=12, label='Robot')
        
        # 绘制障碍物
        for i, obs_id in enumerate(self.obstacles_ids):
            obs_pos = self.get_2d_position(obs_id)
            self.obstacle_trajectories[obs_id].append(obs_pos)
            
            # 获取障碍物类型（静态/动态）
            obs_info = p.getBodyInfo(obs_id)
            is_dynamic = "dynamic" in obs_info[1].decode('utf-8').lower()
            
            # 绘制障碍物轨迹
            if len(self.obstacle_trajectories[obs_id]) > 1:
                obs_path = np.array(self.obstacle_trajectories[obs_id])
                line_style = '--' if is_dynamic else ':'
                color = self.colors[i+1]
                self.ax.plot(obs_path[:, 0], obs_path[:, 1], 
                           line_style, color=color, alpha=0.5,
                           label=f'Obstacle {i} Path' if is_dynamic else f'Static {i}')
            
            # 绘制障碍物当前位置
            marker = 's' if is_dynamic else 'o'
            color = 'red' if is_dynamic else 'gray'
            self.ax.plot(obs_pos[0], obs_pos[1], 
                        marker=marker, color=color, markersize=15)
            
            # 添加障碍物半径（如果是圆形）
            try:
                # 获取碰撞形状信息
                visual_data = p.getVisualShapeData(obs_id)
                if visual_data and len(visual_data[0]) > 3:
                    dimensions = visual_data[0][3]
                    if len(dimensions) >= 2:
                        radius = dimensions[0]
                        circle = plt.Circle(obs_pos, radius, 
                                          color=color, alpha=0.2)
                        self.ax.add_patch(circle)
            except:
                pass
        
        # 设置图形属性
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)')
        self.ax.set_title('PyBullet Simulation - 2D View')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left', fontsize='small')
        self.ax.set_aspect('equal')
        
        # 自动调整视野
        if self.robot_positions:
            all_points = np.array(self.robot_positions)
            for traj in self.obstacle_trajectories.values():
                if traj:
                    all_points = np.vstack([all_points, np.array(traj[-20:])])  # 最近20个点
            if len(all_points) > 1:
                x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
                y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
                margin = max((x_max-x_min), (y_max-y_min)) * 0.2
                self.ax.set_xlim(x_min-margin, x_max+margin)
                self.ax.set_ylim(y_min-margin, y_max+margin)
    
    def animate(self):
        """实时动画"""
        anim = FuncAnimation(self.fig, self.update_plot, 
                           interval=50, cache_frame_data=False)
        plt.show()

# 使用示例
def setup_simulation():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 加载地面
    planeId = p.loadURDF("plane.urdf")
    
    # 加载机器人（例如TurtleBot）
    robot_start_pos = [0, 0, 0.1]
    robot_id = p.loadURDF("racecar/racecar.urdf", robot_start_pos)
    
    # 创建障碍物
    obstacles = []
    
    # 静态障碍物
    static_obs = p.loadURDF("sphere_small.urdf", [2, 0, 0.5])
    obstacles.append(static_obs)
    
    # 动态障碍物（移动）
    dynamic_obs = p.loadURDF("cube_small.urdf", [-2, 0, 0.5])
    obstacles.append(dynamic_obs)
    
    return robot_id, obstacles

# 主程序
if __name__ == "__main__":
    robot_id, obstacles = setup_simulation()
    visualizer = PyBullet2DVisualizer(robot_id, obstacles)
    
    # 在另一个线程中运行动画
    import threading
    anim_thread = threading.Thread(target=visualizer.animate)
    anim_thread.start()
    
    # 仿真循环
    for i in range(10000):
        p.stepSimulation()
        
        # 移动动态障碍物
        pos = np.array([-2 + 0.05*i % 4, 0, 0.5])
        p.resetBasePositionAndOrientation(obstacles[1], pos, [0,0,0,1])
        
        time.sleep(1./240.)
    
    p.disconnect()