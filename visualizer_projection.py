import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
import math

class ProjectionVisualizer:
    def __init__(self):
        self.colors = cm.tab10.colors
    
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
    
    def visualize_simulation(self, prisms_data, sim_state, title="Simulation Visualization"):
        """可视化仿真环境中的棱柱投影"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        patches_list = []
        
        for i, (prism_info, state) in enumerate(zip(prisms_data, sim_state)):
            # 获取棱柱信息
            base_points = prism_info['base_points']
            position = state['position']
            angle_z = state.get('z_angle', 0)
            
            # 颜色
            color = self.colors[i % len(self.colors)]
            
            # 创建投影补丁
            polygon = self.create_projection_patch(
                base_points, position, angle_z, 
                color=color, alpha=0.6, scale=1.0
            )
            
            patches_list.append(polygon)
            
            # 添加标签
            center = np.mean(polygon.xy, axis=0)
            ax.text(
                center[0], center[1], 
                f"P{i+1}\nθ={angle_z:.2f}",
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white'
            )
        
        # 添加所有补丁到坐标轴
        collection = PatchCollection(patches_list, match_original=True)
        ax.add_collection(collection)
        
        # 设置坐标轴
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        
        # 添加图例
        legend_elements = []
        for i in range(len(patches_list)):
            legend_elements.append(
                patches.Patch(
                    color=self.colors[i % len(self.colors)],
                    alpha=0.6,
                    label=f'Prism {i+1}'
                )
            )
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig, ax
    
    def save_visualization(self, fig, filename="simulation_projection.png", dpi=150):
        """保存可视化图像"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
    
    def plot_individual_projections(self, prisms_data, sim_state):
        """绘制每个棱柱的单独投影"""
        n_prisms = len(prisms_data)
        n_cols = min(3, n_prisms)
        n_rows = math.ceil(n_prisms / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        if n_prisms == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for i, (prism_info, state) in enumerate(zip(prisms_data, sim_state)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            base_points = prism_info['base_points']
            position = state['position']
            angle_z = state.get('z_angle', 0)
            
            # 获取投影
            projected_points = self.get_base_projection(base_points, angle_z)
            
            # 绘制多边形
            polygon = Polygon(
                projected_points,
                closed=True,
                edgecolor=self.colors[i % len(self.colors)],
                facecolor=self.colors[i % len(self.colors)] + (1 - np.array(self.colors[i % len(self.colors)])) * 0.3,
                alpha=0.7,
                linewidth=2
            )
            
            ax.add_patch(polygon)
            
            # 设置坐标轴
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Prism {i+1}: θ={angle_z:.2f} rad\n{prism_info["name"]}')
            
            # 添加坐标轴
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        return fig, axes

if __name__ == "__main__":
    # 示例用法
    from urdf_reader import URDFReader
    
    # 读取棱柱信息
    reader = URDFReader()
    prisms_info = reader.read_prisms_from_info_file()
    
    # 创建模拟的状态数据
    sim_state = []
    for i, prism in enumerate(prisms_info):
        # 随机位置和角度
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        angle_z = np.random.uniform(0, 2*np.pi)
        
        sim_state.append({
            'position': [x, y, prism['height']/2],
            'z_angle': angle_z
        })
    
    # 创建可视化
    visualizer = ProjectionVisualizer()
    
    # 绘制整体仿真视图
    fig1, ax1 = visualizer.visualize_simulation(
        prisms_info, sim_state, 
        title="Quad Prism Projections in Simulation"
    )
    
    # 绘制单独投影
    fig2, axes2 = visualizer.plot_individual_projections(prisms_info, sim_state)
    
    # 保存图像
    visualizer.save_visualization(fig1, "simulation_overview.png")
    visualizer.save_visualization(fig2, "individual_projections.png")
    
    plt.show()