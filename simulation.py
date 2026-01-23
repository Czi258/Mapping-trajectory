import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import os

class PrismSimulation:
    def __init__(self, gui=True, gravity=-9.8):
        """初始化仿真环境"""
        # 连接物理引擎
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        # 设置参数
        p.setGravity(0, 0, gravity)
        p.setTimeStep(1/240.0)
        p.setRealTimeSimulation(0)
        
        # 设置摄像头
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # 添加资源路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.prisms = []
        self.ground = None
    
    def load_ground(self):
        """加载地面"""
        self.ground = p.loadURDF("plane.urdf")
        return self.ground
    
    def load_prism(self, urdf_file, position=[0, 0, 0], 
                  orientation=[0, 0, 0, 1], scale=1.0):
        """加载棱柱模型"""
        # 确保URDF文件路径正确
        urdf_path = os.path.abspath(urdf_file)
        
        # 加载模型
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
    
    def random_place_prisms(self, prisms_info, num_prisms=None):
        """随机放置多个棱柱"""
        if num_prisms is None:
            num_prisms = len(prisms_info)
        
        placed_prisms = []
        
        for i in range(min(num_prisms, len(prisms_info))):
            info = prisms_info[i]
            
            # 随机位置（确保不重叠）
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                # 随机位置
                x = random.uniform(-2, 2)
                y = random.uniform(-2, 2)
                
                # 随机绕Z轴旋转
                angle = random.uniform(0, 2 * np.pi)
                orientation = p.getQuaternionFromEuler([0, 0, angle])
                
                # 高度（确保底面在地面上）
                height = info.get('height', 1.0)
                position = [x, y, height/2 + 0.01]  # 稍微抬高避免穿透
                
                # 检查是否与其他物体重叠（简单检查）
                overlap = False
                for prism in placed_prisms:
                    dx = prism['position'][0] - x
                    dy = prism['position'][1] - y
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < 1.0:  # 安全距离
                        overlap = True
                        break
                
                if not overlap:
                    # 加载棱柱
                    prism_id = self.load_prism(
                        info['urdf_file'],
                        position=position,
                        orientation=orientation
                    )
                    
                    placed_prisms.append({
                        'id': prism_id,
                        'info': info,
                        'position': position,
                        'orientation': orientation,
                        'angle_z': angle
                    })
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place prism {info['name']} without overlap")
        
        return placed_prisms
    
    def get_prism_state(self, prism_id):
        """获取棱柱状态"""
        position, orientation = p.getBasePositionAndOrientation(prism_id)
        euler = p.getEulerFromQuaternion(orientation)
        
        return {
            'position': position,
            'orientation': orientation,
            'euler': euler,
            'z_angle': euler[2]  # Z轴旋转角度
        }
    
    def simulate(self, steps=1000):
        """运行仿真"""
        for i in range(steps):
            p.stepSimulation()
            time.sleep(1/240.0)
    
    def close(self):
        """关闭仿真"""
        p.disconnect()

if __name__ == "__main__":
    # 示例用法
    from urdf_reader import URDFReader
    
    # 读取棱柱信息
    reader = URDFReader()
    prisms_info = reader.read_prisms_from_info_file()
    
    # 创建仿真环境
    sim = PrismSimulation()
    sim.load_ground()
    
    # 随机放置棱柱
    placed = sim.random_place_prisms(prisms_info, num_prisms=3)
    
    # 运行仿真
    sim.simulate(steps=1000)
    
    # 获取状态
    for prism in placed:
        state = sim.get_prism_state(prism['id'])
        print(f"Prism {prism['info']['name']}:")
        print(f"  Position: {state['position']}")
        print(f"  Z-angle: {state['z_angle']:.3f} rad")
    
    # 保持窗口打开
    input("Press Enter to exit...")
    sim.close()