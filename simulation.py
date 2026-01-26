import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import os
import threading
import queue
from urdf_reader import URDFReader


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
        self.simulation_running = False
        self.simulation_thread = None
        self.key_events = queue.Queue()
    
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
                x = random.uniform(-5, 5)
                y = random.uniform(-5, 5)
                
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
                    if distance < 2.0:  # 安全距离
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

    def _keyboard_callback(self,key):
        """键盘回调函数"""
        self.key_events.put(key)
        print(f"Key pressed: {key}")

    def _setup_keyboard_listener(self):
        """设置键盘监听"""
        # 设置键盘回调
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS,1)

        print("\n" + "="*50)
        print("Simulation Control Keys:")
        print("  SPACE: Stop simulation")
        print("  'q'  : Quit simulation")
        print("  's'  : Get prism states")
        print("  'r'  : Reset view")
        print("="*50 + "\n")

        # GUI模式下持续监听
        if p.isConnected():
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def _check_keyboard_input(self):
        """"检查键盘输入"""
        try:
            # 从队列中获取按键事件（非阻塞）
            key = self.key_events.get_nowait()
            return key
        except queue.Empty:
            return None
    
    def simulate_with_keyboard_control(self, stop_key='space', quit_key='q'):
        """
        通过键盘控制仿真运行
        """
        self._setup_keyboard_listener()

        print(f"仿真开始，按 '{stop_key}' 停止，按 '{quit_key}' 退出。")

        self.simulation_running = True 
        step_count= 0

        try:
            while self.simulation_running:
                # 运行仿真step
                p.stepSimulation()
                time.sleep(1./240.)
                step_count += 1

                # 每隔x步显示状态
                if step_count % 240 == 0:
                    print(f"仿真已运行{step_count/240:.1f}秒,停止请按'{stop_key}'")

                # 处理GUI事件
                keys = p.getKeyboardEvents()
                if keys:
                    for key in keys:
                        if keys[key] & p.KEY_WAS_TRIGGERED:
                            if key == 32:   # 空格键
                                print("停止仿真。")
                                self.simulation_running = False
                                break
                            elif key == 113:    # 'q'键
                                print("退出仿真。")
                                self.simulation_running = False
                                return 'quit'
                            elif key == 115:    # 's'键
                                self._print_prism_states()
                            elif key == 114:    # 'r'键
                                p.resetDebugVisualizerCamera(
                                    cameraDistance=5,
                                    cameraYaw=45,
                                    cameraPitch=-30,
                                    cameraTargetPosition=[0, 0, 0]
                                )
                                print("视图已重置。")
                        
                # 检查键盘输入队列
                key_input = self._check_keyboard_input()
                if key_input:
                    if key_input.lower() == ' ' or key_input == 'space':
                        print("\n接收到空格键 - 停止仿真")
                        self.simulation_running = False
                        break
                    elif key_input.lower() == 'q':
                        print("\n接收到 'q' 键 - 退出仿真")
                        self.simulation_running = False
                        return 'quit'

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            self.simulation_running = False
        
        print(f"\nSimulation stopped after {step_count} steps")
        return step_count

    def simulate(self, steps=1000):
    #     """运行仿真"""
    #     for i in range(steps):
    #         p.stepSimulation()
    #         time.sleep(1/240.0)
        return self.simulate_with_keyboard_control()

    def _print_prism_states(self):
        """"打印所有棱柱的状态"""
        print("\n" + "-"*50)
        print("Current Prism States:")
        print("-"*50)
        for prism in self.prisms:
            state = self.get_prism_state(prism['id'])
            print(f"ID {prism['id']}:")
            print(f"  Position: [{state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f}]")
            print(f"  Z-angle: {state['z_angle']:.3f} rad ({state['z_angle']*180/np.pi:.1f}°)")
        print("-"*50)

    def close(self):
        """关闭仿真"""
        p.disconnect()

if __name__ == "__main__":    
    # 读取棱柱信息
    reader = URDFReader()
    prisms_info = reader.read_prisms_from_info_file()
    
    # 创建仿真环境
    sim = PrismSimulation()
    sim.load_ground()
    
    # 随机放置棱柱
    placed = sim.random_place_prisms(prisms_info, num_prisms=3)
    
    # 运行仿真
    result = sim.simulate(steps=1000)
    
    # 获取状态
    for prism in placed:
        state = sim.get_prism_state(prism['id'])
        print(f"Prism {prism['info']['name']}:")
        print(f"  Position: {state['position']}")
        print(f"  Z-angle: {state['z_angle']:.3f} rad")
    
    if result != 'quit':
        keep_on = True
        start_time = time.time()
        while keep_on and time.time() - start_time < 10:
            time.sleep(0.1)
            keys = p.getKeyboardEvents()
            if keys:
                for key in keys:
                    if keys[key] & p.KEY_WAS_TRIGGERED:
                        if key == 113:
                            keep_on = False
                                
    sim.close()