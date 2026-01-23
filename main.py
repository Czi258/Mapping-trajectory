#!/usr/bin/env python3
"""
主程序：生成、仿真和可视化四边棱柱
"""

import os
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Quad Prism Simulation System')
    parser.add_argument('--num-prisms', type=int, default=3, 
                       help='Number of prisms to generate (3-4)')
    parser.add_argument('--no-gui', action='store_true', 
                       help='Run PyBullet without GUI')
    parser.add_argument('--sim-steps', type=int, default=500,
                       help='Number of simulation steps')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Quad Prism Simulation System")
    print("=" * 60)
    
    # 步骤1: 生成棱柱URDF文件
    print("\nStep 1: Generating quad prisms...")
    from quad_prism_generator import QuadPrismGenerator
    
    generator = QuadPrismGenerator(urdf_dir="urdf")
    prisms_info = generator.generate_multiple_prisms(
        num_prisms=args.num_prisms,
        output_file=os.path.join(args.output_dir, "prisms_info.txt")
    )
    
    # 步骤2: 读取URDF文件信息
    print("\nStep 2: Reading URDF information...")
    from urdf_reader import URDFReader
    
    reader = URDFReader()
    prisms_data = reader.read_prisms_from_info_file(
        os.path.join(args.output_dir, "prisms_info.txt")
    )
    
    # 保存为JSON
    reader.save_to_json(
        prisms_data,
        os.path.join(args.output_dir, "prisms_data.json")
    )
    
    # 步骤3: PyBullet仿真
    print("\nStep 3: Running PyBullet simulation...")
    from simulation import PrismSimulation
    
    sim = PrismSimulation(gui=not args.no_gui)
    sim.load_ground()
    
    # 随机放置棱柱
    placed_prisms = sim.random_place_prisms(prisms_data)
    
    print(f"Placed {len(placed_prisms)} prisms in simulation")
    
    # 运行仿真
    sim.simulate(steps=args.sim_steps)
    
    # 收集仿真状态
    sim_state = []
    for prism in placed_prisms:
        state = sim.get_prism_state(prism['id'])
        sim_state.append(state)
        
        print(f"\n{prism['info']['name']}:")
        print(f"  Final position: {state['position']}")
        print(f"  Z-axis rotation: {state['z_angle']:.3f} rad ({state['z_angle']*180/np.pi:.1f}°)")
    
    # 步骤4: 创建2D投影可视化
    print("\nStep 4: Creating 2D projection visualizations...")
    from visualizer_projection import ProjectionVisualizer
    
    visualizer = ProjectionVisualizer()
    
    # 创建整体仿真视图
    fig1, ax1 = visualizer.visualize_simulation(
        prisms_data, sim_state,
        title=f"Quad Prism Simulation (n={len(prisms_data)})"
    )
    
    # 创建单独投影视图
    fig2, axes2 = visualizer.plot_individual_projections(prisms_data, sim_state)
    
    # 保存可视化结果
    visualizer.save_visualization(
        fig1, 
        os.path.join(args.output_dir, "simulation_overview.png")
    )
    
    visualizer.save_visualization(
        fig2,
        os.path.join(args.output_dir, "individual_projections.png")
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print(f"\nGenerated files in '{args.output_dir}':")
    print("  - prisms_info.txt: Prism dimensions and parameters")
    print("  - prisms_data.json: JSON format prism data")
    print("  - simulation_overview.png: Overall simulation view")
    print("  - individual_projections.png: Individual prism projections")
    print(f"\nURDF files are in 'urdf/' directory")
    
    if not args.no_gui:
        print("\nPyBullet simulation window is open.")
        print("Close the PyBullet window to continue...")
        
        # 保持PyBullet窗口打开
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nClosing simulation...")
    
    # 显示matplotlib图像
    print("\nDisplaying matplotlib visualizations...")
    import matplotlib.pyplot as plt
    plt.show()
    
    # 清理
    sim.close()
    
    print("\nDone!")

if __name__ == "__main__":
    import numpy as np  # 为print语句中的np.pi添加导入
    main()