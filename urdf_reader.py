import os
import numpy as np
import xml.etree.ElementTree as ET
import re
import json

class URDFReader:
    def __init__(self):
        pass
    
    def read_urdf_info(self, urdf_file):
        """读取URDF文件基本信息"""
        info = {
            'file': urdf_file,
            'name': '',
            'mesh_file': '',
            'dimensions': {}
        }
        
        try:
            tree = ET.parse(urdf_file)
            root = tree.getroot()
            
            # 获取机器人名称
            info['name'] = root.get('name', 'unknown')
            
            # 查找网格文件
            for mesh in root.findall('.//mesh'):
                mesh_filename = mesh.get('filename', '')
                if mesh_filename:
                    info['mesh_file'] = mesh_filename
                    break
            
            # 尝试从STL文件获取尺寸
            if info['mesh_file']:
                dims = self.get_dimensions_from_stl(
                    os.path.join(os.path.dirname(urdf_file), info['mesh_file'])
                )
                info['dimensions'] = dims
            
            return info
            
        except Exception as e:
            print(f"Error reading URDF file {urdf_file}: {e}")
            return info
    
    def get_dimensions_from_stl(self, stl_file):
        """从STL文件获取尺寸信息"""
        try:
            import trimesh
            mesh = trimesh.load(stl_file)
            
            # 获取边界框
            bbox = mesh.bounds
            min_coords = bbox[0]
            max_coords = bbox[1]
            
            dimensions = {
                'min_x': float(min_coords[0]),
                'max_x': float(max_coords[0]),
                'min_y': float(min_coords[1]),
                'max_y': float(max_coords[1]),
                'min_z': float(min_coords[2]),
                'max_z': float(max_coords[2]),
                'width': float(max_coords[0] - min_coords[0]),
                'depth': float(max_coords[1] - min_coords[1]),
                'height': float(max_coords[2] - min_coords[2])
            }
            
            return dimensions
            
        except Exception as e:
            print(f"Error reading STL file {stl_file}: {e}")
            return {}
    
    def read_prisms_from_info_file(self, info_file="urdf/prisms_info.txt"):
        """从信息文件读取所有棱柱信息"""
        prisms = []
        current_prism = None
        reading_points = False
        point_count = 0
        
        with open(info_file, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('Name:'):
                if current_prism:
                    prisms.append(current_prism)
                current_prism = {'name': line.split(': ')[1]}
                reading_points = False
                point_count = 0
                
            elif line.startswith('URDF File:'):
                if current_prism:
                    current_prism['urdf_file'] = line.split(': ')[1]
                reading_points = False
                
            elif line.startswith('Height:'):
                if current_prism:
                    try:
                        current_prism['height'] = float(line.split(': ')[1])
                    except:
                        current_prism['height'] = 1.0
                reading_points = False
                
            elif line.startswith('Base Points:'):
                if current_prism:
                    current_prism['base_points'] = []
                    reading_points = True
                    point_count = 0
                    
            elif reading_points and point_count < 4:
                # 尝试解析点的坐标
                match = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if match and len(match) >= 2:
                    try:
                        x = float(match[0])
                        y = float(match[1])
                        current_prism['base_points'].append([x, y])
                        point_count += 1
                    except ValueError:
                        pass
                
                if point_count >= 4:
                    reading_points = False
                    
            elif line.startswith('Center:'):
                if current_prism:
                    match = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                    if match and len(match) >= 3:
                        try:
                            current_prism['center'] = [
                                float(match[0]), 
                                float(match[1]), 
                                float(match[2])
                            ]
                        except ValueError:
                            current_prism['center'] = [0, 0, 0]
                reading_points = False
                
            elif line.startswith('-' * 40):  # 分隔线
                reading_points = False
        
        # 添加最后一个棱柱
        if current_prism:
            # 确保base_points存在
            if 'base_points' not in current_prism:
                current_prism['base_points'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
            prisms.append(current_prism)
        
        print(f"Successfully read {len(prisms)} prisms from {info_file}")
        return prisms
    
    def save_to_json(self, prisms_info, output_file="urdf/prisms_data.json"):
        """保存信息到JSON文件"""
        with open(output_file, 'w') as f:
            json.dump(prisms_info, f, indent=2)
        print(f"Data saved to {output_file}")
    
    def read_directly_from_prisms_info(self, prisms_info_list):
        """直接从quad_prism_generator返回的信息列表读取"""
        prisms_data = []
        
        for info in prisms_info_list:
            prism_data = {
                'name': info['name'],
                'urdf_file': info['urdf_file'],
                'base_points': info['base_points'],
                'height': info['height'],
                'center': info['center']
            }
            prisms_data.append(prism_data)
        
        return prisms_data

if __name__ == "__main__":
    reader = URDFReader()
    
    # 检查文件是否存在
    info_file = "urdf/prisms_info.txt"
    if os.path.exists(info_file):
        # 从信息文件读取
        prisms = reader.read_prisms_from_info_file(info_file)
        
        # 打印读取的信息
        for i, prism in enumerate(prisms):
            print(f"\nPrism {i+1}: {prism['name']}")
            print(f"  URDF File: {prism.get('urdf_file', 'N/A')}")
            print(f"  Height: {prism.get('height', 'N/A')}")
            print(f"  Base Points: {len(prism.get('base_points', []))} points")
            for j, point in enumerate(prism.get('base_points', [])):
                print(f"    Point {j+1}: ({point[0]:.3f}, {point[1]:.3f})")
            print(f"  Center: {prism.get('center', 'N/A')}")
        
        # 保存为JSON
        reader.save_to_json(prisms)
    else:
        print(f"Warning: {info_file} not found. Please run quad_prism_generator.py first.")
        
        # 尝试从urdf目录读取URDF文件
        urdf_dir = "urdf"
        if os.path.exists(urdf_dir):
            print(f"\nAttempting to read URDF files from {urdf_dir}/")
            urdf_files = [f for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
            
            if urdf_files:
                prisms = []
                for urdf_file in urdf_files:
                    urdf_path = os.path.join(urdf_dir, urdf_file)
                    info = reader.read_urdf_info(urdf_path)
                    prisms.append(info)
                    print(f"  Read: {urdf_file}")
                
                if prisms:
                    reader.save_to_json(prisms, "urdf/urdf_info.json")
            else:
                print("No URDF files found.")