import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
import math

class QuadPrismGenerator:
    def __init__(self, urdf_dir="urdf"):
        """初始化生成器，设置URDF文件保存目录"""
        self.urdf_dir = urdf_dir
        os.makedirs(urdf_dir, exist_ok=True)
        
    def generate_random_quadrilateral(self, min_size=0.5, max_size=2.0, area_range=None):
        """生成随机四边形顶点（逆时针顺序）"""
        # 从随机矩形开始，然后添加随机扰动
        width = random.uniform(min_size, max_size)
        height = random.uniform(min_size, max_size)
        
        # 基础矩形顶点
        base_points = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2]
        ])
        
        # 添加随机扰动
        perturbation = np.random.uniform(-0.3, 0.3, (4, 2)) * np.array([width, height])
        points = base_points + perturbation
        
        # 新增功能: 控制四边形面积在指定范围内
        if area_range is not None:
            min_area, max_area = area_range
            
            # 计算当前面积
            current_area = self._calculate_polygon_area(points)
            
            # 如果面积不在指定范围内，调整四边形
            if current_area < min_area or current_area > max_area:
                # 方法1: 按比例缩放
                target_area = random.uniform(min_area, max_area)
                scale_factor = math.sqrt(target_area / current_area)
                points = points * scale_factor

        
        return points
    
    def _calculate_polygon_area(self, points):
        """计算多边形面积（鞋带公式）"""
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2.0
        return area


    def calculate_height(self, base_points, min_height=0.5, max_height=3.0):
        """现采用固定高度"""
        # # 计算四边形面积（使用鞋带公式）
        # area = 0
        # n = len(base_points)
        # for i in range(n):
        #     j = (i + 1) % n
        #     area += base_points[i][0] * base_points[j][1]
        #     area -= base_points[j][0] * base_points[i][1]
        # area = abs(area) / 2.0
        
        # # 面积越大，高度越小（保持体积相对稳定）
        # base_size = math.sqrt(area)
        # height = min_height + (max_height - min_height) / (1 + base_size)
        
        # 现采用固定高度
        height = 3.0
        return height

    def create_mesh_stl(self, base_points, height, filename, is_convex=True):
        """创建STL网格文件，支持凸/凹四边形处理"""
        import trimesh

        # 创建底面和顶面顶点
        bottom_vertices = np.hstack([base_points, np.zeros((4, 1))])
        top_vertices = np.hstack([base_points, np.full((4, 1), height)])

        vertices = np.vstack([bottom_vertices, top_vertices])

        if is_convex:
            # 凸四边形处理（原来的方法）
            faces = []
            
            # 侧面
            for i in range(4):
                j = (i + 1) % 4
                # 侧面四边形分成两个三角形
                faces.append([i, j, 4 + i])       # 三角形1
                faces.append([j, 4 + j, 4 + i])   # 三角形2
            
            # 底面（分成2个三角形）- 假设为凸四边形
            faces.append([0, 2, 1])  # 对角线0-2
            faces.append([0, 3, 2])  # 对角线0-2的另一半
            
            # 顶面
            faces.append([4, 5, 6])  # 顶面对应
            faces.append([4, 6, 7])
            
        else:
            # 凹四边形处理 - 使用更稳健的方法
            faces = []
            
            # 侧面处理（与凸四边形相同，因为侧面总是凸的）
            for i in range(4):
                j = (i + 1) % 4
                faces.append([i, j, 4 + i])
                faces.append([j, 4 + j, 4 + i])
            
            # 底面处理 - 使用三角剖分
            bottom_points_2d = np.array(base_points)
                    
            # 手动三角剖分（适用于简单凹四边形）
            # 找到凹点（内角大于180度的点）
            concave_point = self.find_concave_point(bottom_points_2d)
            
            if concave_point >= 0:
                # 如果有凹点，使用凹点与相对点分割
                opposite = (concave_point + 2) % 4
                faces.append([concave_point, (concave_point + 1) % 4, opposite])
                faces.append([opposite, (concave_point + 3) % 4, concave_point])
                faces.append([(concave_point + 1) % 4, (concave_point + 2) % 4, opposite])
            else:
                # 默认三角剖分
                faces.append([0, 1, 2])
                faces.append([0, 2, 3])
            
            # 顶面处理 - 使用与底面相同的三角剖分
            for face in faces[-2:]:  # 取最后两个面（底面的三角形）
                top_face = [v + 4 for v in face]  # 偏移到顶面顶点
                faces.append(top_face)

        # 确保所有面都是三角形
        faces = [face for face in faces if len(face) == 3]

        # 创建网格
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 验证网格
        if not mesh.is_watertight:
            print(f"Warning: Mesh may not be watertight for {filename}")

        # 保存为STL
        mesh.export(filename)

        print(f"Created STL ({'convex' if is_convex else 'concave'}): {filename}")
        return vertices

    def find_concave_point(self, points):
        """找到凹四边形的凹点索引"""
        n = len(points)
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
            
            # 计算叉积
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            # 如果叉积为负（在右手坐标系中），则为凹点
            if cross < 0:
                return (i + 1) % n
        
        return -1  # 没有找到凹点（凸多边形）

    def check_convexity(self, points):
        """检查多边形是否为凸多边形"""
        n = len(points)
        if n < 3:
            return True
        
        # 计算所有连续边的叉积符号
        sign = 0
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
            
            # 计算向量叉积
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            if cross != 0:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif sign * cross < 0:  # 符号不同
                    return False
        
        return True


    # def create_mesh_stl(self, base_points, height, filename):
    #     """创建STL网格文件"""
    #     import trimesh
        
    #     # 创建底面和顶面顶点
    #     bottom_vertices = np.hstack([base_points, np.zeros((4, 1))])
    #     top_vertices = np.hstack([base_points, np.full((4, 1), height)])
        
    #     # 创建侧面（4个四边形，每个分成2个三角形）
    #     vertices = np.vstack([bottom_vertices, top_vertices])
        
    #     # 定义三角形面（8个侧面三角形 + 2个底面三角形）
    #     faces = []
        
    #     # 侧面
    #     for i in range(4):
    #         j = (i + 1) % 4
    #         # 侧面四边形分成两个三角形
    #         faces.append([i, j, 4 + i])       # 三角形1
    #         faces.append([j, 4 + j, 4 + i])   # 三角形2
        
    #     # 底面（分成2个三角形）
    #     faces.append([0, 2, 1])
    #     faces.append([0, 3, 2])
        
    #     # 顶面
    #     faces.append([4, 5, 6])
    #     faces.append([4, 6, 7])
        
    #     # 创建网格
    #     mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
    #     # 保存为STL
    #     mesh.export(filename)
        
    #     return vertices
    
    def generate_urdf(self, base_points, height, name, color=None):
        """生成URDF文件"""
        if color is None:
            color = [random.random() for _ in range(3)] + [1.0]  # RGBA
        
        # 创建URDF结构
        robot = ET.Element("robot", name=name)
        
        # 链接
        link = ET.SubElement(robot, "link", name="base_link")
        
        # 视觉元素
        visual = ET.SubElement(link, "visual")
        
        # 原点
        origin = ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        
        # 几何 - 网格
        geometry = ET.SubElement(visual, "geometry")
        mesh_file = f"{name}.stl"
        mesh = ET.SubElement(geometry, "mesh", filename=mesh_file)
        
        # 材质
        material = ET.SubElement(visual, "material", name="color")
        color_elem = ET.SubElement(material, "color", rgba=f"{color[0]} {color[1]} {color[2]} {color[3]}")
        
        # 碰撞元素
        collision = ET.SubElement(link, "collision")
        origin_coll = ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry_coll = ET.SubElement(collision, "geometry")
        mesh_coll = ET.SubElement(geometry_coll, "mesh", filename=mesh_file)
        
        # 惯性元素
        inertia = ET.SubElement(link, "inertial")
        origin_inertia = ET.SubElement(inertia, "origin", xyz="0 0 0", rpy="0 0 0")
        mass = ET.SubElement(inertia, "mass", value="1.0")
        inertia_val = ET.SubElement(inertia, "inertia", 
                                   ixx="0.1", ixy="0", ixz="0",
                                   iyy="0.1", iyz="0", izz="0.1")
        
        # 美化XML输出
        xml_str = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ")
        
        # 保存URDF文件
        urdf_filename = os.path.join(self.urdf_dir, f"{name}.urdf")
        with open(urdf_filename, "w") as f:
            f.write(xml_str)
        
        # 创建对应的STL文件
        stl_filename = os.path.join(self.urdf_dir, mesh_file)
        is_convex = self.check_convexity(base_points)
        self.create_mesh_stl(base_points, height, stl_filename, is_convex)
        
        print(f"Generated URDF: {urdf_filename}")
        print(f"Base points: {base_points}")
        print(f"Height: {height}")
        
        return urdf_filename, base_points, height
    
    def generate_multiple_prisms(self, num_prisms=3, output_file="urdf/prisms_info.txt"):
        """生成多个四边棱柱"""
        prisms_info = []
        
        for i in range(num_prisms):
            name = f"quad_prism_{i+1}"
            
            # 生成随机四边形底面
            base_points = self.generate_random_quadrilateral()
            
            # 计算高度
            height = self.calculate_height(base_points)
            
            # 生成URDF
            urdf_file = self.generate_urdf(base_points, height, name)
            
            # 存储信息
            prism_info = {
                'name': name,
                'urdf_file': urdf_file[0],
                'base_points': base_points.tolist(),
                'height': height,
                'center': [0, 0, height/2]  # 几何中心
            }
            prisms_info.append(prism_info)
        
        # 保存信息到文件
        with open(output_file, 'w') as f:
            for info in prisms_info:
                f.write(f"Name: {info['name']}\n")
                f.write(f"URDF File: {info['urdf_file']}\n")
                f.write(f"Height: {info['height']:.3f}\n")
                f.write(f"Base Points:\n")
                for point in info['base_points']:
                    f.write(f"  ({point[0]:.3f}, {point[1]:.3f})\n")
                f.write(f"Center: {info['center']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"\nGenerated {num_prisms} prisms. Info saved to {output_file}")
        return prisms_info

if __name__ == "__main__":
    generator = QuadPrismGenerator()
    prisms = generator.generate_multiple_prisms(num_prisms=3)