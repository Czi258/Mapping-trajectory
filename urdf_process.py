"""
URDF模型投影模块
功能：读取URDF文件，生成模型的2D投影图像
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
import pybullet as p
import pybullet_data
import tempfile
import shutil


class URDFProjector:
    """URDF模型二维投影生成器"""
    
    def __init__(self, urdf_path: str = None, scaling: float = 1.0):
        """
        初始化URDF投影器
        
        Args:
            urdf_path: URDF文件路径
            scaling: 模型缩放比例
        """
        self.urdf_path = urdf_path
        self.scaling = scaling
        self.model_info = {}
        
        if urdf_path:
            self.load_urdf(urdf_path)
    
    def load_urdf(self, urdf_path: str) -> Dict:
        """
        加载URDF文件并解析模型信息
        
        Args:
            urdf_path: URDF文件路径
            
        Returns:
            dict: 模型信息
        """
        if not os.path.exists(urdf_path):
            # 尝试在pybullet数据目录中查找
            pybullet_data_path = pybullet_data.getDataPath()
            possible_path = os.path.join(pybullet_data_path, urdf_path)
            if os.path.exists(possible_path):
                urdf_path = possible_path
            else:
                raise FileNotFoundError(f"URDF文件不存在: {urdf_path}")
        
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # 获取URDF文件所在目录，用于解析相对路径
            self.urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
            
            # 解析模型基本信息
            self.model_info = {
                'name': root.get('name', 'unknown'),
                'links': self._parse_links(root),
                'joints': self._parse_joints(root),
                'materials': self._parse_materials(root),
                'bounding_box': self._calculate_bounding_box(root),
                'file_path': urdf_path
            }
            
            print(f"✓ 成功加载URDF模型: {self.model_info['name']}")
            print(f"  文件路径: {urdf_path}")
            print(f"  包含 {len(self.model_info['links'])} 个链接")
            print(f"  包含 {len(self.model_info['joints'])} 个关节")
            
            return self.model_info
            
        except Exception as e:
            raise ValueError(f"解析URDF文件失败: {e}")
    
    def _parse_links(self, root: ET.Element) -> List[Dict]:
        """解析所有链接"""
        links = []
        
        for link_elem in root.findall('link'):
            link_info = {
                'name': link_elem.get('name'),
                'visuals': [],
                'collisions': [],
                'inertial': None
            }
            
            # 解析视觉元素
            for visual in link_elem.findall('visual'):
                visual_info = self._parse_geometry(visual)
                if visual_info:
                    link_info['visuals'].append(visual_info)
            
            # 解析碰撞元素
            for collision in link_elem.findall('collision'):
                collision_info = self._parse_geometry(collision)
                if collision_info:
                    link_info['collisions'].append(collision_info)
            
            links.append(link_info)
        
        return links
    
    def _parse_geometry(self, elem: ET.Element) -> Optional[Dict]:
        """解析几何元素"""
        geometry = elem.find('geometry')
        if geometry is None:
            return None
        
        # 查找几何类型
        if geometry.find('box') is not None:
            box = geometry.find('box')
            size_str = box.get('size', '1 1 1')
            size = [float(x) * self.scaling for x in size_str.split()]
            return {
                'type': 'box',
                'dimensions': size,  # [长, 宽, 高]
                'pose': self._parse_pose(elem.find('origin')),
                'material': self._parse_material(elem.find('material'))
            }
        
        elif geometry.find('sphere') is not None:
            sphere = geometry.find('sphere')
            radius = float(sphere.get('radius', '0.5')) * self.scaling
            return {
                'type': 'sphere',
                'radius': radius,
                'pose': self._parse_pose(elem.find('origin')),
                'material': self._parse_material(elem.find('material'))
            }
        
        elif geometry.find('cylinder') is not None:
            cylinder = geometry.find('cylinder')
            radius = float(cylinder.get('radius', '0.5')) * self.scaling
            length = float(cylinder.get('length', '1.0')) * self.scaling
            return {
                'type': 'cylinder',
                'radius': radius,
                'length': length,
                'pose': self._parse_pose(elem.find('origin')),
                'material': self._parse_material(elem.find('material'))
            }
        
        elif geometry.find('mesh') is not None:
            mesh = geometry.find('mesh')
            filename = mesh.get('filename', '')
            
            # 处理相对路径
            if filename.startswith('package://'):
                # 移除package://前缀
                filename = filename.replace('package://', '')
            elif not os.path.isabs(filename):
                # 相对路径转换为绝对路径
                filename = os.path.join(self.urdf_dir, filename)
            
            scale_str = mesh.get('scale', '1 1 1')
            scale = [float(x) * self.scaling for x in scale_str.split()]
            
            # 对于网格文件，我们尝试估算尺寸
            estimated_size = self._estimate_mesh_size(filename, scale)
            
            return {
                'type': 'mesh',
                'filename': filename,
                'scale': scale,
                'estimated_size': estimated_size,
                'pose': self._parse_pose(elem.find('origin')),
                'material': self._parse_material(elem.find('material'))
            }
        
        return None
    
    def _estimate_mesh_size(self, filename: str, scale: List[float]) -> List[float]:
        """估算网格文件尺寸"""
        # 默认尺寸
        default_size = [1.0 * scale[0], 1.0 * scale[1], 1.0 * scale[2]]
        
        if not os.path.exists(filename):
            print(f"⚠️ 网格文件不存在: {filename}，使用默认尺寸")
            return default_size
        
        # 根据文件扩展名尝试不同方法
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.obj':
            # 尝试解析OBJ文件获取边界框
            try:
                return self._parse_obj_bbox(filename, scale)
            except:
                return default_size
        elif ext == '.stl':
            # STL文件处理
            try:
                return self._parse_stl_bbox(filename, scale)
            except:
                return default_size
        else:
            return default_size
    
    def _parse_obj_bbox(self, filename: str, scale: List[float]) -> List[float]:
        """解析OBJ文件获取边界框"""
        vertices = []
        
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x * scale[0], y * scale[1], z * scale[2]])
        
        if not vertices:
            return [1.0 * scale[0], 1.0 * scale[1], 1.0 * scale[2]]
        
        vertices = np.array(vertices)
        bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
        return bbox_size.tolist()
    
    def _parse_stl_bbox(self, filename: str, scale: List[float]) -> List[float]:
        """解析STL文件获取边界框（简化版）"""
        # 简化的STL解析，只读取前一些顶点
        vertices = []
        
        try:
            with open(filename, 'rb') as f:
                # 跳过头部
                f.read(80)
                
                # 读取三角形数量
                triangle_count_bytes = f.read(4)
                if len(triangle_count_bytes) == 4:
                    triangle_count = int.from_bytes(triangle_count_bytes, 'little')
                    
                    # 只读取前100个三角形用于估算
                    max_triangles = min(triangle_count, 100)
                    
                    for _ in range(max_triangles):
                        # 读取法向量和3个顶点
                        f.read(12)  # 跳过法向量
                        
                        for _ in range(3):
                            # 读取顶点坐标
                            x_bytes = f.read(4)
                            y_bytes = f.read(4)
                            z_bytes = f.read(4)
                            
                            if len(x_bytes) == 4 and len(y_bytes) == 4 and len(z_bytes) == 4:
                                x = struct.unpack('<f', x_bytes)[0] * scale[0]
                                y = struct.unpack('<f', y_bytes)[0] * scale[1]
                                z = struct.unpack('<f', z_bytes)[0] * scale[2]
                                vertices.append([x, y, z])
                        
                        f.read(2)  # 跳过属性字节计数
        except:
            pass
        
        if not vertices:
            return [1.0 * scale[0], 1.0 * scale[1], 1.0 * scale[2]]
        
        vertices = np.array(vertices)
        bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
        return bbox_size.tolist()
    
    def _parse_pose(self, origin_elem: Optional[ET.Element]) -> Dict:
        """解析位姿信息"""
        if origin_elem is None:
            return {
                'xyz': [0, 0, 0],
                'rpy': [0, 0, 0]
            }
        
        xyz_str = origin_elem.get('xyz', '0 0 0')
        rpy_str = origin_elem.get('rpy', '0 0 0')
        
        return {
            'xyz': [float(x) * self.scaling for x in xyz_str.split()],
            'rpy': [float(r) for r in rpy_str.split()]
        }
    
    def _parse_material(self, material_elem: Optional[ET.Element]) -> Optional[Dict]:
        """解析材质信息"""
        if material_elem is None:
            return None
        
        color_elem = material_elem.find('color')
        if color_elem is not None:
            rgba_str = color_elem.get('rgba', '1 1 1 1')
            rgba = [float(c) for c in rgba_str.split()]
            return {'rgba': rgba}
        
        texture_elem = material_elem.find('texture')
        if texture_elem is not None:
            return {'texture': texture_elem.get('filename')}
        
        return {'name': material_elem.get('name')}
    
    def _parse_materials(self, root: ET.Element) -> Dict:
        """解析材质定义"""
        materials = {}
        
        for material_elem in root.findall('material'):
            name = material_elem.get('name')
            color_elem = material_elem.find('color')
            
            if color_elem is not None:
                rgba_str = color_elem.get('rgba', '1 1 1 1')
                rgba = [float(c) for c in rgba_str.split()]
                materials[name] = {'rgba': rgba}
            
            texture_elem = material_elem.find('texture')
            if texture_elem is not None:
                if name not in materials:
                    materials[name] = {}
                materials[name]['texture'] = texture_elem.get('filename')
        
        return materials
    
    def _parse_joints(self, root: ET.Element) -> List[Dict]:
        """解析所有关节"""
        joints = []
        
        for joint_elem in root.findall('joint'):
            joint_info = {
                'name': joint_elem.get('name'),
                'type': joint_elem.get('type', 'fixed'),
                'parent': joint_elem.find('parent').get('link') if joint_elem.find('parent') else None,
                'child': joint_elem.find('child').get('link') if joint_elem.find('child') else None,
                'pose': self._parse_pose(joint_elem.find('origin')),
                'axis': self._parse_axis(joint_elem.find('axis'))
            }
            joints.append(joint_info)
        
        return joints
    
    def _parse_axis(self, axis_elem: Optional[ET.Element]) -> Optional[List[float]]:
        """解析关节轴"""
        if axis_elem is None:
            return None
        
        xyz_str = axis_elem.get('xyz', '0 0 1')
        return [float(x) for x in xyz_str.split()]
    
    def _calculate_bounding_box(self, root: ET.Element) -> Dict:
        """计算模型的边界框"""
        all_points = []
        
        for link_elem in root.findall('link'):
            for visual in link_elem.findall('visual'):
                geom_info = self._parse_geometry(visual)
                if geom_info:
                    pose = geom_info['pose']
                    x, y, z = pose['xyz']
                    
                    if geom_info['type'] == 'box':
                        dx, dy, dz = geom_info['dimensions']
                        # 计算立方体的8个顶点
                        for dx_sign in [-0.5, 0.5]:
                            for dy_sign in [-0.5, 0.5]:
                                for dz_sign in [-0.5, 0.5]:
                                    point = [
                                        x + dx_sign * dx,
                                        y + dy_sign * dy,
                                        z + dz_sign * dz
                                    ]
                                    all_points.append(point)
                    
                    elif geom_info['type'] == 'sphere':
                        r = geom_info['radius']
                        # 球体的边界点
                        for axis in range(3):
                            point_neg = [x, y, z]
                            point_pos = [x, y, z]
                            point_neg[axis] -= r
                            point_pos[axis] += r
                            all_points.extend([point_neg, point_pos])
                    
                    elif geom_info['type'] == 'cylinder':
                        r = geom_info['radius']
                        h = geom_info['length']
                        # 圆柱体的边界点
                        for angle in np.linspace(0, 2*np.pi, 8):
                            for z_sign in [-0.5, 0.5]:
                                point = [
                                    x + r * np.cos(angle),
                                    y + r * np.sin(angle),
                                    z + z_sign * h
                                ]
                                all_points.append(point)
                    
                    elif geom_info['type'] == 'mesh':
                        if 'estimated_size' in geom_info:
                            dx, dy, dz = geom_info['estimated_size']
                            # 网格物体的边界点
                            for dx_sign in [-0.5, 0.5]:
                                for dy_sign in [-0.5, 0.5]:
                                    for dz_sign in [-0.5, 0.5]:
                                        point = [
                                            x + dx_sign * dx,
                                            y + dy_sign * dy,
                                            z + dz_sign * dz
                                        ]
                                        all_points.append(point)
        
        if not all_points:
            return {'min': [0, 0, 0], 'max': [0, 0, 0], 'center': [0, 0, 0]}
        
        all_points = np.array(all_points)
        bbox_min = all_points.min(axis=0)
        bbox_max = all_points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        
        return {
            'min': bbox_min.tolist(),
            'max': bbox_max.tolist(),
            'center': bbox_center.tolist(),
            'size': (bbox_max - bbox_min).tolist()
        }
    
    def create_2d_projection(self, resolution: int = 100, 
                           projection_plane: str = 'xy',
                           fill_color: str = None,
                           edge_color: str = 'black',
                           alpha: float = 0.8,
                           show_axes: bool = True) -> plt.Figure:
        """
        创建模型的2D投影图像
        
        Args:
            resolution: 图像分辨率（像素）
            projection_plane: 投影平面 ('xy', 'xz', 'yz')
            fill_color: 填充颜色，如果为None则使用模型材质颜色
            edge_color: 边缘颜色
            alpha: 透明度
            show_axes: 是否显示坐标轴
            
        Returns:
            matplotlib Figure对象
        """
        if not self.model_info:
            raise ValueError("请先加载URDF模型")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        
        # 设置坐标轴
        plane_labels = {
            'xy': ('X (m)', 'Y (m)'),
            'xz': ('X (m)', 'Z (m)'),
            'yz': ('Y (m)', 'Z (m)')
        }
        
        if projection_plane not in plane_labels:
            raise ValueError(f"投影平面必须是: {list(plane_labels.keys())}")
        
        x_label, y_label = plane_labels[projection_plane]
        
        if show_axes:
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
        
        model_name = self.model_info['name']
        ax.set_title(f'URDF模型: {model_name} - {projection_plane.upper()}平面投影', 
                    fontsize=14, pad=20)
        
        # 绘制所有链接的投影
        patches_list = []
        
        for link_info in self.model_info['links']:
            for visual_info in link_info['visuals']:
                patch = self._create_geometry_patch(visual_info, projection_plane, 
                                                  fill_color, edge_color, alpha)
                if patch:
                    patches_list.append(patch)
        
        # 批量添加补丁
        for patch in patches_list:
            ax.add_patch(patch)
        
        # 设置显示范围
        bbox = self.model_info['bounding_box']
        if projection_plane == 'xy':
            min_x, min_y = bbox['min'][0], bbox['min'][1]
            max_x, max_y = bbox['max'][0], bbox['max'][1]
        elif projection_plane == 'xz':
            min_x, min_y = bbox['min'][0], bbox['min'][2]
            max_x, max_y = bbox['max'][0], bbox['max'][2]
        else:  # yz
            min_x, min_y = bbox['min'][1], bbox['min'][2]
            max_x, max_y = bbox['max'][1], bbox['max'][2]
        
        # 添加一些边界余量
        margin_x = (max_x - min_x) * 0.2
        margin_y = (max_y - min_y) * 0.2
        margin = max(margin_x, margin_y, 0.1)
        
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        
        if show_axes:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 添加信息文本
        info_text = f"""
模型信息:
• 名称: {model_name}
• 链接数: {len(self.model_info['links'])}
• 关节数: {len(self.model_info['joints'])}
• 投影平面: {projection_plane.upper()}
• 边界框: [{bbox['min'][0]:.2f}, {bbox['min'][1]:.2f}] -> [{bbox['max'][0]:.2f}, {bbox['max'][1]:.2f}]
"""
        
        if show_axes:
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                           alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        
        return fig
    
    def _create_geometry_patch(self, geom_info: Dict, projection_plane: str,
                             fill_color: Optional[str], edge_color: str, alpha: float) -> Optional[patches.Patch]:
        """创建几何体的matplotlib补丁"""
        pose = geom_info['pose']
        x, y, z = pose['xyz']
        rroll, rpitch, ryaw = pose['rpy']
        
        # 确定要使用的颜色
        if fill_color is None and geom_info.get('material'):
            material = geom_info['material']
            if 'rgba' in material:
                rgba = material['rgba']
                facecolor = rgba[:3]  # RGB
                if len(rgba) > 3:
                    alpha = rgba[3] * alpha
            else:
                facecolor = 'gray'
        else:
            facecolor = fill_color or 'gray'
        
        geom_type = geom_info['type']
        
        if projection_plane == 'xy':
            # XY平面投影（从上往下看）
            if geom_type == 'box':
                dx, dy, dz = geom_info['dimensions']
                # 投影为矩形
                rect = patches.Rectangle(
                    (x - dx/2, y - dy/2), dx, dy,
                    linewidth=1.5, edgecolor=edge_color, 
                    facecolor=facecolor, alpha=alpha,
                    angle=np.degrees(ryaw),  # 考虑Z轴旋转
                    rotation_point='center'
                )
                return rect
                
            elif geom_type == 'sphere':
                radius = geom_info['radius']
                circle = patches.Circle(
                    (x, y), radius,
                    linewidth=1.5, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                return circle
                
            elif geom_type == 'cylinder':
                radius = geom_info['radius']
                circle = patches.Circle(
                    (x, y), radius,
                    linewidth=1.5, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                return circle
                
            elif geom_type == 'mesh':
                if 'estimated_size' in geom_info:
                    dx, dy, dz = geom_info['estimated_size']
                    # 使用椭圆表示网格物体，更符合实际
                    ellipse = patches.Ellipse(
                        (x, y), dx, dy,
                        linewidth=1.5, edgecolor=edge_color,
                        facecolor=facecolor, alpha=alpha,
                        angle=np.degrees(ryaw)
                    )
                    return ellipse
        
        return None
    
    def create_matplotlib_patch(self, position: List[float], 
                              orientation: List[float] = None,
                              projection_plane: str = 'xy',
                              color: str = None,
                              alpha: float = 0.6,
                              scale: float = 1.0) -> List[patches.Patch]:
        """
        创建matplotlib补丁对象，可以直接添加到matplotlib图表中
        
        Args:
            position: 位置 [x, y, z]
            orientation: 方向（四元数或欧拉角），如果为None则使用默认方向
            projection_plane: 投影平面
            color: 覆盖颜色
            alpha: 透明度
            scale: 缩放比例
            
        Returns:
            matplotlib补丁对象列表
        """
        if not self.model_info:
            raise ValueError("请先加载URDF模型")
        
        patches_list = []
        px, py, pz = position
        
        # 解析方向
        if orientation is None:
            yaw, pitch, roll = 0, 0, 0
        elif len(orientation) == 4:
            # 四元数 [x, y, z, w]
            import math
            w, x, y, z = orientation[3], orientation[0], orientation[1], orientation[2]
            # 转换为欧拉角
            roll = math.atan2(2*(w*x + y*z), 1-2*(x*x + y*y))
            pitch = math.asin(2*(w*y - z*x))
            yaw = math.atan2(2*(w*z + x*y), 1-2*(y*y + z*z))
        elif len(orientation) == 3:
            yaw, pitch, roll = orientation
        else:
            yaw, pitch, roll = 0, 0, 0
        
        for link_info in self.model_info['links']:
            for visual_info in link_info['visuals']:
                # 获取几何体相对位置
                rel_pose = visual_info['pose']
                rx, ry, rz = rel_pose['xyz']
                rroll, rpitch, ryaw = rel_pose['rpy']
                
                # 计算绝对位置
                abs_x = px + rx * scale
                abs_y = py + ry * scale
                abs_z = pz + rz * scale
                
                # 计算绝对旋转
                abs_yaw = yaw + ryaw
                
                geom_type = visual_info['type']
                
                # 确定颜色
                if color is None and visual_info.get('material'):
                    material = visual_info['material']
                    if 'rgba' in material:
                        rgba = material['rgba']
                        facecolor = rgba[:3]
                        patch_alpha = rgba[3] if len(rgba) > 3 else alpha
                    else:
                        facecolor = 'gray'
                        patch_alpha = alpha
                else:
                    facecolor = color or 'gray'
                    patch_alpha = alpha
                
                if projection_plane == 'xy':
                    # XY平面投影
                    if geom_type == 'box':
                        dx, dy, dz = visual_info['dimensions']
                        dx, dy = dx * scale, dy * scale
                        rect = patches.Rectangle(
                            (abs_x - dx/2, abs_y - dy/2), dx, dy,
                            linewidth=1.0, edgecolor='black',
                            facecolor=facecolor, alpha=patch_alpha,
                            angle=np.degrees(abs_yaw)
                        )
                        patches_list.append(rect)
                        
                    elif geom_type == 'sphere':
                        radius = visual_info['radius'] * scale
                        circle = patches.Circle(
                            (abs_x, abs_y), radius,
                            linewidth=1.0, edgecolor='black',
                            facecolor=facecolor, alpha=patch_alpha
                        )
                        patches_list.append(circle)
                        
                    elif geom_type == 'cylinder':
                        radius = visual_info['radius'] * scale
                        circle = patches.Circle(
                            (abs_x, abs_y), radius,
                            linewidth=1.0, edgecolor='black',
                            facecolor=facecolor, alpha=patch_alpha
                        )
                        patches_list.append(circle)
                        
                    elif geom_type == 'mesh':
                        if 'estimated_size' in visual_info:
                            dx, dy, dz = visual_info['estimated_size']
                            dx, dy = dx * scale, dy * scale
                            ellipse = patches.Ellipse(
                                (abs_x, abs_y), dx, dy,
                                linewidth=1.0, edgecolor='black',
                                facecolor=facecolor, alpha=patch_alpha,
                                angle=np.degrees(abs_yaw)
                            )
                            patches_list.append(ellipse)
        
        return patches_list
    
    def get_bounding_box_2d(self, projection_plane: str = 'xy') -> Dict:
        """
        获取2D投影的边界框
        
        Args:
            projection_plane: 投影平面
            
        Returns:
            边界框信息
        """
        if not self.model_info:
            raise ValueError("请先加载URDF模型")
        
        bbox = self.model_info['bounding_box']
        
        if projection_plane == 'xy':
            return {
                'min': [bbox['min'][0], bbox['min'][1]],
                'max': [bbox['max'][0], bbox['max'][1]],
                'center': [bbox['center'][0], bbox['center'][1]],
                'width': bbox['size'][0],
                'height': bbox['size'][1]
            }
        elif projection_plane == 'xz':
            return {
                'min': [bbox['min'][0], bbox['min'][2]],
                'max': [bbox['max'][0], bbox['max'][2]],
                'center': [bbox['center'][0], bbox['center'][2]],
                'width': bbox['size'][0],
                'height': bbox['size'][2]
            }
        else:  # yz
            return {
                'min': [bbox['min'][1], bbox['min'][2]],
                'max': [bbox['max'][1], bbox['max'][2]],
                'center': [bbox['center'][1], bbox['center'][2]],
                'width': bbox['size'][1],
                'height': bbox['size'][2]
            }
    
    def save_projection_image(self, filename: str, dpi: int = 300, **kwargs):
        """
        保存投影图像到文件
        
        Args:
            filename: 输出文件名
            dpi: 图像分辨率
            **kwargs: 传递给create_2d_projection的参数
        """
        fig = self.create_2d_projection(**kwargs)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 投影图像已保存: {filename}")
    
    def print_model_summary(self):
        """打印模型摘要信息"""
        if not self.model_info:
            print("模型未加载")
            return
        
        print("\n" + "="*60)
        print(f"URDF模型摘要: {self.model_info['name']}")
        print("="*60)
        
        print(f"\n1. 基本信息:")
        print(f"   • 文件路径: {self.model_info.get('file_path', 'N/A')}")
        print(f"   • 链接数量: {len(self.model_info['links'])}")
        print(f"   • 关节数量: {len(self.model_info['joints'])}")
        
        bbox = self.model_info['bounding_box']
        print(f"\n2. 边界框信息:")
        print(f"   • 最小坐标: [{bbox['min'][0]:.3f}, {bbox['min'][1]:.3f}, {bbox['min'][2]:.3f}]")
        print(f"   • 最大坐标: [{bbox['max'][0]:.3f}, {bbox['max'][1]:.3f}, {bbox['max'][2]:.3f}]")
        print(f"   • 中心坐标: [{bbox['center'][0]:.3f}, {bbox['center'][1]:.3f}, {bbox['center'][2]:.3f}]")
        print(f"   • 尺寸: [{bbox['size'][0]:.3f}, {bbox['size'][1]:.3f}, {bbox['size'][2]:.3f}]")
        
        print(f"\n3. 链接信息:")
        for i, link in enumerate(self.model_info['links'][:5]):  # 只显示前5个
            print(f"   [{i+1}] {link['name']}: {len(link['visuals'])}个视觉元素")
        
        if len(self.model_info['links']) > 5:
            print(f"   ... 还有{len(self.model_info['links']) - 5}个链接")
        
        print(f"\n4. 关节信息:")
        for i, joint in enumerate(self.model_info['joints'][:5]):  # 只显示前5个
            print(f"   [{i+1}] {joint['name']}: {joint['type']}")
            print(f"       父链接: {joint['parent']} -> 子链接: {joint['child']}")
        
        if len(self.model_info['joints']) > 5:
            print(f"   ... 还有{len(self.model_info['joints']) - 5}个关节")
        
        print("="*60)


class PyBulletURDFProjector:
    """基于PyBullet的URDF投影器（直接使用PyBullet加载模型）"""
    
    def __init__(self):
        self.body_id = None
        self.visual_shape_data = None
        self.collision_shape_data = None
    
    def load_from_body_id(self, body_id: int):
        """
        从PyBullet中的物体ID加载模型信息
        
        Args:
            body_id: PyBullet物体ID
        """
        self.body_id = body_id
        
        # 获取视觉形状数据
        self.visual_shape_data = p.getVisualShapeData(body_id)
        self.collision_shape_data = p.getCollisionShapeData(body_id, -1)
        
        if not self.visual_shape_data and not self.collision_shape_data:
            raise ValueError(f"物体 {body_id} 没有形状数据")
        
        data_count = len(self.visual_shape_data) if self.visual_shape_data else 0
        print(f"✓ 从PyBullet加载物体 {body_id}")
        print(f"  包含 {data_count} 个视觉形状")
        
        if self.collision_shape_data:
            print(f"  包含 {len(self.collision_shape_data)} 个碰撞形状")
    
    def create_matplotlib_patch(self, position: List[float], 
                              orientation: List[float] = None,
                              projection_plane: str = 'xy',
                              color: str = None,
                              alpha: float = 0.6,
                              scale: float = 1.0) -> List[patches.Patch]:
        """
        基于PyBullet数据创建matplotlib补丁
        
        Args:
            position: 位置 [x, y, z]
            orientation: 方向（四元数）
            projection_plane: 投影平面
            color: 覆盖颜色
            alpha: 透明度
            scale: 缩放比例
            
        Returns:
            matplotlib补丁对象列表
        """
        patches_list = []
        
        # 优先使用视觉形状数据
        shape_data_list = self.visual_shape_data or self.collision_shape_data or []
        
        for shape_data in shape_data_list:
            if len(shape_data) < 7:
                continue
                
            shape_type = shape_data[2]  # 形状类型
            dimensions = list(shape_data[3])  # 尺寸
            pos = list(shape_data[5])  # 位置（相对）
            
            # 获取颜色（如果有）
            facecolor = color or 'gray'
            if len(shape_data) > 7:
                rgba_color = list(shape_data[7])  # RGBA颜色
                if not color:
                    facecolor = (rgba_color[0], rgba_color[1], rgba_color[2])
                    if len(rgba_color) > 3:
                        alpha = rgba_color[3]
            
            # 计算绝对位置
            abs_pos = [
                position[0] + pos[0] * scale,
                position[1] + pos[1] * scale,
                position[2] + pos[2] * scale
            ]
            
            # 处理不同形状
            if projection_plane == 'xy':
                if shape_type == p.GEOM_BOX:
                    # 立方体
                    dx, dy, dz = dimensions[0] * scale, dimensions[1] * scale, dimensions[2] * scale
                    patch = patches.Rectangle(
                        (abs_pos[0] - dx/2, abs_pos[1] - dy/2), dx, dy,
                        linewidth=1.0, edgecolor='black',
                        facecolor=facecolor, alpha=alpha
                    )
                    patches_list.append(patch)
                    
                elif shape_type == p.GEOM_SPHERE:
                    # 球体
                    radius = dimensions[0] * scale
                    patch = patches.Circle(
                        (abs_pos[0], abs_pos[1]), radius,
                        linewidth=1.0, edgecolor='black',
                        facecolor=facecolor, alpha=alpha
                    )
                    patches_list.append(patch)
                    
                elif shape_type in [p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
                    # 圆柱体/胶囊体
                    radius = dimensions[0] * scale
                    patch = patches.Circle(
                        (abs_pos[0], abs_pos[1]), radius,
                        linewidth=1.0, edgecolor='black',
                        facecolor=facecolor, alpha=alpha
                    )
                    patches_list.append(patch)
                    
                elif shape_type == p.GEOM_MESH:
                    # 网格物体
                    if len(dimensions) >= 3:
                        dx, dy, dz = dimensions[0] * scale, dimensions[1] * scale, dimensions[2] * scale
                        patch = patches.Ellipse(
                            (abs_pos[0], abs_pos[1]), dx, dy,
                            linewidth=1.0, edgecolor='black',
                            facecolor=facecolor, alpha=alpha
                        )
                        patches_list.append(patch)
                    else:
                        # 默认圆形
                        patch = patches.Circle(
                            (abs_pos[0], abs_pos[1]), 0.5 * scale,
                            linewidth=1.0, edgecolor='black',
                            facecolor=facecolor, alpha=alpha
                        )
                        patches_list.append(patch)
        
        return patches_list
    
    def get_2d_projection_bbox(self, position: List[float], 
                             projection_plane: str = 'xy',
                             scale: float = 1.0) -> Dict:
        """
        获取2D投影的边界框
        
        Args:
            position: 位置
            projection_plane: 投影平面
            scale: 缩放比例
            
        Returns:
            边界框信息
        """
        shape_data_list = self.visual_shape_data or self.collision_shape_data or []
        
        if not shape_data_list:
            return {'min': [0, 0], 'max': [0, 0], 'center': [0, 0], 'width': 0, 'height': 0}
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = -float('inf'), -float('inf')
        
        for shape_data in shape_data_list:
            if len(shape_data) < 6:
                continue
                
            shape_type = shape_data[2]
            dimensions = list(shape_data[3])
            pos = list(shape_data[5])
            
            # 计算绝对位置
            abs_pos = [
                position[0] + pos[0] * scale,
                position[1] + pos[1] * scale,
                position[2] + pos[2] * scale
            ]
            
            if projection_plane == 'xy':
                if shape_type == p.GEOM_BOX:
                    dx, dy = dimensions[0] * scale, dimensions[1] * scale
                    min_x = min(min_x, abs_pos[0] - dx/2)
                    max_x = max(max_x, abs_pos[0] + dx/2)
                    min_y = min(min_y, abs_pos[1] - dy/2)
                    max_y = max(max_y, abs_pos[1] + dy/2)
                    
                elif shape_type in [p.GEOM_SPHERE, p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
                    radius = dimensions[0] * scale
                    min_x = min(min_x, abs_pos[0] - radius)
                    max_x = max(max_x, abs_pos[0] + radius)
                    min_y = min(min_y, abs_pos[1] - radius)
                    max_y = max(max_y, abs_pos[1] + radius)
                    
                elif shape_type == p.GEOM_MESH:
                    if len(dimensions) >= 2:
                        dx, dy = dimensions[0] * scale, dimensions[1] * scale
                        min_x = min(min_x, abs_pos[0] - dx/2)
                        max_x = max(max_x, abs_pos[0] + dx/2)
                        min_y = min(min_y, abs_pos[1] - dy/2)
                        max_y = max(max_y, abs_pos[1] + dy/2)
                    else:
                        # 默认半径
                        radius = 0.5 * scale
                        min_x = min(min_x, abs_pos[0] - radius)
                        max_x = max(max_x, abs_pos[0] + radius)
                        min_y = min(min_y, abs_pos[1] - radius)
                        max_y = max(max_y, abs_pos[1] + radius)
        
        if min_x == float('inf'):
            return {'min': [0, 0], 'max': [0, 0], 'center': [0, 0], 'width': 0, 'height': 0}
        
        return {
            'min': [min_x, min_y],
            'max': [max_x, max_y],
            'center': [(min_x + max_x)/2, (min_y + max_y)/2],
            'width': max_x - min_x,
            'height': max_y - min_y
        }


# 使用示例函数
def demo_urdf_projection():
    """演示URDF投影功能 - 使用PyBullet内置模型"""
    
    print("=" * 60)
    print("PyBullet URDF投影演示")
    print("=" * 60)
    
    # 获取PyBullet数据路径
    pybullet_data_path = pybullet_data.getDataPath()
    print(f"PyBullet数据路径: {pybullet_data_path}")
    
    # PyBullet内置的URDF文件示例
    urdf_examples = [
        "franka_panda/panda.urdf",  # 弗兰卡机械臂
        "kuka_iiwa/model.urdf",      # KUKA机械臂
        "racecar/racecar.urdf",      # 赛车模型
        "duck_vhacd.urdf",           # 鸭子模型
        "teddy_vhacd.urdf",          # 泰迪熊模型
        "lego/lego.urdf",            # 乐高模型
        "table/table.urdf",          # 桌子模型
        "cube.urdf",                 # 立方体
        "sphere2.urdf",              # 球体
        "cylinder.urdf",             # 圆柱体
    ]
    
    # 显示可用的URDF文件
    print("\n可用的URDF文件示例:")
    for i, example in enumerate(urdf_examples):
        full_path = os.path.join(pybullet_data_path, example)
        if os.path.exists(full_path):
            print(f"  [{i+1}] {example}")
        else:
            print(f"  [{i+1}] {example} (不存在)")
    
    # 选择几个有代表性的模型进行演示
    demo_models = [
        "cube.urdf",          # 简单的几何体
        "sphere2.urdf",       # 球体
        "table/table.urdf",   # 复杂模型
        "duck_vhacd.urdf",    # 网格模型
    ]
    
    # 创建输出目录
    output_dir = "urdf_projections"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in demo_models:
        model_path = os.path.join(pybullet_data_path, model_name)
        
        if not os.path.exists(model_path):
            print(f"\n⚠️ 模型文件不存在: {model_path}")
            continue
        
        print(f"\n{'='*40}")
        print(f"处理模型: {model_name}")
        print(f"{'='*40}")
        
        try:
            # 创建投影器
            projector = URDFProjector(model_path)
            
            # 打印模型信息
            projector.print_model_summary()
            
            # 创建不同平面的投影
            for plane in ['xy', 'xz', 'yz']:
                print(f"\n创建{plane.upper()}平面投影...")
                
                fig = projector.create_2d_projection(
                    projection_plane=plane,
                    alpha=0.7,
                    show_axes=True
                )
                
                # 保存图像
                output_filename = f"{output_dir}/{model_name.replace('/', '_').replace('.urdf', '')}_{plane}_projection.png"
                fig.savefig(output_filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"✓ 已保存: {output_filename}")
            
            # 创建综合投影图（三个平面）
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'URDF模型: {model_name} - 三视图投影', fontsize=16)
            
            for idx, plane in enumerate(['xy', 'xz', 'yz']):
                ax = axes[idx]
                ax.set_aspect('equal')
                
                # 为每个平面创建补丁
                patches_list = []
                for link_info in projector.model_info['links']:
                    for visual_info in link_info['visuals']:
                        patch = projector._create_geometry_patch(
                            visual_info, plane, None, 'black', 0.6
                        )
                        if patch:
                            patches_list.append(patch)
                
                # 添加补丁
                for patch in patches_list:
                    ax.add_patch(patch)
                
                # 设置坐标轴
                bbox = projector.model_info['bounding_box']
                if plane == 'xy':
                    min_x, min_y = bbox['min'][0], bbox['min'][1]
                    max_x, max_y = bbox['max'][0], bbox['max'][1]
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                elif plane == 'xz':
                    min_x, min_y = bbox['min'][0], bbox['min'][2]
                    max_x, max_y = bbox['max'][0], bbox['max'][2]
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Z (m)')
                else:  # yz
                    min_x, min_y = bbox['min'][1], bbox['min'][2]
                    max_x, max_y = bbox['max'][1], bbox['max'][2]
                    ax.set_xlabel('Y (m)')
                    ax.set_ylabel('Z (m)')
                
                # 添加边界余量
                margin = max((max_x - min_x) * 0.1, (max_y - min_y) * 0.1, 0.1)
                ax.set_xlim(min_x - margin, max_x + margin)
                ax.set_ylim(min_y - margin, max_y + margin)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_title(f'{plane.upper()}平面')
            
            plt.tight_layout()
            multiview_filename = f"{output_dir}/{model_name.replace('/', '_').replace('.urdf', '')}_multiview.png"
            fig.savefig(multiview_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ 已保存三视图: {multiview_filename}")
            
        except Exception as e:
            print(f"❌ 处理模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("演示完成!")
    print(f"所有投影图像已保存到: {output_dir}/")
    print("=" * 60)
    
    # 额外示例：演示如何创建matplotlib补丁
    print("\n" + "="*60)
    print("示例：在自定义图表中使用URDF投影补丁")
    print("="*60)
    
    try:
        # 使用一个简单的模型
        simple_model = os.path.join(pybullet_data_path, "cube.urdf")
        
        if os.path.exists(simple_model):
            projector = URDFProjector(simple_model)
            
            # 创建自定义图表
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_aspect('equal')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.grid(True, alpha=0.3)
            ax.set_title('在自定义图表中使用URDF投影补丁', fontsize=14)
            
            # 在不同位置添加多个立方体投影
            positions = [
                [0, 0, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [-1, 1, 0]
            ]
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (pos, color) in enumerate(zip(positions, colors)):
                patches = projector.create_matplotlib_patch(
                    position=pos,
                    projection_plane='xy',
                    color=color,
                    alpha=0.5,
                    scale=0.5
                )
                
                for patch in patches:
                    ax.add_patch(patch)
                
                # 添加标签
                ax.text(pos[0], pos[1] + 0.3, f'物体{i+1}', 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            print("✓ 已显示自定义图表示例")
        else:
            print("⚠️ 示例模型不存在，跳过自定义图表示例")
            
    except Exception as e:
        print(f"❌ 创建自定义图表时出错: {e}")


def test_pybullet_projector():
    """测试PyBullet投影器"""
    print("\n" + "="*60)
    print("测试PyBullet投影器")
    print("="*60)
    
    # 初始化PyBullet（非图形界面模式）
    try:
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载几个模型
        cube_id = p.loadURDF("cube.urdf", [0, 0, 0], globalScaling=0.5)
        sphere_id = p.loadURDF("sphere2.urdf", [1, 0, 0], globalScaling=0.3)
        
        # 创建PyBullet投影器
        projector = PyBulletURDFProjector()
        
        # 测试立方体
        print("\n1. 测试立方体投影:")
        projector.load_from_body_id(cube_id)
        
        # 获取边界框
        bbox = projector.get_2d_projection_bbox([0, 0, 0], scale=0.5)
        print(f"   边界框: {bbox}")
        
        # 创建补丁
        patches = projector.create_matplotlib_patch([0, 0, 0], scale=0.5)
        print(f"   创建的补丁数量: {len(patches)}")
        
        # 测试球体
        print("\n2. 测试球体投影:")
        projector.load_from_body_id(sphere_id)
        
        bbox = projector.get_2d_projection_bbox([1, 0, 0], scale=0.3)
        print(f"   边界框: {bbox}")
        
        patches = projector.create_matplotlib_patch([1, 0, 0], scale=0.3)
        print(f"   创建的补丁数量: {len(patches)}")
        
        p.disconnect()
        print("\n✓ PyBullet投影器测试完成")
        
    except Exception as e:
        print(f"❌ PyBullet投影器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='URDF投影工具')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--urdf', type=str, help='要处理的URDF文件路径')
    parser.add_argument('--plane', type=str, default='xy', choices=['xy', 'xz', 'yz'], 
                       help='投影平面 (默认: xy)')
    parser.add_argument('--output', type=str, default='projection.png', 
                       help='输出文件名 (默认: projection.png)')
    
    args = parser.parse_args()
    
    if args.demo:
        # 运行完整演示
        demo_urdf_projection()
        
    elif args.test:
        # 运行测试
        test_pybullet_projector()
        
    elif args.urdf:
        # 处理指定的URDF文件
        try:
            projector = URDFProjector(args.urdf)
            projector.print_model_summary()
            
            fig = projector.create_2d_projection(projection_plane=args.plane)
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n✓ 投影图像已保存: {args.output}")
            
        except Exception as e:
            print(f"❌ 处理URDF文件时出错: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        # 默认运行演示
        demo_urdf_projection()