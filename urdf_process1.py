"""
URDF模型投影模块
功能：读取URDF文件，生成模型的2D投影图像
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon
from matplotlib.collections import PatchCollection
import xml.etree.ElementTree as ET
import os
import tempfile
import warnings
from typing import Dict, List, Tuple, Optional, Union
import pybullet as p
import pybullet_data


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
            raise FileNotFoundError(f"URDF文件不存在: {urdf_path}")
        
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # 解析模型基本信息
            self.model_info = {
                'name': root.get('name', 'unknown'),
                'links': self._parse_links(root),
                'joints': self._parse_joints(root),
                'materials': self._parse_materials(root),
                'bounding_box': self._calculate_bounding_box(root)
            }
            
            print(f"✓ 成功加载URDF模型: {self.model_info['name']}")
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
            scale_str = mesh.get('scale', '1 1 1')
            scale = [float(x) * self.scaling for x in scale_str.split()]
            
            # 对于网格文件，我们只能获取大致尺寸
            return {
                'type': 'mesh',
                'filename': filename,
                'scale': scale,
                'pose': self._parse_pose(elem.find('origin')),
                'material': self._parse_material(elem.find('material'))
            }
        
        return None
    
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
                           alpha: float = 0.8) -> plt.Figure:
        """
        创建模型的2D投影图像
        
        Args:
            resolution: 图像分辨率（像素）
            projection_plane: 投影平面 ('xy', 'xz', 'yz')
            fill_color: 填充颜色，如果为None则使用模型材质颜色
            edge_color: 边缘颜色
            alpha: 透明度
            
        Returns:
            matplotlib Figure对象
        """
        if not self.model_info:
            raise ValueError("请先加载URDF模型")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        
        # 设置坐标轴
        plane_labels = {
            'xy': ('X', 'Y'),
            'xz': ('X', 'Z'),
            'yz': ('Y', 'Z')
        }
        
        if projection_plane not in plane_labels:
            raise ValueError(f"投影平面必须是: {list(plane_labels.keys())}")
        
        x_label, y_label = plane_labels[projection_plane]
        ax.set_xlabel(f'{x_label} (m)', fontsize=12)
        ax.set_ylabel(f'{y_label} (m)', fontsize=12)
        ax.set_title(f'URDF模型{projection_plane.upper()}平面投影', fontsize=14)
        
        # 绘制所有链接的投影
        for link_info in self.model_info['links']:
            for visual_info in link_info['visuals']:
                self._plot_geometry_projection(ax, visual_info, projection_plane, 
                                             fill_color, edge_color, alpha)
        
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
        margin = max((max_x - min_x) * 0.1, (max_y - min_y) * 0.1, 0.1)
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return fig
    
    def _plot_geometry_projection(self, ax, geom_info: Dict, projection_plane: str,
                                fill_color: Optional[str], edge_color: str, alpha: float):
        """绘制几何体的投影"""
        pose = geom_info['pose']
        x, y, z = pose['xyz']
        
        # 确定要使用的颜色
        if fill_color is None and geom_info.get('material'):
            rgba = geom_info['material'].get('rgba', [0.5, 0.5, 0.5, 1.0])
            facecolor = rgba[:3]  # RGB
            if len(rgba) > 3:
                alpha = rgba[3] * alpha
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
                    linewidth=2, edgecolor=edge_color, 
                    facecolor=facecolor, alpha=alpha,
                    angle=np.degrees(pose['rpy'][2])  # 考虑Z轴旋转
                )
                ax.add_patch(rect)
                
            elif geom_type == 'sphere':
                radius = geom_info['radius']
                circle = patches.Circle(
                    (x, y), radius,
                    linewidth=2, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                ax.add_patch(circle)
                
            elif geom_type == 'cylinder':
                radius = geom_info['radius']
                circle = patches.Circle(
                    (x, y), radius,
                    linewidth=2, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                ax.add_patch(circle)
                
            elif geom_type == 'mesh':
                # 网格物体简化为矩形
                scale = geom_info.get('scale', [1, 1, 1])
                dx, dy = scale[0] * 0.5, scale[1] * 0.5  # 估计尺寸
                rect = patches.Rectangle(
                    (x - dx, y - dy), dx*2, dy*2,
                    linewidth=2, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha,
                    hatch='//'  # 添加网格线表示
                )
                ax.add_patch(rect)
        
        elif projection_plane == 'xz':
            # XZ平面投影（从侧面看）
            if geom_type == 'box':
                dx, dy, dz = geom_info['dimensions']
                rect = patches.Rectangle(
                    (x - dx/2, z - dz/2), dx, dz,
                    linewidth=2, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                ax.add_patch(rect)
                
            elif geom_type == 'sphere':
                radius = geom_info['radius']
                circle = patches.Circle(
                    (x, z), radius,
                    linewidth=2, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                ax.add_patch(circle)
                
            elif geom_type == 'cylinder':
                radius = geom_info['radius']
                length = geom_info['length']
                rect = patches.Rectangle(
                    (x - radius, z - length/2), radius*2, length,
                    linewidth=2, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                ax.add_patch(rect)
        
        else:  # yz
            if geom_type == 'box':
                dx, dy, dz = geom_info['dimensions']
                rect = patches.Rectangle(
                    (y - dy/2, z - dz/2), dy, dz,
                    linewidth=2, edgecolor=edge_color,
                    facecolor=facecolor, alpha=alpha
                )
                ax.add_patch(rect)
    
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
                    rgba = visual_info['material'].get('rgba', [0.5, 0.5, 0.5, 1.0])
                    facecolor = rgba[:3]
                    patch_alpha = rgba[3] if len(rgba) > 3 else alpha
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
                            linewidth=1.5, edgecolor='black',
                            facecolor=facecolor, alpha=patch_alpha,
                            angle=np.degrees(abs_yaw)
                        )
                        patches_list.append(rect)
                        
                    elif geom_type == 'sphere':
                        radius = visual_info['radius'] * scale
                        circle = patches.Circle(
                            (abs_x, abs_y), radius,
                            linewidth=1.5, edgecolor='black',
                            facecolor=facecolor, alpha=patch_alpha
                        )
                        patches_list.append(circle)
                        
                    elif geom_type == 'cylinder':
                        radius = visual_info['radius'] * scale
                        circle = patches.Circle(
                            (abs_x, abs_y), radius,
                            linewidth=1.5, edgecolor='black',
                            facecolor=facecolor, alpha=patch_alpha
                        )
                        patches_list.append(circle)
        
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


class PyBulletURDFProjector:
    """基于PyBullet的URDF投影器（直接使用PyBullet加载模型）"""
    
    def __init__(self):
        self.body_id = None
        self.visual_shape_data = None
    
    def load_from_body_id(self, body_id: int):
        """
        从PyBullet中的物体ID加载模型信息
        
        Args:
            body_id: PyBullet物体ID
        """
        self.body_id = body_id
        
        # 获取视觉形状数据
        self.visual_shape_data = p.getVisualShapeData(body_id)
        
        if not self.visual_shape_data:
            raise ValueError(f"物体 {body_id} 没有视觉形状数据")
        
        print(f"✓ 从PyBullet加载物体 {body_id}")
        print(f"  包含 {len(self.visual_shape_data)} 个视觉形状")
    
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
        if not self.visual_shape_data:
            raise ValueError("请先加载模型数据")
        
        patches_list = []
        
        for shape_data in self.visual_shape_data:
            shape_type = shape_data[2]  # 形状类型
            dimensions = list(shape_data[3])  # 尺寸
            pos = list(shape_data[5])  # 位置（相对）
            orn = list(shape_data[6])  # 方向（四元数，相对）
            
            # 获取颜色
            rgba_color = list(shape_data[7])  # RGBA颜色
            
            # 应用绝对位置和方向
            if orientation:
                # 合并方向（这里简化处理）
                final_orn = orientation
            else:
                final_orn = [0, 0, 0, 1]  # 无旋转
            
            # 应用缩放
            dimensions = [d * scale for d in dimensions]
            pos = [p * scale for p in pos]
            
            # 计算绝对位置
            abs_pos = [
                position[0] + pos[0],
                position[1] + pos[1],
                position[2] + pos[2]
            ]
            
            # 确定颜色
            if color:
                facecolor = color
            else:
                facecolor = (rgba_color[0], rgba_color[1], rgba_color[2])
            
            patch_alpha = rgba_color[3] if len(rgba_color) > 3 else alpha
            
            if projection_plane == 'xy':
                if shape_type == p.GEOM_BOX:
                    # 立方体
                    dx, dy, dz = dimensions[0], dimensions[1], dimensions[2]
                    patch = patches.Rectangle(
                        (abs_pos[0] - dx/2, abs_pos[1] - dy/2), dx, dy,
                        linewidth=1.5, edgecolor='black',
                        facecolor=facecolor, alpha=patch_alpha
                    )
                    patches_list.append(patch)
                    
                elif shape_type == p.GEOM_SPHERE:
                    # 球体
                    radius = dimensions[0]
                    patch = patches.Circle(
                        (abs_pos[0], abs_pos[1]), radius,
                        linewidth=1.5, edgecolor='black',
                        facecolor=facecolor, alpha=patch_alpha
                    )
                    patches_list.append(patch)
                    
                elif shape_type == p.GEOM_CYLINDER:
                    # 圆柱体
                    radius = dimensions[0]
                    patch = patches.Circle(
                        (abs_pos[0], abs_pos[1]), radius,
                        linewidth=1.5, edgecolor='black',
                        facecolor=facecolor, alpha=patch_alpha
                    )
                    patches_list.append(patch)
                    
                elif shape_type == p.GEOM_CAPSULE:
                    # 胶囊体
                    radius = dimensions[0]
                    patch = patches.Circle(
                        (abs_pos[0], abs_pos[1]), radius,
                        linewidth=1.5, edgecolor='black',
                        facecolor=facecolor, alpha=patch_alpha
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
        if not self.visual_shape_data:
            raise ValueError("请先加载模型数据")
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = -float('inf'), -float('inf')
        
        for shape_data in self.visual_shape_data:
            shape_type = shape_data[2]
            dimensions = list(shape_data[3])
            pos = list(shape_data[5])
            
            # 应用缩放
            dimensions = [d * scale for d in dimensions]
            pos = [p * scale for p in pos]
            
            # 计算绝对位置
            abs_pos = [
                position[0] + pos[0],
                position[1] + pos[1],
                position[2] + pos[2]
            ]
            
            if projection_plane == 'xy':
                if shape_type == p.GEOM_BOX:
                    dx, dy = dimensions[0], dimensions[1]
                    min_x = min(min_x, abs_pos[0] - dx/2)
                    max_x = max(max_x, abs_pos[0] + dx/2)
                    min_y = min(min_y, abs_pos[1] - dy/2)
                    max_y = max(max_y, abs_pos[1] + dy/2)
                    
                elif shape_type in [p.GEOM_SPHERE, p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
                    radius = dimensions[0]
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
    """演示URDF投影功能 - 使用PyBullet内置模型，只生成Z轴从上往下的投影图"""
    
    print("=" * 60)
    print("PyBullet URDF投影演示 - Z轴从上往下投影（XY平面）")
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
        "r2d2.urdf",                 # R2D2机器人
    ]
    
    # 显示可用的URDF文件
    print("\n可用的URDF文件示例:")
    available_models = []
    for i, example in enumerate(urdf_examples):
        full_path = os.path.join(pybullet_data_path, example)
        if os.path.exists(full_path):
            available_models.append(example)
            print(f"  [{len(available_models)}] {example}")
    
    if not available_models:
        print("❌ 没有找到可用的URDF文件")
        return
    
    # 创建输出目录
    output_dir = "urdf_projections"
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择几个有代表性的模型进行演示（Z轴投影）
    demo_models = [
        "cube.urdf",          # 简单的几何体
        "sphere2.urdf",       # 球体
        "cylinder.urdf",      # 圆柱体
        "table/table.urdf",   # 复杂模型
        "racecar/racecar.urdf", # 车辆模型
        "duck_vhacd.urdf",    # 网格模型
    ]
    
    # 过滤出实际存在的模型
    demo_models = [m for m in demo_models if m in available_models]
    
    # 创建一个大图，显示所有模型的投影对比
    print(f"\n创建所有模型的Z轴投影对比图...")
    
    num_models = len(demo_models)
    if num_models > 0:
        # 计算网格布局
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        fig.suptitle('URDF模型Z轴投影（从上往下看 - XY平面）', fontsize=16, y=0.98)
        
        # 如果只有一个子图，确保axes是列表
        if num_models == 1:
            axes = np.array([axes])
        
        # 展平axes数组以便遍历
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for idx, model_name in enumerate(demo_models):
            if idx >= len(axes_flat):
                break
                
            model_path = os.path.join(pybullet_data_path, model_name)
            ax = axes_flat[idx]
            
            try:
                # 创建投影器
                projector = URDFProjector(model_path)
                
                # 创建补丁列表
                patches_list = []
                for link_info in projector.model_info['links']:
                    for visual_info in link_info['visuals']:
                        patch = projector._create_geometry_patch(
                            visual_info, 'xy', None, 'black', 0.7
                        )
                        if patch:
                            patches_list.append(patch)
                
                # 添加所有补丁到子图
                for patch in patches_list:
                    ax.add_patch(patch)
                
                # 设置子图属性
                ax.set_aspect('equal')
                
                # 获取边界框并设置显示范围
                bbox = projector.model_info['bounding_box']
                min_x, min_y = bbox['min'][0], bbox['min'][1]
                max_x, max_y = bbox['max'][0], bbox['max'][1]
                
                # 添加边界余量
                margin_x = (max_x - min_x) * 0.1
                margin_y = (max_y - min_y) * 0.1
                margin = max(margin_x, margin_y, 0.1)
                
                ax.set_xlim(min_x - margin, max_x + margin)
                ax.set_ylim(min_y - margin, max_y + margin)
                
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_xlabel('X (m)', fontsize=10)
                ax.set_ylabel('Y (m)', fontsize=10)
                
                # 设置标题（显示模型名称）
                title = model_name.split('/')[-1].replace('.urdf', '')
                ax.set_title(f'{title}', fontsize=12, pad=10)
                
                # 添加模型信息
                info_text = f"链接: {len(projector.model_info['links'])}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                               alpha=0.8, edgecolor='gray'))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'加载失败:\n{str(e)[:30]}...',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
                ax.set_title(f'{model_name} (错误)', fontsize=10, color='red')
                ax.set_facecolor('#ffe6e6')
        
        # 隐藏多余的子图
        for idx in range(len(demo_models), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        comparison_filename = f"{output_dir}/all_models_z_projection_comparison.png"
        fig.savefig(comparison_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 已保存对比图: {comparison_filename}")
    
    # 单独处理每个模型，生成详细的Z轴投影图
    print(f"\n单独处理每个模型的Z轴投影...")
    
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
            
            # 只创建XY平面的投影（Z轴从上往下看）
            print(f"\n创建Z轴投影（XY平面）...")
            
            fig = projector.create_2d_projection(
                projection_plane='xy',  # 只生成XY平面投影
                alpha=0.7,
                show_axes=True
            )
            
            # 添加额外信息说明
            ax = fig.axes[0]
            ax.text(0.02, 0.02, "投影方向: Z轴从上往下（XY平面）", 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
            
            # 保存图像
            model_basename = model_name.split('/')[-1].replace('.urdf', '')
            output_filename = f"{output_dir}/{model_basename}_z_projection.png"
            fig.savefig(output_filename, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ 已保存Z轴投影图: {output_filename}")
            
            # 创建简单的示意图，展示投影原理
            print(f"创建投影原理示意图...")
            
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle(f'Z轴投影原理示意图 - {model_basename}', fontsize=14)
            
            # 左侧：3D示意图
            ax1.set_aspect('equal')
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            
            # 绘制3D物体在XY平面的投影
            if 'cube' in model_name:
                # 立方体
                rect = patches.Rectangle((-0.5, -0.5), 1, 1,
                                       facecolor='blue', alpha=0.6,
                                       edgecolor='darkblue', linewidth=2)
                ax1.add_patch(rect)
                ax1.text(0, 0, '立方体', ha='center', va='center', fontsize=12, color='white')
                
            elif 'sphere' in model_name:
                # 球体
                circle = patches.Circle((0, 0), 0.5,
                                       facecolor='red', alpha=0.6,
                                       edgecolor='darkred', linewidth=2)
                ax1.add_patch(circle)
                ax1.text(0, 0, '球体', ha='center', va='center', fontsize=12, color='white')
                
            elif 'cylinder' in model_name:
                # 圆柱体
                ellipse = patches.Ellipse((0, 0), 1, 0.6,
                                         facecolor='green', alpha=0.6,
                                         edgecolor='darkgreen', linewidth=2)
                ax1.add_patch(ellipse)
                ax1.text(0, 0, '圆柱体', ha='center', va='center', fontsize=12, color='white')
            
            # 添加Z轴箭头表示投影方向
            ax1.arrow(1.2, 1.2, 0, -0.3, head_width=0.05, head_length=0.1, fc='black', ec='black')
            ax1.text(1.25, 1.4, 'Z轴投影方向', fontsize=10)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('3D物体的XY平面投影', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 右侧：实际的投影结果
            ax2.set_aspect('equal')
            
            # 复制实际的投影到右侧
            for link_info in projector.model_info['links']:
                for visual_info in link_info['visuals']:
                    patch = projector._create_geometry_patch(
                        visual_info, 'xy', None, 'black', 0.7
                    )
                    if patch:
                        ax2.add_patch(patch)
            
            # 设置显示范围
            bbox = projector.model_info['bounding_box']
            min_x, min_y = bbox['min'][0], bbox['min'][1]
            max_x, max_y = bbox['max'][0], bbox['max'][1]
            
            margin_x = (max_x - min_x) * 0.1
            margin_y = (max_y - min_y) * 0.1
            margin = max(margin_x, margin_y, 0.1)
            
            ax2.set_xlim(min_x - margin, max_x + margin)
            ax2.set_ylim(min_y - margin, max_y + margin)
            
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('实际URDF模型的Z轴投影', fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            diagram_filename = f"{output_dir}/{model_basename}_projection_diagram.png"
            fig2.savefig(diagram_filename, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
            print(f"✓ 已保存投影原理图: {diagram_filename}")
            
        except Exception as e:
            print(f"❌ 处理模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 创建使用示例图
    print(f"\n{'='*60}")
    print("创建Matplotlib补丁使用示例...")
    print("="*60)
    
    try:
        # 使用立方体模型演示如何在自定义图表中使用投影
        cube_model = os.path.join(pybullet_data_path, "cube.urdf")
        
        if os.path.exists(cube_model):
            projector = URDFProjector(cube_model)
            
            # 创建场景图
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_aspect('equal')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title('在Matplotlib场景中使用URDF投影补丁', fontsize=14)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 添加背景网格
            for x in range(-3, 4):
                ax.axvline(x=x, color='gray', alpha=0.2, linestyle='-', linewidth=0.5)
            for y in range(-3, 4):
                ax.axhline(y=y, color='gray', alpha=0.2, linestyle='-', linewidth=0.5)
            
            # 在不同位置添加多个物体投影
            positions = [
                [-2, -2, 0],  # 左下
                [0, -2, 0],   # 下中
                [2, -2, 0],   # 右下
                [-2, 0, 0],   # 左中
                [0, 0, 0],    # 中心
                [2, 0, 0],    # 右中
                [-2, 2, 0],   # 左上
                [0, 2, 0],    # 上中
                [2, 2, 0],    # 右上
            ]
            
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', 
                     '#dda0dd', '#98d8c8', '#f7dc6f', '#bb8fce']
            
            labels = ['物体A', '物体B', '物体C', '物体D', '物体E', 
                     '物体F', '物体G', '物体H', '物体I']
            
            for i, (pos, color, label) in enumerate(zip(positions, colors, labels)):
                # 创建补丁（使用不同缩放）
                scale = 0.3 + (i % 3) * 0.1  # 不同大小
                
                patches = projector.create_matplotlib_patch(
                    position=pos,
                    projection_plane='xy',
                    color=color,
                    alpha=0.6,
                    scale=scale
                )
                
                for patch in patches:
                    ax.add_patch(patch)
                
                # 添加标签和连接线
                label_pos = [pos[0], pos[1] + scale * 0.8]
                ax.annotate(label, xy=(pos[0], pos[1]), xytext=label_pos,
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                           ha='center', va='center', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # 显示坐标
                coord_text = f"({pos[0]}, {pos[1]})"
                ax.text(pos[0], pos[1] - scale * 0.8, coord_text,
                       ha='center', va='center', fontsize=8, color='gray')
            
            # 添加图例说明
            info_text = """
            示例说明：
            • 每个彩色方块是立方体的Z轴投影
            • 不同颜色和大小表示不同的物体实例
            • 标签显示物体名称
            • 坐标显示物体位置
            """
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                           alpha=0.9, edgecolor='gray'))
            
            plt.tight_layout()
            example_filename = f"{output_dir}/matplotlib_patch_example.png"
            fig.savefig(example_filename, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"✓ 已保存Matplotlib补丁示例: {example_filename}")
        else:
            print("⚠️ 立方体模型不存在，跳过示例创建")
            
    except Exception as e:
        print(f"❌ 创建示例图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("演示完成!")
    print(f"所有投影图像已保存到: {output_dir}/")
    print("=" * 60)
   

if __name__ == "__main__":
    demo_urdf_projection()