#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML标注解析模块
支持PASCAL VOC格式的XML标注文件解析
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """边界框数据类"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    
    @property
    def center_x(self) -> float:
        """计算中心点x坐标"""
        return (self.xmin + self.xmax) / 2
    
    @property
    def center_y(self) -> float:
        """计算中心点y坐标"""
        return (self.ymin + self.ymax) / 2
    
    @property
    def width(self) -> float:
        """计算边界框宽度"""
        return self.xmax - self.xmin
    
    @property
    def height(self) -> float:
        """计算边界框高度"""
        return self.ymax - self.ymin
    
    def normalize(self, img_width: int, img_height: int) -> 'BoundingBox':
        """归一化边界框坐标到[0,1]范围"""
        return BoundingBox(
            xmin=self.xmin / img_width,
            ymin=self.ymin / img_height,
            xmax=self.xmax / img_width,
            ymax=self.ymax / img_height
        )
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """转换为YOLO格式 (center_x, center_y, width, height) 归一化坐标"""
        center_x = self.center_x / img_width
        center_y = self.center_y / img_height
        width = self.width / img_width
        height = self.height / img_height
        return center_x, center_y, width, height

@dataclass
class ObjectAnnotation:
    """目标标注数据类"""
    class_name: str
    class_id: int
    bbox: BoundingBox
    difficult: bool = False
    truncated: bool = False
    pose: str = "Unspecified"

@dataclass
class ImageAnnotation:
    """图像标注数据类"""
    filename: str
    width: int
    height: int
    depth: int
    objects: List[ObjectAnnotation]
    folder: str = ""
    path: str = ""

class XMLParser:
    """XML标注解析器"""
    
    def __init__(self, class_mapping: Optional[Dict[str, int]] = None):
        """
        初始化XML解析器
        
        Args:
            class_mapping: 类别名称到ID的映射字典
        """
        self.class_mapping = class_mapping or {
            "car": 0,
            "truck": 1,
            "bike": 2,
            "human": 3
        }
        self.valid_classes = set(self.class_mapping.keys())
        logger.info(f"初始化XML解析器，支持类别: {list(self.valid_classes)}")
    
    def parse_xml_file(self, xml_path: str) -> Optional[ImageAnnotation]:
        """
        解析单个XML标注文件
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            ImageAnnotation对象，解析失败返回None
        """
        try:
            if not os.path.exists(xml_path):
                logger.error(f"XML文件不存在: {xml_path}")
                return None
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 解析图像基本信息
            filename = root.find('filename').text if root.find('filename') is not None else ""
            folder = root.find('folder').text if root.find('folder') is not None else ""
            path = root.find('path').text if root.find('path') is not None else ""
            
            # 解析图像尺寸
            size_elem = root.find('size')
            if size_elem is None:
                logger.error(f"XML文件缺少size信息: {xml_path}")
                return None
            
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            depth = int(size_elem.find('depth').text) if size_elem.find('depth') is not None else 3
            
            # 解析目标对象
            objects = []
            for obj_elem in root.findall('object'):
                obj_annotation = self._parse_object(obj_elem)
                if obj_annotation is not None:
                    objects.append(obj_annotation)
            
            return ImageAnnotation(
                filename=filename,
                width=width,
                height=height,
                depth=depth,
                objects=objects,
                folder=folder,
                path=path
            )
            
        except ET.ParseError as e:
            logger.error(f"XML解析错误 {xml_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"解析XML文件时发生错误 {xml_path}: {e}")
            return None
    
    def _parse_object(self, obj_elem) -> Optional[ObjectAnnotation]:
        """
        解析单个目标对象
        
        Args:
            obj_elem: XML对象元素
            
        Returns:
            ObjectAnnotation对象，解析失败返回None
        """
        try:
            # 获取类别名称
            name_elem = obj_elem.find('name')
            if name_elem is None:
                logger.warning("目标对象缺少name字段")
                return None
            
            class_name = name_elem.text.lower().strip()
            
            # 检查类别是否有效
            if class_name not in self.valid_classes:
                logger.warning(f"未知类别: {class_name}，跳过该目标")
                return None
            
            class_id = self.class_mapping[class_name]
            
            # 解析边界框
            bndbox_elem = obj_elem.find('bndbox')
            if bndbox_elem is None:
                logger.warning("目标对象缺少bndbox字段")
                return None
            
            xmin = float(bndbox_elem.find('xmin').text)
            ymin = float(bndbox_elem.find('ymin').text)
            xmax = float(bndbox_elem.find('xmax').text)
            ymax = float(bndbox_elem.find('ymax').text)
            
            # 验证边界框坐标
            if xmin >= xmax or ymin >= ymax:
                logger.warning(f"无效的边界框坐标: ({xmin}, {ymin}, {xmax}, {ymax})")
                return None
            
            bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            
            # 解析其他属性
            difficult = obj_elem.find('difficult')
            difficult = bool(int(difficult.text)) if difficult is not None else False
            
            truncated = obj_elem.find('truncated')
            truncated = bool(int(truncated.text)) if truncated is not None else False
            
            pose = obj_elem.find('pose')
            pose = pose.text if pose is not None else "Unspecified"
            
            return ObjectAnnotation(
                class_name=class_name,
                class_id=class_id,
                bbox=bbox,
                difficult=difficult,
                truncated=truncated,
                pose=pose
            )
            
        except Exception as e:
            logger.error(f"解析目标对象时发生错误: {e}")
            return None
    
    def parse_xml_directory(self, xml_dir: str) -> List[ImageAnnotation]:
        """
        批量解析XML标注文件目录
        
        Args:
            xml_dir: XML文件目录路径
            
        Returns:
            ImageAnnotation对象列表
        """
        annotations = []
        
        if not os.path.exists(xml_dir):
            logger.error(f"XML目录不存在: {xml_dir}")
            return annotations
        
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        logger.info(f"找到 {len(xml_files)} 个XML文件")
        
        for xml_file in xml_files:
            xml_path = os.path.join(xml_dir, xml_file)
            annotation = self.parse_xml_file(xml_path)
            if annotation is not None:
                annotations.append(annotation)
        
        logger.info(f"成功解析 {len(annotations)} 个XML文件")
        return annotations
    
    def get_statistics(self, annotations: List[ImageAnnotation]) -> Dict:
        """
        获取数据集统计信息
        
        Args:
            annotations: 标注数据列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_images': len(annotations),
            'total_objects': 0,
            'class_counts': {class_name: 0 for class_name in self.valid_classes},
            'image_sizes': [],
            'bbox_sizes': []
        }
        
        for annotation in annotations:
            stats['total_objects'] += len(annotation.objects)
            stats['image_sizes'].append((annotation.width, annotation.height))
            
            for obj in annotation.objects:
                stats['class_counts'][obj.class_name] += 1
                stats['bbox_sizes'].append((obj.bbox.width, obj.bbox.height))
        
        return stats
    
    def print_statistics(self, annotations: List[ImageAnnotation]):
        """
        打印数据集统计信息
        
        Args:
            annotations: 标注数据列表
        """
        stats = self.get_statistics(annotations)
        
        print("\n=== 数据集统计信息 ===")
        print(f"总图像数量: {stats['total_images']}")
        print(f"总目标数量: {stats['total_objects']}")
        print(f"平均每张图像目标数: {stats['total_objects'] / max(stats['total_images'], 1):.2f}")
        
        print("\n类别分布:")
        for class_name, count in stats['class_counts'].items():
            percentage = count / max(stats['total_objects'], 1) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        if stats['image_sizes']:
            widths, heights = zip(*stats['image_sizes'])
            print(f"\n图像尺寸统计:")
            print(f"  宽度范围: {min(widths)} - {max(widths)}")
            print(f"  高度范围: {min(heights)} - {max(heights)}")
            print(f"  平均尺寸: {sum(widths)/len(widths):.0f} x {sum(heights)/len(heights):.0f}")


def main():
    """测试函数"""
    # 创建解析器
    parser = XMLParser()
    
    # 解析XML文件目录
    xml_dir = "train_label/train_label"
    annotations = parser.parse_xml_directory(xml_dir)
    
    # 打印统计信息
    parser.print_statistics(annotations)
    
    # 示例：获取第一个标注的详细信息
    if annotations:
        first_annotation = annotations[0]
        print(f"\n=== 示例标注信息 ===")
        print(f"文件名: {first_annotation.filename}")
        print(f"图像尺寸: {first_annotation.width} x {first_annotation.height}")
        print(f"目标数量: {len(first_annotation.objects)}")
        
        for i, obj in enumerate(first_annotation.objects):
            print(f"  目标 {i+1}: {obj.class_name} (ID: {obj.class_id})")
            print(f"    边界框: ({obj.bbox.xmin}, {obj.bbox.ymin}, {obj.bbox.xmax}, {obj.bbox.ymax})")
            yolo_coords = obj.bbox.to_yolo_format(first_annotation.width, first_annotation.height)
            print(f"    YOLO格式: {yolo_coords}")

if __name__ == "__main__":
    main()