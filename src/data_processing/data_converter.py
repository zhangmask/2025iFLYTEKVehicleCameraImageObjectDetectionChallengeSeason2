#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换模块
将XML格式标注转换为YOLO格式，支持数据集划分
"""

import os
import shutil
import random
import yaml
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from tqdm import tqdm

from .xml_parser import XMLParser, ImageAnnotation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataConverter:
    """数据格式转换器"""
    
    def __init__(self, class_mapping: Dict[str, int], output_dir: str):
        """
        初始化数据转换器
        
        Args:
            class_mapping: 类别名称到ID的映射
            output_dir: 输出目录
        """
        self.class_mapping = class_mapping
        self.output_dir = Path(output_dir)
        self.xml_parser = XMLParser(class_mapping)
        
        # 创建输出目录结构
        self._create_output_structure()
        
        logger.info(f"数据转换器初始化完成，输出目录: {self.output_dir}")
    
    def _create_output_structure(self):
        """创建YOLO数据集目录结构"""
        directories = [
            self.output_dir,
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("YOLO数据集目录结构创建完成")
    
    def convert_xml_to_yolo(self, 
                           image_dir: str, 
                           xml_dir: str, 
                           train_split: float = 0.8,
                           val_split: float = 0.2,
                           random_seed: int = 42) -> Dict:
        """
        将XML格式转换为YOLO格式
        
        Args:
            image_dir: 图像文件目录
            xml_dir: XML标注文件目录
            train_split: 训练集比例
            val_split: 验证集比例
            random_seed: 随机种子
            
        Returns:
            转换统计信息
        """
        logger.info("开始XML到YOLO格式转换...")
        
        # 设置随机种子
        random.seed(random_seed)
        
        # 解析XML标注
        annotations = self.xml_parser.parse_xml_directory(xml_dir)
        if not annotations:
            logger.error("未找到有效的XML标注文件")
            return {}
        
        # 过滤存在对应图像文件的标注
        valid_annotations = self._filter_valid_annotations(annotations, image_dir)
        logger.info(f"找到 {len(valid_annotations)} 个有效的图像-标注对")
        
        # 划分数据集
        train_annotations, val_annotations = self._split_dataset(
            valid_annotations, train_split, val_split
        )
        
        # 转换训练集
        train_stats = self._convert_annotations(
            train_annotations, image_dir, "train"
        )
        
        # 转换验证集
        val_stats = self._convert_annotations(
            val_annotations, image_dir, "val"
        )
        
        # 生成数据集配置文件
        self._generate_dataset_yaml()
        
        # 统计信息
        total_stats = {
            "total_images": len(valid_annotations),
            "train_images": len(train_annotations),
            "val_images": len(val_annotations),
            "train_objects": train_stats.get("total_objects", 0),
            "val_objects": val_stats.get("total_objects", 0),
            "class_distribution": self._merge_class_stats(
                train_stats.get("class_counts", {}),
                val_stats.get("class_counts", {})
            )
        }
        
        logger.info("XML到YOLO格式转换完成")
        self._print_conversion_stats(total_stats)
        
        return total_stats
    
    def _filter_valid_annotations(self, 
                                 annotations: List[ImageAnnotation], 
                                 image_dir: str) -> List[ImageAnnotation]:
        """
        过滤存在对应图像文件的标注
        
        Args:
            annotations: 标注列表
            image_dir: 图像目录
            
        Returns:
            有效标注列表
        """
        valid_annotations = []
        image_dir = Path(image_dir)
        
        logger.info(f"图像目录: {image_dir.absolute()}")
        logger.info(f"图像目录是否存在: {image_dir.exists()}")
        
        for annotation in annotations:
            # 查找对应的图像文件
            image_path = image_dir / annotation.filename
            logger.debug(f"检查图像文件: {image_path.absolute()}")
            
            # 尝试不同的图像扩展名
            if not image_path.exists():
                base_name = Path(annotation.filename).stem
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = image_dir / (base_name + ext)
                    if image_path.exists():
                        annotation.filename = base_name + ext
                        break
            
            if image_path.exists() and annotation.objects:
                valid_annotations.append(annotation)
                logger.debug(f"找到有效标注: {annotation.filename}")
            else:
                logger.warning(f"跳过标注文件，原因：图像不存在或无有效目标 - {annotation.filename}，路径: {image_path.absolute()}")
        
        return valid_annotations
    
    def _split_dataset(self, 
                      annotations: List[ImageAnnotation], 
                      train_split: float, 
                      val_split: float) -> Tuple[List[ImageAnnotation], List[ImageAnnotation]]:
        """
        划分数据集
        
        Args:
            annotations: 标注列表
            train_split: 训练集比例
            val_split: 验证集比例
            
        Returns:
            (训练集标注, 验证集标注)
        """
        # 确保比例和为1
        total_split = train_split + val_split
        if abs(total_split - 1.0) > 1e-6:
            logger.warning(f"训练集和验证集比例和不为1: {total_split}，自动调整")
            train_split = train_split / total_split
            val_split = val_split / total_split
        
        # 随机打乱
        shuffled_annotations = annotations.copy()
        random.shuffle(shuffled_annotations)
        
        # 计算划分点
        total_count = len(shuffled_annotations)
        train_count = int(total_count * train_split)
        
        train_annotations = shuffled_annotations[:train_count]
        val_annotations = shuffled_annotations[train_count:]
        
        logger.info(f"数据集划分完成: 训练集 {len(train_annotations)} 张，验证集 {len(val_annotations)} 张")
        
        return train_annotations, val_annotations
    
    def _convert_annotations(self, 
                           annotations: List[ImageAnnotation], 
                           image_dir: str, 
                           split: str) -> Dict:
        """
        转换标注数据
        
        Args:
            annotations: 标注列表
            image_dir: 图像目录
            split: 数据集划分 (train/val)
            
        Returns:
            转换统计信息
        """
        image_dir = Path(image_dir)
        images_output_dir = self.output_dir / "images" / split
        labels_output_dir = self.output_dir / "labels" / split
        
        stats = {
            "total_objects": 0,
            "class_counts": {name: 0 for name in self.class_mapping.keys()}
        }
        
        logger.info(f"开始转换 {split} 集，共 {len(annotations)} 张图像")
        
        for annotation in tqdm(annotations, desc=f"转换{split}集"):
            # 复制图像文件
            src_image_path = image_dir / annotation.filename
            dst_image_path = images_output_dir / annotation.filename
            
            try:
                shutil.copy2(src_image_path, dst_image_path)
            except Exception as e:
                logger.error(f"复制图像文件失败: {src_image_path} -> {dst_image_path}, 错误: {e}")
                continue
            
            # 生成YOLO格式标注文件
            label_filename = Path(annotation.filename).stem + ".txt"
            label_path = labels_output_dir / label_filename
            
            with open(label_path, 'w', encoding='utf-8') as f:
                for obj in annotation.objects:
                    # 转换为YOLO格式坐标
                    center_x, center_y, width, height = obj.bbox.to_yolo_format(
                        annotation.width, annotation.height
                    )
                    
                    # 写入标注行
                    f.write(f"{obj.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    
                    # 更新统计信息
                    stats["total_objects"] += 1
                    stats["class_counts"][obj.class_name] += 1
        
        logger.info(f"{split} 集转换完成，共 {stats['total_objects']} 个目标")
        return stats
    
    def _generate_dataset_yaml(self):
        """
        生成YOLO数据集配置文件
        """
        # 使用相对路径避免中文字符和空格问题
        dataset_config = {
            'path': '.',
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.keys())
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"数据集配置文件已生成: {yaml_path}")
    
    def _merge_class_stats(self, train_stats: Dict, val_stats: Dict) -> Dict:
        """
        合并训练集和验证集的类别统计
        
        Args:
            train_stats: 训练集统计
            val_stats: 验证集统计
            
        Returns:
            合并后的统计信息
        """
        merged_stats = {}
        all_classes = set(train_stats.keys()) | set(val_stats.keys())
        
        for class_name in all_classes:
            train_count = train_stats.get(class_name, 0)
            val_count = val_stats.get(class_name, 0)
            merged_stats[class_name] = {
                'train': train_count,
                'val': val_count,
                'total': train_count + val_count
            }
        
        return merged_stats
    
    def _print_conversion_stats(self, stats: Dict):
        """
        打印转换统计信息
        
        Args:
            stats: 统计信息字典
        """
        print("\n=== 数据转换统计信息 ===")
        print(f"总图像数量: {stats['total_images']}")
        print(f"训练集图像: {stats['train_images']}")
        print(f"验证集图像: {stats['val_images']}")
        print(f"训练集目标: {stats['train_objects']}")
        print(f"验证集目标: {stats['val_objects']}")
        print(f"总目标数量: {stats['train_objects'] + stats['val_objects']}")
        
        print("\n类别分布:")
        for class_name, counts in stats['class_distribution'].items():
            print(f"  {class_name}:")
            print(f"    训练集: {counts['train']}")
            print(f"    验证集: {counts['val']}")
            print(f"    总计: {counts['total']}")
    
    def validate_conversion(self) -> bool:
        """
        验证转换结果
        
        Returns:
            验证是否通过
        """
        logger.info("开始验证转换结果...")
        
        # 检查目录结构
        required_dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val"
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                logger.error(f"缺少必要目录: {directory}")
                return False
        
        # 检查数据集配置文件
        yaml_path = self.output_dir / "dataset.yaml"
        if not yaml_path.exists():
            logger.error(f"缺少数据集配置文件: {yaml_path}")
            return False
        
        # 检查图像和标注文件数量是否匹配
        for split in ['train', 'val']:
            images_dir = self.output_dir / "images" / split
            labels_dir = self.output_dir / "labels" / split
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt"))
            
            if len(image_files) != len(label_files):
                logger.warning(f"{split}集图像和标注文件数量不匹配: {len(image_files)} vs {len(label_files)}")
        
        logger.info("转换结果验证完成")
        return True


def main():
    """测试函数"""
    # 类别映射
    class_mapping = {
        "car": 0,
        "truck": 1,
        "bike": 2,
        "human": 3
    }
    
    # 创建转换器
    converter = DataConverter(
        class_mapping=class_mapping,
        output_dir="outputs/yolo_data"
    )
    
    # 执行转换
    stats = converter.convert_xml_to_yolo(
        image_dir="train/train",
        xml_dir="train_label/train_label",
        train_split=0.8,
        val_split=0.2
    )
    
    # 验证转换结果
    converter.validate_conversion()
    
    print("\n数据转换完成！")

if __name__ == "__main__":
    main()