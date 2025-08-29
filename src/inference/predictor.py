#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理引擎
支持批量图像处理、GPU加速和结果后处理
"""

import os
import cv2
import yaml
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class Detection:
    """
    检测结果数据类
    """
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (left, top, right, bottom)
    class_id: int

class YOLOv8Predictor:
    """
    YOLOv8模型推理器
    """
    
    def __init__(self, config_path: str, model_path: str):
        """
        初始化推理器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型文件路径
        """
        self.config_path = config_path
        self.model_path = model_path
        self.config = self._load_config()
        self.model = None
        self.device = self._get_device()
        self.class_names = self.config['classes']['names']
        self._setup_logging()
        self._load_model()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"无法加载配置文件 {self.config_path}: {e}")
    
    def _get_device(self) -> str:
        """
        获取推理设备
        
        Returns:
            设备名称 ('cuda' 或 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"使用GPU进行推理: {gpu_name}")
        else:
            device = 'cpu'
            print("使用CPU进行推理")
        return device
    
    def _setup_logging(self):
        """
        设置日志记录
        """
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'inference.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """
        加载YOLO模型
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.logger.info(f"加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置模型到指定设备
            self.model.to(self.device)
            
            self.logger.info(f"模型加载完成，使用设备: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像数组
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 转换颜色空间 (BGR -> RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像预处理失败 {image_path}: {e}")
            raise
    
    def predict_single(self, image_path: str, conf_threshold: float = None, 
                      iou_threshold: float = None) -> List[Detection]:
        """
        对单张图像进行预测
        
        Args:
            image_path: 图像文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            检测结果列表
        """
        try:
            # 使用配置文件中的默认阈值
            if conf_threshold is None:
                conf_threshold = self.config['inference']['confidence_threshold']
            if iou_threshold is None:
                iou_threshold = self.config['inference']['iou_threshold']
            
            # 预处理图像
            image = self.preprocess_image(image_path)
            
            # 执行推理
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # 解析结果
            detections = self._parse_results(results[0], image.shape)
            
            self.logger.debug(f"图像 {image_path} 检测到 {len(detections)} 个目标")
            return detections
            
        except Exception as e:
            self.logger.error(f"单张图像预测失败 {image_path}: {e}")
            raise
    
    def predict_batch(self, image_paths: List[str], batch_size: int = None,
                     conf_threshold: float = None, iou_threshold: float = None) -> Dict[str, List[Detection]]:
        """
        批量预测图像
        
        Args:
            image_paths: 图像文件路径列表
            batch_size: 批处理大小
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            图像路径到检测结果的映射字典
        """
        try:
            # 使用配置文件中的默认值
            if batch_size is None:
                batch_size = self.config['inference']['batch_size']
            if conf_threshold is None:
                conf_threshold = self.config['inference']['confidence_threshold']
            if iou_threshold is None:
                iou_threshold = self.config['inference']['iou_threshold']
            
            results_dict = {}
            total_images = len(image_paths)
            
            self.logger.info(f"开始批量预测 {total_images} 张图像，批大小: {batch_size}")
            
            # 分批处理
            for i in range(0, total_images, batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                self.logger.info(f"处理批次 {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
                
                # 批量推理
                batch_results = self.model.predict(
                    source=batch_paths,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # 解析每张图像的结果
                for j, result in enumerate(batch_results):
                    image_path = batch_paths[j]
                    
                    # 读取图像尺寸
                    image = cv2.imread(image_path)
                    if image is not None:
                        detections = self._parse_results(result, image.shape)
                        results_dict[image_path] = detections
                    else:
                        self.logger.warning(f"无法读取图像: {image_path}")
                        results_dict[image_path] = []
            
            self.logger.info(f"批量预测完成，共处理 {len(results_dict)} 张图像")
            return results_dict
            
        except Exception as e:
            self.logger.error(f"批量预测失败: {e}")
            raise
    
    def _parse_results(self, result, image_shape: Tuple[int, int, int]) -> List[Detection]:
        """
        解析YOLO推理结果
        
        Args:
            result: YOLO推理结果
            image_shape: 图像形状 (height, width, channels)
            
        Returns:
            检测结果列表
        """
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
            confidences = result.boxes.conf.cpu().numpy()  # 置信度
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 类别ID
            
            for i in range(len(boxes)):
                # 获取边界框坐标
                left, top, right, bottom = boxes[i]
                
                # 确保坐标在图像范围内
                left = max(0, min(left, image_shape[1]))
                top = max(0, min(top, image_shape[0]))
                right = max(0, min(right, image_shape[1]))
                bottom = max(0, min(bottom, image_shape[0]))
                
                # 获取类别信息
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # 确保类别ID有效
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    
                    detection = Detection(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=(float(left), float(top), float(right), float(bottom)),
                        class_id=int(class_id)
                    )
                    detections.append(detection)
        
        return detections
    
    def predict_folder(self, folder_path: str, output_dir: str = None,
                      conf_threshold: float = None, iou_threshold: float = None) -> Dict[str, List[Detection]]:
        """
        预测文件夹中的所有图像
        
        Args:
            folder_path: 图像文件夹路径
            output_dir: 输出目录（可选）
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            图像路径到检测结果的映射字典
        """
        try:
            # 获取所有图像文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_paths = []
            
            folder_path = Path(folder_path)
            for ext in image_extensions:
                image_paths.extend(folder_path.glob(f'*{ext}'))
                image_paths.extend(folder_path.glob(f'*{ext.upper()}'))
            
            image_paths = [str(p) for p in image_paths]
            
            if not image_paths:
                self.logger.warning(f"文件夹中未找到图像文件: {folder_path}")
                return {}
            
            self.logger.info(f"在文件夹 {folder_path} 中找到 {len(image_paths)} 张图像")
            
            # 批量预测
            results = self.predict_batch(image_paths, conf_threshold=conf_threshold, 
                                       iou_threshold=iou_threshold)
            
            # 保存结果（如果指定了输出目录）
            if output_dir:
                self._save_predictions(results, output_dir)
            
            return results
            
        except Exception as e:
            self.logger.error(f"文件夹预测失败: {e}")
            raise
    
    def _save_predictions(self, predictions: Dict[str, List[Detection]], output_dir: str):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果字典
            output_dir: 输出目录
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存详细结果
            results_file = output_path / 'predictions.txt'
            with open(results_file, 'w', encoding='utf-8') as f:
                for image_path, detections in predictions.items():
                    f.write(f"Image: {image_path}\n")
                    for detection in detections:
                        f.write(f"  {detection.class_name}: {detection.confidence:.4f} "
                               f"[{detection.bbox[0]:.1f}, {detection.bbox[1]:.1f}, "
                               f"{detection.bbox[2]:.1f}, {detection.bbox[3]:.1f}]\n")
                    f.write("\n")
            
            self.logger.info(f"预测结果已保存到: {results_file}")
            
        except Exception as e:
            self.logger.error(f"保存预测结果失败: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            return {}
        
        try:
            info = {
                'model_path': self.model_path,
                'device': self.device,
                'class_names': self.class_names,
                'num_classes': len(self.class_names),
                'model_type': 'YOLOv8'
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {}


def main():
    """
    主函数 - 用于测试推理器
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8车载摄像机目标检测推理')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像或文件夹路径')
    parser.add_argument('--output', type=str, help='输出目录路径')
    parser.add_argument('--conf', type=float, help='置信度阈值')
    parser.add_argument('--iou', type=float, help='IoU阈值')
    
    args = parser.parse_args()
    
    try:
        # 初始化推理器
        predictor = YOLOv8Predictor(args.config, args.model)
        
        # 检查输入是文件还是文件夹
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 单张图像预测
            results = predictor.predict_single(str(input_path), 
                                              conf_threshold=args.conf,
                                              iou_threshold=args.iou)
            
            print(f"检测结果 ({len(results)} 个目标):")
            for detection in results:
                print(f"  {detection.class_name}: {detection.confidence:.4f} "
                     f"[{detection.bbox[0]:.1f}, {detection.bbox[1]:.1f}, "
                     f"{detection.bbox[2]:.1f}, {detection.bbox[3]:.1f}]")
        
        elif input_path.is_dir():
            # 文件夹预测
            results = predictor.predict_folder(str(input_path),
                                              output_dir=args.output,
                                              conf_threshold=args.conf,
                                              iou_threshold=args.iou)
            
            total_detections = sum(len(dets) for dets in results.values())
            print(f"批量预测完成: {len(results)} 张图像，共检测到 {total_detections} 个目标")
        
        else:
            print(f"输入路径无效: {input_path}")
            return 1
        
    except Exception as e:
        print(f"推理失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())