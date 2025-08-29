#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果后处理模块
实现NMS、置信度筛选等后处理功能
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    def __post_init__(self):
        """验证检测结果数据"""
        if len(self.bbox) != 4:
            raise ValueError("bbox必须包含4个坐标值 [x1, y1, x2, y2]")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence必须在0-1之间")
        if self.class_id < 0:
            raise ValueError("class_id必须为非负整数")

    @property
    def area(self) -> float:
        """计算边界框面积"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        """计算边界框中心点"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'area': self.area,
            'center': self.center
        }


class PostProcessor:
    """结果后处理器"""
    
    def __init__(self, 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_detections: int = 1000,
                 class_names: Optional[List[str]] = None):
        """
        初始化后处理器
        
        Args:
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            max_detections: 最大检测数量
            class_names: 类别名称列表
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.class_names = class_names or ['car', 'truck', 'bike', 'human']
        
        self.logger = logging.getLogger(__name__)
        
        # 验证参数
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证初始化参数"""
        if not 0 <= self.conf_threshold <= 1:
            raise ValueError("conf_threshold必须在0-1之间")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold必须在0-1之间")
        if self.max_detections <= 0:
            raise ValueError("max_detections必须为正整数")
    
    def process_predictions(self, 
                          predictions: torch.Tensor,
                          image_shape: Tuple[int, int]) -> List[Detection]:
        """
        处理模型预测结果
        
        Args:
            predictions: 模型预测结果 [N, 6] (x1, y1, x2, y2, conf, class)
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            处理后的检测结果列表
        """
        try:
            if predictions is None or len(predictions) == 0:
                return []
            
            # 转换为numpy数组
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            # 置信度筛选
            conf_mask = predictions[:, 4] >= self.conf_threshold
            predictions = predictions[conf_mask]
            
            if len(predictions) == 0:
                return []
            
            # 坐标裁剪到图像范围内
            predictions = self._clip_boxes(predictions, image_shape)
            
            # 执行NMS
            keep_indices = self._nms(predictions)
            predictions = predictions[keep_indices]
            
            # 限制最大检测数量
            if len(predictions) > self.max_detections:
                # 按置信度排序并取前max_detections个
                conf_indices = np.argsort(predictions[:, 4])[::-1]
                predictions = predictions[conf_indices[:self.max_detections]]
            
            # 转换为Detection对象
            detections = []
            for pred in predictions:
                x1, y1, x2, y2, conf, cls = pred
                class_id = int(cls)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                detection = Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
            
            self.logger.debug(f"后处理完成: {len(detections)} 个检测结果")
            return detections
            
        except Exception as e:
            self.logger.error(f"后处理失败: {e}")
            return []
    
    def _clip_boxes(self, 
                   predictions: np.ndarray, 
                   image_shape: Tuple[int, int]) -> np.ndarray:
        """
        将边界框裁剪到图像范围内
        
        Args:
            predictions: 预测结果
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            裁剪后的预测结果
        """
        height, width = image_shape
        
        # 裁剪坐标
        predictions[:, 0] = np.clip(predictions[:, 0], 0, width)   # x1
        predictions[:, 1] = np.clip(predictions[:, 1], 0, height)  # y1
        predictions[:, 2] = np.clip(predictions[:, 2], 0, width)   # x2
        predictions[:, 3] = np.clip(predictions[:, 3], 0, height)  # y2
        
        # 确保x2 > x1, y2 > y1
        predictions[:, 2] = np.maximum(predictions[:, 2], predictions[:, 0] + 1)
        predictions[:, 3] = np.maximum(predictions[:, 3], predictions[:, 1] + 1)
        
        return predictions
    
    def _nms(self, predictions: np.ndarray) -> np.ndarray:
        """
        非极大值抑制 (NMS)
        
        Args:
            predictions: 预测结果 [N, 6] (x1, y1, x2, y2, conf, class)
            
        Returns:
            保留的索引数组
        """
        if len(predictions) == 0:
            return np.array([], dtype=int)
        
        # 按类别分组进行NMS
        keep_indices = []
        unique_classes = np.unique(predictions[:, 5])
        
        for cls in unique_classes:
            cls_mask = predictions[:, 5] == cls
            cls_predictions = predictions[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            if len(cls_predictions) == 0:
                continue
            
            # 对当前类别执行NMS
            cls_keep = self._nms_single_class(cls_predictions)
            keep_indices.extend(cls_indices[cls_keep])
        
        return np.array(keep_indices, dtype=int)
    
    def _nms_single_class(self, predictions: np.ndarray) -> np.ndarray:
        """
        单类别NMS
        
        Args:
            predictions: 单类别预测结果
            
        Returns:
            保留的索引数组
        """
        if len(predictions) == 0:
            return np.array([], dtype=int)
        
        # 提取坐标和置信度
        x1 = predictions[:, 0]
        y1 = predictions[:, 1]
        x2 = predictions[:, 2]
        y2 = predictions[:, 3]
        scores = predictions[:, 4]
        
        # 计算面积
        areas = (x2 - x1) * (y2 - y1)
        
        # 按置信度排序
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # 保留置信度最高的
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # 保留IoU小于阈值的
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=int)
    
    def filter_by_confidence(self, 
                           detections: List[Detection], 
                           min_confidence: float) -> List[Detection]:
        """
        按置信度筛选检测结果
        
        Args:
            detections: 检测结果列表
            min_confidence: 最小置信度阈值
            
        Returns:
            筛选后的检测结果
        """
        return [det for det in detections if det.confidence >= min_confidence]
    
    def filter_by_class(self, 
                       detections: List[Detection], 
                       target_classes: List[str]) -> List[Detection]:
        """
        按类别筛选检测结果
        
        Args:
            detections: 检测结果列表
            target_classes: 目标类别列表
            
        Returns:
            筛选后的检测结果
        """
        return [det for det in detections if det.class_name in target_classes]
    
    def filter_by_area(self, 
                      detections: List[Detection], 
                      min_area: float = 0, 
                      max_area: float = float('inf')) -> List[Detection]:
        """
        按面积筛选检测结果
        
        Args:
            detections: 检测结果列表
            min_area: 最小面积
            max_area: 最大面积
            
        Returns:
            筛选后的检测结果
        """
        return [det for det in detections if min_area <= det.area <= max_area]
    
    def sort_by_confidence(self, 
                          detections: List[Detection], 
                          descending: bool = True) -> List[Detection]:
        """
        按置信度排序检测结果
        
        Args:
            detections: 检测结果列表
            descending: 是否降序排列
            
        Returns:
            排序后的检测结果
        """
        return sorted(detections, key=lambda x: x.confidence, reverse=descending)
    
    def get_statistics(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        获取检测结果统计信息
        
        Args:
            detections: 检测结果列表
            
        Returns:
            统计信息字典
        """
        if not detections:
            return {
                'total_detections': 0,
                'class_counts': {},
                'confidence_stats': {},
                'area_stats': {}
            }
        
        # 类别统计
        class_counts = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        # 置信度统计
        confidences = [det.confidence for det in detections]
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
        
        # 面积统计
        areas = [det.area for det in detections]
        area_stats = {
            'mean': np.mean(areas),
            'std': np.std(areas),
            'min': np.min(areas),
            'max': np.max(areas),
            'median': np.median(areas)
        }
        
        return {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'confidence_stats': confidence_stats,
            'area_stats': area_stats
        }


def main():
    """测试后处理器"""
    import torch
    
    # 创建测试数据
    predictions = torch.tensor([
        [100, 100, 200, 200, 0.9, 0],  # car
        [150, 150, 250, 250, 0.8, 0],  # car (重叠)
        [300, 300, 400, 400, 0.7, 1],  # truck
        [50, 50, 80, 80, 0.3, 2],      # bike (低置信度)
        [500, 500, 600, 600, 0.85, 3], # human
    ])
    
    # 创建后处理器
    processor = PostProcessor(
        conf_threshold=0.5,
        iou_threshold=0.5,
        max_detections=100
    )
    
    # 处理预测结果
    detections = processor.process_predictions(predictions, (800, 800))
    
    # 打印结果
    print(f"检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det.class_name}: {det.confidence:.3f} at {det.bbox}")
    
    # 获取统计信息
    stats = processor.get_statistics(detections)
    print(f"\n统计信息: {stats}")


if __name__ == "__main__":
    main()