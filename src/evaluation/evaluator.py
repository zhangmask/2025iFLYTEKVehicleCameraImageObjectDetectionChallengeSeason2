#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mAP评估工具
用于计算目标检测模型的性能指标
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Detection:
    """
    检测结果数据类
    """
    image_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    
@dataclass
class GroundTruth:
    """
    真实标注数据类
    """
    image_id: str
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    difficult: bool = False

class APCalculator:
    """
    AP计算器
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        初始化AP计算器
        
        Args:
            iou_threshold: IoU阈值
        """
        self.iou_threshold = iou_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_iou(self, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            box1: 边界框1 (x1, y1, x2, y2)
            box2: 边界框2 (x1, y1, x2, y2)
            
        Returns:
            IoU值
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        # 交集面积
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 并集面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_ap(self, detections: List[Detection], 
                    ground_truths: List[GroundTruth]) -> Dict[str, float]:
        """
        计算单个类别的AP
        
        Args:
            detections: 检测结果列表
            ground_truths: 真实标注列表
            
        Returns:
            包含AP、precision、recall等指标的字典
        """
        if not detections:
            return {'ap': 0.0, 'precision': [], 'recall': [], 'f1': 0.0}
        
        # 按置信度降序排序
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        # 统计每张图像的真实目标数量
        gt_count_per_image = defaultdict(int)
        for gt in ground_truths:
            if not gt.difficult:
                gt_count_per_image[gt.image_id] += 1
        
        total_gt = sum(gt_count_per_image.values())
        
        if total_gt == 0:
            return {'ap': 0.0, 'precision': [], 'recall': [], 'f1': 0.0}
        
        # 为每张图像的真实标注创建匹配标记
        gt_matched = {}
        for gt in ground_truths:
            if gt.image_id not in gt_matched:
                gt_matched[gt.image_id] = []
            gt_matched[gt.image_id].append(False)
        
        tp = []
        fp = []
        
        for detection in detections:
            image_id = detection.image_id
            
            # 获取该图像的所有真实标注
            image_gts = [gt for gt in ground_truths if gt.image_id == image_id]
            
            if not image_gts:
                # 该图像没有真实标注，检测为假正例
                tp.append(0)
                fp.append(1)
                continue
            
            # 计算与所有真实标注的IoU
            max_iou = 0.0
            max_gt_idx = -1
            
            for gt_idx, gt in enumerate(image_gts):
                if gt.difficult:
                    continue
                    
                iou = self.calculate_iou(detection.bbox, gt.bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # 判断是否为真正例
            if max_iou >= self.iou_threshold and max_gt_idx >= 0:
                # 检查该真实标注是否已被匹配
                if not gt_matched[image_id][max_gt_idx]:
                    tp.append(1)
                    fp.append(0)
                    gt_matched[image_id][max_gt_idx] = True
                else:
                    # 该真实标注已被匹配，当前检测为假正例
                    tp.append(0)
                    fp.append(1)
            else:
                # IoU不足，为假正例
                tp.append(0)
                fp.append(1)
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算precision和recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / total_gt
        
        # 计算AP (使用11点插值法)
        ap = self._calculate_ap_11_point(precision, recall)
        
        # 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        max_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
        
        return {
            'ap': ap,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': max_f1,
            'tp_count': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
            'fp_count': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
            'total_gt': total_gt
        }
    
    def _calculate_ap_11_point(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        使用11点插值法计算AP
        
        Args:
            precision: precision数组
            recall: recall数组
            
        Returns:
            AP值
        """
        # 11个recall点
        recall_points = np.linspace(0, 1, 11)
        
        # 对每个recall点进行插值
        interpolated_precision = []
        
        for r in recall_points:
            # 找到recall >= r的所有点
            indices = np.where(recall >= r)[0]
            
            if len(indices) == 0:
                interpolated_precision.append(0.0)
            else:
                # 取这些点中precision的最大值
                max_precision = np.max(precision[indices])
                interpolated_precision.append(max_precision)
        
        # 计算平均precision
        return np.mean(interpolated_precision)

class mAPEvaluator:
    """
    mAP评估器
    """
    
    def __init__(self, class_names: List[str], iou_thresholds: List[float] = None):
        """
        初始化mAP评估器
        
        Args:
            class_names: 类别名称列表
            iou_thresholds: IoU阈值列表
        """
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_detections_from_yolo(self, predictions_dir: str, 
                                 image_size: Tuple[int, int] = (1920, 1020)) -> List[Detection]:
        """
        从YOLO格式的预测结果加载检测数据
        
        Args:
            predictions_dir: 预测结果目录
            image_size: 图像尺寸 (width, height)
            
        Returns:
            检测结果列表
        """
        detections = []
        predictions_path = Path(predictions_dir)
        
        for txt_file in predictions_path.glob('*.txt'):
            image_id = txt_file.stem
            
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5])
                        
                        # 转换为绝对坐标
                        img_w, img_h = image_size
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h
                        
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                        
                        detection = Detection(
                            image_id=image_id,
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2)
                        )
                        detections.append(detection)
        
        self.logger.info(f"加载了 {len(detections)} 个检测结果")
        return detections
    
    def load_ground_truths_from_yolo(self, labels_dir: str, 
                                   image_size: Tuple[int, int] = (1920, 1020)) -> List[GroundTruth]:
        """
        从YOLO格式的标注文件加载真实标注
        
        Args:
            labels_dir: 标注文件目录
            image_size: 图像尺寸 (width, height)
            
        Returns:
            真实标注列表
        """
        ground_truths = []
        labels_path = Path(labels_dir)
        
        for txt_file in labels_path.glob('*.txt'):
            image_id = txt_file.stem
            
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为绝对坐标
                        img_w, img_h = image_size
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h
                        
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                        
                        gt = GroundTruth(
                            image_id=image_id,
                            class_id=class_id,
                            class_name=class_name,
                            bbox=(x1, y1, x2, y2)
                        )
                        ground_truths.append(gt)
        
        self.logger.info(f"加载了 {len(ground_truths)} 个真实标注")
        return ground_truths
    
    def evaluate(self, detections: List[Detection], 
                ground_truths: List[GroundTruth]) -> Dict[str, Any]:
        """
        执行mAP评估
        
        Args:
            detections: 检测结果列表
            ground_truths: 真实标注列表
            
        Returns:
            评估结果字典
        """
        results = {
            'class_results': {},
            'overall_results': {},
            'iou_results': {}
        }
        
        # 按IoU阈值评估
        for iou_threshold in self.iou_thresholds:
            calculator = APCalculator(iou_threshold)
            
            class_aps = []
            class_results = {}
            
            # 按类别评估
            for class_name in self.class_names:
                class_id = self.class_to_id[class_name]
                
                # 过滤该类别的检测结果和真实标注
                class_detections = [d for d in detections if d.class_id == class_id]
                class_gts = [gt for gt in ground_truths if gt.class_id == class_id]
                
                # 计算AP
                ap_result = calculator.calculate_ap(class_detections, class_gts)
                class_results[class_name] = ap_result
                class_aps.append(ap_result['ap'])
            
            # 计算mAP
            mean_ap = np.mean(class_aps) if class_aps else 0.0
            
            results['iou_results'][f'IoU_{iou_threshold:.2f}'] = {
                'mAP': mean_ap,
                'class_results': class_results
            }
        
        # 计算IoU 0.5的详细结果
        if 0.5 in self.iou_thresholds:
            results['class_results'] = results['iou_results']['IoU_0.50']['class_results']
            results['overall_results']['mAP@0.5'] = results['iou_results']['IoU_0.50']['mAP']
        
        # 计算IoU 0.5:0.95的平均mAP
        all_maps = [results['iou_results'][key]['mAP'] for key in results['iou_results']]
        results['overall_results']['mAP@0.5:0.95'] = np.mean(all_maps) if all_maps else 0.0
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            output_path: 输出文件路径
        """
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {output_path}")
    
    def generate_report(self, results: Dict[str, Any], output_dir: str):
        """
        生成评估报告
        
        Args:
            results: 评估结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        report_file = output_path / 'evaluation_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("车载摄像机目标检测模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体结果
            f.write("总体性能指标:\n")
            f.write("-" * 20 + "\n")
            overall = results.get('overall_results', {})
            f.write(f"mAP@0.5: {overall.get('mAP@0.5', 0.0):.4f}\n")
            f.write(f"mAP@0.5:0.95: {overall.get('mAP@0.5:0.95', 0.0):.4f}\n\n")
            
            # 各类别结果
            f.write("各类别性能指标 (IoU@0.5):\n")
            f.write("-" * 30 + "\n")
            class_results = results.get('class_results', {})
            
            for class_name in self.class_names:
                if class_name in class_results:
                    result = class_results[class_name]
                    f.write(f"{class_name}:\n")
                    f.write(f"  AP: {result['ap']:.4f}\n")
                    f.write(f"  F1: {result['f1']:.4f}\n")
                    f.write(f"  TP: {result['tp_count']}\n")
                    f.write(f"  FP: {result['fp_count']}\n")
                    f.write(f"  GT: {result['total_gt']}\n\n")
            
            # 不同IoU阈值下的mAP
            f.write("不同IoU阈值下的mAP:\n")
            f.write("-" * 25 + "\n")
            iou_results = results.get('iou_results', {})
            for iou_key, iou_result in iou_results.items():
                f.write(f"{iou_key}: {iou_result['mAP']:.4f}\n")
        
        self.logger.info(f"评估报告已保存到: {report_file}")
        
        # 生成可视化图表
        self._plot_results(results, output_path)
    
    def _plot_results(self, results: Dict[str, Any], output_dir: Path):
        """
        生成可视化图表
        
        Args:
            results: 评估结果
            output_dir: 输出目录
        """
        plt.style.use('seaborn-v0_8')
        
        # 1. 各类别AP柱状图
        class_results = results.get('class_results', {})
        if class_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            classes = list(class_results.keys())
            aps = [class_results[cls]['ap'] for cls in classes]
            
            bars = ax.bar(classes, aps, color='skyblue', alpha=0.7)
            ax.set_ylabel('Average Precision (AP)')
            ax.set_title('各类别AP性能 (IoU@0.5)')
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, ap in zip(bars, aps):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{ap:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'class_ap_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 不同IoU阈值下的mAP曲线
        iou_results = results.get('iou_results', {})
        if iou_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            iou_thresholds = []
            maps = []
            
            for iou_key, iou_result in iou_results.items():
                iou_val = float(iou_key.split('_')[1])
                iou_thresholds.append(iou_val)
                maps.append(iou_result['mAP'])
            
            ax.plot(iou_thresholds, maps, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('IoU Threshold')
            ax.set_ylabel('mAP')
            ax.set_title('不同IoU阈值下的mAP性能')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'map_vs_iou.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"可视化图表已保存到: {output_dir}")

def main():
    """
    测试函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='mAP评估工具')
    parser.add_argument('--predictions', type=str, required=True,
                       help='预测结果目录')
    parser.add_argument('--ground-truths', type=str, required=True,
                       help='真实标注目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--classes', type=str, nargs='+',
                       default=['car', 'truck', 'bike', 'human'],
                       help='类别名称列表')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化评估器
    evaluator = mAPEvaluator(args.classes)
    
    # 加载数据
    detections = evaluator.load_detections_from_yolo(args.predictions)
    ground_truths = evaluator.load_ground_truths_from_yolo(args.ground_truths)
    
    # 执行评估
    results = evaluator.evaluate(detections, ground_truths)
    
    # 保存结果
    evaluator.save_results(results, os.path.join(args.output, 'evaluation_results.json'))
    evaluator.generate_report(results, args.output)
    
    print(f"评估完成，结果保存到: {args.output}")

if __name__ == '__main__':
    main()