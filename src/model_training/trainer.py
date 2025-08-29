#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8模型训练引擎
支持GPU加速训练、验证和模型保存
"""

import os
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO
from ultralytics.utils import LOGGER

class YOLOv8Trainer:
    """
    YOLOv8模型训练器
    """
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.device = self._get_device()
        self._setup_logging()
        
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
        获取训练设备
        
        Returns:
            设备名称 ('cuda' 或 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到 {gpu_count} 个GPU: {gpu_name}")
        else:
            device = 'cpu'
            print("未检测到GPU，使用CPU训练")
        return device
    
    def _setup_logging(self):
        """
        设置日志记录
        """
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self, model_size: str = 'yolov8n') -> None:
        """
        初始化YOLO模型
        
        Args:
            model_size: 模型大小 ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
        """
        try:
            # 检查是否有预训练模型
            pretrained_path = self.config['model'].get('pretrained_path')
            if pretrained_path and os.path.exists(pretrained_path):
                self.logger.info(f"加载预训练模型: {pretrained_path}")
                self.model = YOLO(pretrained_path)
            else:
                self.logger.info(f"初始化新的{model_size}模型")
                self.model = YOLO(f'{model_size}.pt')
            
            self.logger.info(f"模型初始化完成，使用设备: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {e}")
    
    def prepare_dataset_config(self, dataset_path: str) -> str:
        """
        准备数据集配置文件
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            数据集配置文件路径
        """
        dataset_config = {
            'path': '.',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.config['classes']['names']),
            'names': self.config['classes']['names']
        }
        
        config_file = os.path.join(dataset_path, 'dataset.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"数据集配置文件已创建: {config_file}")
        return config_file
    
    def train(self, dataset_config_path: str, resume: bool = False) -> Dict[str, Any]:
        """
        开始训练模型
        
        Args:
            dataset_config_path: 数据集配置文件路径
            resume: 是否恢复训练
            
        Returns:
            训练结果字典
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先调用 initialize_model()")
        
        try:
            # 获取训练参数
            train_params = self.config['train']
            
            # 设置输出目录
            output_dir = Path(self.config['paths']['models'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("开始训练模型...")
            self.logger.info(f"训练参数: {train_params}")
            
            # 转换为绝对路径并确保路径格式正确
            dataset_config_abs = os.path.abspath(dataset_config_path)
            self.logger.info(f"使用数据集配置文件: {dataset_config_abs}")
            
            # 切换到数据集目录以确保相对路径正确解析
            original_cwd = os.getcwd()
            dataset_dir = os.path.dirname(dataset_config_abs)
            os.chdir(dataset_dir)
            self.logger.info(f"切换工作目录到: {dataset_dir}")
            
            try:
                # 开始训练
                results = self.model.train(
                    data=os.path.basename(dataset_config_abs),
                    epochs=train_params['epochs'],
                    batch=train_params['batch_size'],
                    imgsz=train_params.get('image_size', 640),
                    lr0=train_params['learning_rate'],
                    weight_decay=train_params['weight_decay'],
                    momentum=train_params['momentum'],
                    device=self.device,
                    project=str(output_dir),
                    name='yolov8_vehicle_detection',
                    exist_ok=True,
                    resume=resume,
                    save=True,
                    save_period=train_params.get('save_period', 10),
                    val=True,
                    plots=True,
                    verbose=True
                )
                
                self.logger.info("训练完成!")
                return results
                
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
                self.logger.info(f"恢复工作目录到: {original_cwd}")
            
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            raise
    
    def validate(self, dataset_config_path: str, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        验证模型性能
        
        Args:
            dataset_config_path: 数据集配置文件路径
            model_path: 模型文件路径（可选）
            
        Returns:
            验证结果字典
        """
        try:
            # 如果指定了模型路径，加载该模型
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                self.logger.info(f"加载模型进行验证: {model_path}")
            elif self.model is not None:
                model = self.model
            else:
                raise ValueError("没有可用的模型进行验证")
            
            self.logger.info("开始模型验证...")
            
            # 执行验证
            results = model.val(
                data=dataset_config_path,
                device=self.device,
                verbose=True
            )
            
            self.logger.info("验证完成!")
            return results
            
        except Exception as e:
            self.logger.error(f"验证过程中发生错误: {e}")
            raise
    
    def export_model(self, model_path: str, export_format: str = 'onnx') -> str:
        """
        导出模型为指定格式
        
        Args:
            model_path: 模型文件路径
            export_format: 导出格式 ('onnx', 'torchscript', 'tflite'等)
            
        Returns:
            导出文件路径
        """
        try:
            model = YOLO(model_path)
            self.logger.info(f"导出模型为{export_format}格式...")
            
            exported_path = model.export(format=export_format)
            
            self.logger.info(f"模型导出完成: {exported_path}")
            return exported_path
            
        except Exception as e:
            self.logger.error(f"模型导出失败: {e}")
            raise
    
    def get_best_model_path(self) -> Optional[str]:
        """
        获取最佳模型路径
        
        Returns:
            最佳模型文件路径
        """
        try:
            # 首先尝试在outputs/yolo_data/models目录下查找（训练时的实际保存位置）
            output_data_dir = Path(self.config['paths']['output_data'])
            best_model_path = output_data_dir / 'models' / 'yolov8_vehicle_detection' / 'weights' / 'best.pt'
            
            if best_model_path.exists():
                return str(best_model_path)
            
            # 如果没找到，再尝试在原始models目录下查找
            models_dir = Path(self.config['paths']['models'])
            best_model_path = models_dir / 'yolov8_vehicle_detection' / 'weights' / 'best.pt'
            
            if best_model_path.exists():
                return str(best_model_path)
            else:
                self.logger.warning("未找到最佳模型文件")
                return None
                
        except Exception as e:
            self.logger.error(f"获取最佳模型路径失败: {e}")
            return None
    
    def save_training_config(self, output_path: str) -> None:
        """
        保存训练配置
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"训练配置已保存: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存训练配置失败: {e}")
            raise


def main():
    """
    主函数 - 用于测试训练器
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8车载摄像机目标检测训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset', type=str, required=True, help='数据集路径')
    parser.add_argument('--model-size', type=str, default='yolov8n', 
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='模型大小')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    
    args = parser.parse_args()
    
    try:
        # 初始化训练器
        trainer = YOLOv8Trainer(args.config)
        
        # 初始化模型
        trainer.initialize_model(args.model_size)
        
        # 准备数据集配置
        dataset_config = trainer.prepare_dataset_config(args.dataset)
        
        # 开始训练
        results = trainer.train(dataset_config, resume=args.resume)
        
        print("训练完成!")
        print(f"最佳模型路径: {trainer.get_best_model_path()}")
        
    except Exception as e:
        print(f"训练失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())