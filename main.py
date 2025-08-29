#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车载摄像机图像目标检测挑战赛 - 主程序
支持数据处理、模型训练、推理预测和结果输出
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing.xml_parser import XMLParser
from data_processing.data_converter import DataConverter
from model_training.trainer import YOLOv8Trainer
from inference.predictor import YOLOv8Predictor
from inference.result_formatter import CompetitionFormatter

def setup_logging(log_level: str = 'INFO'):
    """
    设置全局日志配置
    
    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"无法加载配置文件 {config_path}: {e}")

def prepare_data(args, config: Dict[str, Any]):
    """
    数据准备和转换
    
    Args:
        args: 命令行参数
        config: 配置字典
    """
    logger = logging.getLogger('data_preparation')
    logger.info("开始数据准备和转换...")
    
    try:
        # 解析XML标注文件
        xml_parser = XMLParser()
        
        train_xml_dir = config['data']['train_label_dir']
        logger.info(f"解析训练标注文件: {train_xml_dir}")
        
        annotations = xml_parser.parse_xml_directory(train_xml_dir)
        logger.info(f"成功解析 {len(annotations)} 个标注文件")
        
        # 获取数据集统计信息
        stats = xml_parser.get_statistics(annotations)
        logger.info(f"数据集统计: {stats}")
        
        # 转换为YOLO格式
        class_mapping = config['classes']['class_mapping']
        converter = DataConverter(
            class_mapping=class_mapping,
            output_dir=args.output_dir or config['paths']['yolo_dataset']
        )
        
        logger.info("开始转换为YOLO格式...")
        dataset_info = converter.convert_xml_to_yolo(
            image_dir=config['data']['train_image_dir'],
            xml_dir=train_xml_dir,
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split']
        )
        
        logger.info(f"YOLO数据集转换完成: {dataset_info}")
        
        # 添加数据集路径到返回信息中
        dataset_info['dataset_path'] = args.output_dir or config['paths']['yolo_dataset']
        
        return dataset_info
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        raise

def train_model(args, config: Dict[str, Any], dataset_path: str):
    """
    训练模型
    
    Args:
        args: 命令行参数
        config: 配置字典
        dataset_path: 数据集路径
    """
    logger = logging.getLogger('model_training')
    logger.info("开始模型训练...")
    
    try:
        # 初始化训练器
        trainer = YOLOv8Trainer(args.config)
        
        # 初始化模型
        model_size = args.model_size or config['model']['size']
        trainer.initialize_model(model_size)
        
        # 准备数据集配置
        dataset_config = trainer.prepare_dataset_config(dataset_path)
        
        # 开始训练
        results = trainer.train(dataset_config, resume=args.resume)
        
        # 获取最佳模型路径
        best_model_path = trainer.get_best_model_path()
        logger.info(f"训练完成，最佳模型: {best_model_path}")
        
        return best_model_path
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

def predict_test_set(args, config: Dict[str, Any], model_path: str):
    """
    对测试集进行预测
    
    Args:
        args: 命令行参数
        config: 配置字典
        model_path: 模型文件路径
    """
    logger = logging.getLogger('prediction')
    logger.info("开始测试集预测...")
    
    try:
        # 初始化推理器
        predictor = YOLOv8Predictor(args.config, model_path)
        
        # 预测测试集
        test_dir = config['data']['test_image_dir']
        logger.info(f"预测测试集: {test_dir}")
        
        predictions = predictor.predict_folder(
            test_dir,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        
        logger.info(f"测试集预测完成，共处理 {len(predictions)} 张图像")
        
        # 格式化输出结果
        output_dir = args.output_dir or config['paths']['outputs']
        formatter = CompetitionFormatter(output_dir)
        
        result_info = formatter.process_test_predictions(
            predictions, 
            create_zip=True
        )
        
        # 生成摘要报告
        summary = formatter.generate_submission_summary(result_info)
        logger.info(f"\n{summary}")
        
        # 验证提交文件格式
        if 'zip_file' in result_info:
            validation_result = formatter.validate_submission_format(result_info['zip_file'])
            if validation_result['is_valid']:
                logger.info("提交文件格式验证通过")
            else:
                logger.warning(f"提交文件格式验证失败: {validation_result['errors']}")
        
        return result_info
        
    except Exception as e:
        logger.error(f"测试集预测失败: {e}")
        raise

def validate_model(args, config: Dict[str, Any], model_path: str, dataset_path: str):
    """
    验证模型性能
    
    Args:
        args: 命令行参数
        config: 配置字典
        model_path: 模型文件路径
        dataset_path: 数据集路径
    """
    logger = logging.getLogger('validation')
    logger.info("开始模型验证...")
    
    try:
        # 初始化训练器进行验证
        trainer = YOLOv8Trainer(args.config)
        
        # 准备数据集配置
        dataset_config = trainer.prepare_dataset_config(dataset_path)
        
        # 执行验证
        results = trainer.validate(dataset_config, model_path)
        
        logger.info("模型验证完成")
        return results
        
    except Exception as e:
        logger.error(f"模型验证失败: {e}")
        raise

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='车载摄像机图像目标检测挑战赛',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流程（数据准备 + 训练 + 预测）
  python main.py --config configs/config.yaml --mode full
  
  # 仅数据准备
  python main.py --config configs/config.yaml --mode prepare
  
  # 仅训练模型
  python main.py --config configs/config.yaml --mode train --dataset-path ./yolo_dataset
  
  # 仅预测测试集
  python main.py --config configs/config.yaml --mode predict --model-path ./models/best.pt
  
  # 验证模型
  python main.py --config configs/config.yaml --mode validate --model-path ./models/best.pt --dataset-path ./yolo_dataset
        """
    )
    
    # 基本参数
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['full', 'prepare', 'train', 'predict', 'validate'],
                       help='运行模式')
    parser.add_argument('--output-dir', type=str,
                       help='输出目录路径')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    # 训练相关参数
    parser.add_argument('--model-size', type=str,
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='模型大小')
    parser.add_argument('--resume', action='store_true',
                       help='恢复训练')
    parser.add_argument('--dataset-path', type=str,
                       help='YOLO数据集路径')
    
    # 推理相关参数
    parser.add_argument('--model-path', type=str,
                       help='模型文件路径')
    parser.add_argument('--conf-threshold', type=float,
                       help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float,
                       help='IoU阈值')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger('main')
    
    try:
        # 加载配置
        config = load_config(args.config)
        logger.info(f"配置加载完成: {args.config}")
        
        # 创建输出目录
        output_dir = args.output_dir or config['paths']['outputs']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        dataset_path = None
        model_path = args.model_path
        
        # 根据模式执行相应操作
        if args.mode in ['full', 'prepare']:
            logger.info("=" * 50)
            logger.info("步骤 1: 数据准备和转换")
            logger.info("=" * 50)
            
            dataset_info = prepare_data(args, config)
            dataset_path = dataset_info['dataset_path']
            
            if args.mode == 'prepare':
                logger.info("数据准备完成")
                return 0
        
        if args.mode in ['full', 'train']:
            logger.info("=" * 50)
            logger.info("步骤 2: 模型训练")
            logger.info("=" * 50)
            
            if dataset_path is None:
                dataset_path = args.dataset_path
                if not dataset_path:
                    raise ValueError("训练模式需要指定 --dataset-path 参数")
            
            model_path = train_model(args, config, dataset_path)
            
            if args.mode == 'train':
                logger.info(f"模型训练完成: {model_path}")
                return 0
        
        if args.mode in ['full', 'predict']:
            logger.info("=" * 50)
            logger.info("步骤 3: 测试集预测")
            logger.info("=" * 50)
            
            if model_path is None:
                raise ValueError("预测模式需要指定 --model-path 参数")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            result_info = predict_test_set(args, config, model_path)
            
            logger.info("=" * 50)
            logger.info("预测完成！")
            logger.info("=" * 50)
            
            if 'zip_file' in result_info:
                logger.info(f"提交文件: {result_info['zip_file']}")
                logger.info("请将此文件提交到比赛平台")
        
        if args.mode == 'validate':
            logger.info("=" * 50)
            logger.info("模型验证")
            logger.info("=" * 50)
            
            if model_path is None:
                raise ValueError("验证模式需要指定 --model-path 参数")
            
            if dataset_path is None:
                dataset_path = args.dataset_path
                if not dataset_path:
                    raise ValueError("验证模式需要指定 --dataset-path 参数")
            
            results = validate_model(args, config, model_path, dataset_path)
            logger.info("模型验证完成")
        
        logger.info("所有任务执行完成！")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"执行失败: {e}")
        return 1

if __name__ == '__main__':
    exit(main())