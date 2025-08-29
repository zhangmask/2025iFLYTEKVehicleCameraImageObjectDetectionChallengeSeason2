#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车载摄像机图像目标检测解决方案
基于YOLOv8的端到端目标检测系统
"""

# 导入主要模块
from . import data_processing
from . import model_training
from . import inference
from . import evaluation
from . import utils

# 导入核心类
from .data_processing import XMLParser, DataConverter
from .model_training import YOLOv8Trainer
from .inference import YOLOv8Predictor, PostProcessor, CompetitionFormatter, Detection
from .evaluation import mAPEvaluator, APCalculator, GroundTruth
from .utils import (
    setup_logging,
    get_device,
    load_config,
    ensure_dir,
    ProgressTracker
)

__version__ = '1.0.0'
__author__ = '车载摄像机目标检测团队'
__description__ = '基于YOLOv8的车载摄像机图像目标检测解决方案'
__license__ = 'MIT'

# 支持的目标类别
SUPPORTED_CLASSES = ['car', 'truck', 'bike', 'human']

# 默认配置
DEFAULT_CONFIG = {
    'model': {
        'size': 'yolov8s',
        'pretrained': True
    },
    'training': {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.01
    },
    'inference': {
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_det': 1000
    }
}

__all__ = [
    # 模块
    'data_processing',
    'model_training', 
    'inference',
    'evaluation',
    'utils',
    
    # 核心类
    'XMLParser',
    'DataConverter',
    'YOLOv8Trainer',
    'YOLOv8Predictor',
    'PostProcessor',
    'CompetitionFormatter',
    'Detection',
    'mAPEvaluator',
    'APCalculator',
    'GroundTruth',
    
    # 工具函数
    'setup_logging',
    'get_device',
    'load_config',
    'ensure_dir',
    'ProgressTracker',
    
    # 常量
    'SUPPORTED_CLASSES',
    'DEFAULT_CONFIG'
]


def get_version():
    """获取版本信息"""
    return __version__


def get_supported_classes():
    """获取支持的目标类别"""
    return SUPPORTED_CLASSES.copy()


def get_default_config():
    """获取默认配置"""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def print_info():
    """打印项目信息"""
    print(f"{__description__}")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print(f"支持的类别: {', '.join(SUPPORTED_CLASSES)}")