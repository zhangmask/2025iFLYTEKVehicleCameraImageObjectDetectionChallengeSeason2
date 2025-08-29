#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模块
包含模型推理引擎和相关工具
"""

from .predictor import YOLOv8Predictor
from .result_formatter import CompetitionFormatter
from .post_processor import PostProcessor, Detection

__all__ = [
    'YOLOv8Predictor',
    'CompetitionFormatter',
    'PostProcessor',
    'Detection'
]

__version__ = '1.0.0'
__author__ = '车载摄像机目标检测团队'
__description__ = '模型推理模块'