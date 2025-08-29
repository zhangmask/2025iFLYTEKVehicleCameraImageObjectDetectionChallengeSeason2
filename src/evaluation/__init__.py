#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估模块
提供mAP计算和模型性能评估功能
"""

from .evaluator import (
    Detection,
    GroundTruth,
    APCalculator,
    mAPEvaluator
)

__version__ = '1.0.0'
__author__ = '车载摄像机目标检测团队'
__description__ = '目标检测模型评估工具包'

__all__ = [
    'Detection',
    'GroundTruth', 
    'APCalculator',
    'mAPEvaluator'
]