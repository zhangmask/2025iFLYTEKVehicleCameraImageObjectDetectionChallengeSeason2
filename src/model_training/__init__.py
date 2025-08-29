#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块
包含YOLOv8训练引擎和相关工具
"""

from .trainer import YOLOv8Trainer

__all__ = [
    'YOLOv8Trainer'
]

__version__ = '1.0.0'
__author__ = 'Vehicle Detection Team'
__description__ = '车载摄像机目标检测模型训练模块'