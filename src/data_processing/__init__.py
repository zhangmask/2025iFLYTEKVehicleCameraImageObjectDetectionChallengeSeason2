#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块
包含XML解析、数据转换、数据增强等功能
"""

from .xml_parser import XMLParser, BoundingBox, ObjectAnnotation, ImageAnnotation
from .data_converter import DataConverter

__all__ = [
    'XMLParser',
    'BoundingBox', 
    'ObjectAnnotation',
    'ImageAnnotation',
    'DataConverter'
]

__version__ = '1.0.0'
__author__ = 'Vehicle Detection Team'
__description__ = '车载摄像机目标检测数据处理模块'