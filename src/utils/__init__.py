#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块
提供通用的辅助函数和工具类
"""

from .common import (
    setup_logging,
    get_device,
    get_gpu_info,
    load_config,
    save_config,
    ensure_dir,
    copy_file,
    get_file_list,
    get_timestamp,
    format_size,
    validate_paths,
    clean_directory,
    merge_configs,
    print_system_info,
    ProgressTracker
)

__version__ = '1.0.0'
__author__ = '车载摄像机目标检测团队'
__description__ = '通用工具包'

__all__ = [
    'setup_logging',
    'get_device',
    'get_gpu_info',
    'load_config',
    'save_config',
    'ensure_dir',
    'copy_file',
    'get_file_list',
    'get_timestamp',
    'format_size',
    'validate_paths',
    'clean_directory',
    'merge_configs',
    'print_system_info',
    'ProgressTracker'
]