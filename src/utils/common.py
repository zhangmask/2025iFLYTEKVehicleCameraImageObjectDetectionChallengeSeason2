#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数
提供日志配置、文件操作、设备检测等功能
"""

import os
import sys
import json
import yaml
import logging
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式
        
    Returns:
        配置好的logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建根logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter(log_format)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    获取可用的计算设备
    
    Args:
        prefer_gpu: 是否优先使用GPU
        
    Returns:
        torch设备对象
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"使用GPU设备: {gpu_name}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU设备")
    
    return device

def get_gpu_info() -> Dict[str, Any]:
    """
    获取GPU信息
    
    Returns:
        GPU信息字典
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_names': [],
        'memory_info': []
    }
    
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(info['gpu_count']):
            gpu_name = torch.cuda.get_device_name(i)
            info['gpu_names'].append(gpu_name)
            
            # 获取显存信息
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_cached = torch.cuda.memory_reserved(i)
            
            info['memory_info'].append({
                'device_id': i,
                'total_memory_gb': memory_total / (1024**3),
                'allocated_memory_gb': memory_allocated / (1024**3),
                'cached_memory_gb': memory_cached / (1024**3)
            })
    
    return info

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        return config
    except Exception as e:
        raise ValueError(f"无法加载配置文件 {config_path}: {e}")

def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    except Exception as e:
        raise ValueError(f"无法保存配置文件 {config_path}: {e}")

def ensure_dir(dir_path: Union[str, Path]):
    """
    确保目录存在
    
    Args:
        dir_path: 目录路径
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    """
    复制文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"源文件不存在: {src_path}")
    
    # 确保目标目录存在
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src_path, dst_path)

def get_file_list(directory: Union[str, Path], 
                 extensions: Optional[List[str]] = None,
                 recursive: bool = True) -> List[Path]:
    """
    获取目录下的文件列表
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表
        recursive: 是否递归搜索
        
    Returns:
        文件路径列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if extensions is None:
        extensions = ['*']
    
    files = []
    
    for ext in extensions:
        if recursive:
            pattern = f"**/*.{ext}" if ext != '*' else "**/*"
            found_files = directory.glob(pattern)
        else:
            pattern = f"*.{ext}" if ext != '*' else "*"
            found_files = directory.glob(pattern)
        
        files.extend([f for f in found_files if f.is_file()])
    
    return sorted(list(set(files)))

def get_timestamp(format_str: str = '%Y%m%d_%H%M%S') -> str:
    """
    获取当前时间戳
    
    Args:
        format_str: 时间格式字符串
        
    Returns:
        格式化的时间戳
    """
    return datetime.now().strftime(format_str)

def format_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化的大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f}{size_names[i]}"

def validate_paths(paths: Dict[str, str], check_existence: bool = True) -> Dict[str, bool]:
    """
    验证路径有效性
    
    Args:
        paths: 路径字典
        check_existence: 是否检查路径存在性
        
    Returns:
        验证结果字典
    """
    results = {}
    
    for name, path in paths.items():
        try:
            path_obj = Path(path)
            
            if check_existence:
                results[name] = path_obj.exists()
            else:
                # 只检查路径格式是否有效
                results[name] = True
        except Exception:
            results[name] = False
    
    return results

def clean_directory(directory: Union[str, Path], 
                   keep_extensions: Optional[List[str]] = None):
    """
    清理目录
    
    Args:
        directory: 目录路径
        keep_extensions: 保留的文件扩展名列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        return
    
    for item in directory.iterdir():
        if item.is_file():
            if keep_extensions is None or item.suffix.lower() not in keep_extensions:
                item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def print_system_info():
    """
    打印系统信息
    """
    logger = logging.getLogger('system_info')
    
    logger.info("=" * 50)
    logger.info("系统信息")
    logger.info("=" * 50)
    
    # Python版本
    logger.info(f"Python版本: {sys.version}")
    
    # PyTorch版本
    logger.info(f"PyTorch版本: {torch.__version__}")
    
    # GPU信息
    gpu_info = get_gpu_info()
    logger.info(f"CUDA可用: {gpu_info['cuda_available']}")
    
    if gpu_info['cuda_available']:
        logger.info(f"GPU数量: {gpu_info['gpu_count']}")
        for i, gpu_name in enumerate(gpu_info['gpu_names']):
            memory_info = gpu_info['memory_info'][i]
            logger.info(f"GPU {i}: {gpu_name}")
            logger.info(f"  总显存: {memory_info['total_memory_gb']:.2f}GB")
            logger.info(f"  已用显存: {memory_info['allocated_memory_gb']:.2f}GB")
    
    logger.info("=" * 50)

class ProgressTracker:
    """
    进度跟踪器
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        初始化进度跟踪器
        
        Args:
            total: 总数量
            description: 描述信息
        """
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update(self, increment: int = 1):
        """
        更新进度
        
        Args:
            increment: 增量
        """
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """
        完成进度
        """
        self.logger.info(f"{self.description}: 完成 ({self.total}/{self.total})")