#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车载摄像机目标检测项目安装脚本
自动检测环境并安装依赖包
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip():
    """检查pip是否可用"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip检查通过")
        return True
    except subprocess.CalledProcessError:
        print("❌ 错误: pip不可用")
        return False


def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✅ CUDA可用: {cuda_version}")
            print(f"   GPU数量: {gpu_count}")
            print(f"   GPU型号: {gpu_name}")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU模式")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查CUDA")
        return False


def install_requirements():
    """安装依赖包"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ 错误: requirements.txt文件不存在")
        return False
    
    print("📦 开始安装依赖包...")
    try:
        # 升级pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        
        # 安装依赖
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                      check=True)
        
        print("✅ 依赖包安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False


def create_directories():
    """创建必要的目录"""
    directories = [
        "outputs",
        "models", 
        "logs",
        "yolo_dataset"
    ]
    
    base_path = Path(__file__).parent
    
    for dir_name in directories:
        dir_path = base_path / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    print("✅ 目录创建完成")


def verify_installation():
    """验证安装"""
    print("🔍 验证安装...")
    
    try:
        # 测试主要依赖
        import torch
        import torchvision
        import ultralytics
        import cv2
        import numpy as np
        import yaml
        import matplotlib
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ TorchVision: {torchvision.__version__}")
        print(f"✅ Ultralytics: {ultralytics.__version__}")
        print(f"✅ OpenCV: {cv2.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        
        # 测试项目模块
        sys.path.insert(0, str(Path(__file__).parent))
        from src.utils.common import get_device
        device = get_device()
        print(f"✅ 检测到设备: {device}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def print_system_info():
    """打印系统信息"""
    print("\n" + "="*50)
    print("🖥️  系统信息")
    print("="*50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print("="*50 + "\n")


def print_next_steps():
    """打印后续步骤"""
    print("\n" + "="*50)
    print("🎉 安装完成！")
    print("="*50)
    print("后续步骤:")
    print("1. 准备数据集:")
    print("   - 将训练图像放入 train/train/ 目录")
    print("   - 将训练标注放入 train_label/train_label/ 目录")
    print("   - 将测试图像放入 test/test/ 目录")
    print("")
    print("2. 运行完整流程:")
    print("   python main.py --config configs/config.yaml --mode full")
    print("")
    print("3. 或分步执行:")
    print("   python main.py --config configs/config.yaml --mode prepare")
    print("   python main.py --config configs/config.yaml --mode train")
    print("   python main.py --config configs/config.yaml --mode predict")
    print("")
    print("4. 查看帮助:")
    print("   python main.py --help")
    print("="*50)


def main():
    """主安装流程"""
    print("🚀 车载摄像机目标检测项目安装程序")
    print_system_info()
    
    # 检查环境
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    # 检查CUDA
    check_cuda()
    
    # 验证安装
    if not verify_installation():
        print("⚠️  验证失败，但基本安装可能已完成")
        print("请手动检查依赖包是否正确安装")
    
    # 打印后续步骤
    print_next_steps()


if __name__ == "__main__":
    main()