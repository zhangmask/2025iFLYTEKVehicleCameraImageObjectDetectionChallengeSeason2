#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复预测文件格式
将坐标从浮点数转换为整数，确保符合比赛要求
"""

import os
import glob
import zipfile
from pathlib import Path

def fix_prediction_format(input_dir, output_dir):
    """
    修复预测文件格式
    
    Args:
        input_dir: 输入目录（解压后的预测文件）
        output_dir: 输出目录（修复后的预测文件）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有预测文件
    prediction_files = glob.glob(os.path.join(input_dir, "prediction_*.txt"))
    
    print(f"找到 {len(prediction_files)} 个预测文件")
    
    fixed_count = 0
    
    for file_path in prediction_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) != 6:
                    print(f"警告：文件 {filename} 中的行格式不正确: {line}")
                    continue
                
                class_name = parts[0]
                confidence = float(parts[1])
                left = int(float(parts[2]))
                top = int(float(parts[3]))
                right = int(float(parts[4]))
                bottom = int(float(parts[5]))
                
                # 验证类别名称
                valid_classes = ['car', 'truck', 'bike', 'human']
                if class_name not in valid_classes:
                    print(f"警告：文件 {filename} 中发现无效类别: {class_name}")
                    continue
                
                # 确保坐标合理
                if left < 0 or top < 0 or right <= left or bottom <= top:
                    print(f"警告：文件 {filename} 中坐标不合理: {left} {top} {right} {bottom}")
                    continue
                
                # 格式化为正确格式：<class_name> <confidence> <left> <top> <right> <bottom>
                fixed_line = f"{class_name} {confidence:.6f} {left} {top} {right} {bottom}"
                fixed_lines.append(fixed_line)
            
            # 写入修复后的文件
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in fixed_lines:
                    f.write(line + '\n')
            
            fixed_count += 1
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
    
    print(f"成功修复 {fixed_count} 个预测文件")
    return fixed_count

def create_submission_zip(prediction_dir, output_zip_path):
    """
    创建提交用的zip文件
    
    Args:
        prediction_dir: 预测文件目录
        output_zip_path: 输出zip文件路径
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        prediction_files = glob.glob(os.path.join(prediction_dir, "prediction_*.txt"))
        
        for file_path in prediction_files:
            filename = os.path.basename(file_path)
            zipf.write(file_path, filename)
    
    print(f"创建提交文件: {output_zip_path}")
    print(f"包含 {len(prediction_files)} 个预测文件")

def main():
    # 设置路径
    input_dir = "outputs/extracted"
    output_dir = "outputs/fixed_predictions"
    output_zip = "outputs/submit_fixed.zip"
    
    print("开始修复预测文件格式...")
    
    # 修复预测文件格式
    fixed_count = fix_prediction_format(input_dir, output_dir)
    
    if fixed_count > 0:
        # 创建新的提交文件
        create_submission_zip(output_dir, output_zip)
        print("\n格式修复完成！")
        print(f"修复后的预测文件保存在: {output_dir}")
        print(f"新的提交文件: {output_zip}")
    else:
        print("没有文件被修复")

if __name__ == "__main__":
    main()