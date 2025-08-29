#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复提交文件格式脚本
根据比赛要求重新组织预测文件
"""

import os
import shutil
import zipfile
from pathlib import Path

def fix_submission_format():
    """
    修复提交文件格式
    1. 创建submit文件夹
    2. 将预测文件重命名并移入submit文件夹
    3. 修复坐标格式（转换为整数）
    4. 打包为submit.zip
    """
    
    # 定义路径
    base_dir = Path('.')
    outputs_dir = base_dir / 'outputs'
    submit_dir = base_dir / 'submit'
    test_dir = base_dir / 'test' / 'test'
    
    print("开始修复提交文件格式...")
    
    # 1. 创建submit文件夹
    if submit_dir.exists():
        shutil.rmtree(submit_dir)
    submit_dir.mkdir(exist_ok=True)
    print(f"创建submit文件夹: {submit_dir}")
    
    # 2. 获取测试集图片列表
    test_images = []
    if test_dir.exists():
        test_images = [f.stem for f in test_dir.glob('*.jpg')]
    print(f"找到 {len(test_images)} 张测试图片")
    
    # 3. 处理预测文件
    processed_count = 0
    
    for image_name in test_images:
        # 查找对应的预测文件
        prediction_file = outputs_dir / f'prediction_{image_name}.txt'
        
        if prediction_file.exists():
            # 读取预测文件内容
            with open(prediction_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 修复格式：将坐标转换为整数
            fixed_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 6:
                        class_name = parts[0]
                        confidence = parts[1]
                        left = str(int(float(parts[2])))
                        top = str(int(float(parts[3])))
                        right = str(int(float(parts[4])))
                        bottom = str(int(float(parts[5])))
                        
                        fixed_line = f"{class_name} {confidence} {left} {top} {right} {bottom}"
                        fixed_lines.append(fixed_line)
            
            # 写入submit文件夹，文件名为图片名（不含扩展名）
            output_file = submit_dir / f'{image_name}.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in fixed_lines:
                    f.write(line + '\n')
            
            processed_count += 1
            if processed_count % 20 == 0:
                print(f"已处理 {processed_count} 个文件...")
    
    print(f"总共处理了 {processed_count} 个预测文件")
    
    # 4. 创建submit.zip
    zip_file = base_dir / 'submit.zip'
    if zip_file.exists():
        zip_file.unlink()
    
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for txt_file in submit_dir.glob('*.txt'):
            # 在zip中的路径应该是 submit/文件名.txt
            arcname = f'submit/{txt_file.name}'
            zipf.write(txt_file, arcname)
    
    print(f"创建压缩文件: {zip_file}")
    
    # 5. 验证结果
    print("\n验证结果:")
    print(f"submit文件夹中的文件数量: {len(list(submit_dir.glob('*.txt')))}")
    
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        zip_files = zipf.namelist()
        print(f"zip文件中的文件数量: {len(zip_files)}")
        print(f"zip文件结构示例: {zip_files[:5]}")
    
    # 6. 检查几个文件的格式
    print("\n检查文件格式:")
    sample_files = list(submit_dir.glob('*.txt'))[:3]
    for sample_file in sample_files:
        print(f"\n文件: {sample_file.name}")
        with open(sample_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:3]  # 只显示前3行
            for i, line in enumerate(lines, 1):
                print(f"  行{i}: {line.strip()}")
    
    print("\n修复完成！")
    print(f"提交文件: {zip_file}")
    print("文件结构: submit.zip -> submit/ -> 预测txt文件")

if __name__ == '__main__':
    fix_submission_format()