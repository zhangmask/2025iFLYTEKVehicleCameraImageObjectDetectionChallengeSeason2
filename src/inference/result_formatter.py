#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比赛格式结果输出模块
生成符合比赛要求的txt格式结果文件
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Any
from .predictor import Detection

class CompetitionFormatter:
    """
    比赛格式结果格式化器
    """
    
    def __init__(self, output_dir: str):
        """
        初始化格式化器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """
        设置日志记录
        """
        self.logger = logging.getLogger(__name__)
    
    def format_single_image(self, image_name: str, detections: List[Detection]) -> str:
        """
        格式化单张图像的检测结果
        
        Args:
            image_name: 图像文件名（不含路径）
            detections: 检测结果列表
            
        Returns:
            格式化后的结果字符串
        """
        lines = []
        
        for detection in detections:
            # 比赛格式: <class_name> <confidence> <left> <top> <right> <bottom>
            line = f"{detection.class_name} {detection.confidence:.6f} " \
                   f"{detection.bbox[0]:.1f} {detection.bbox[1]:.1f} " \
                   f"{detection.bbox[2]:.1f} {detection.bbox[3]:.1f}"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def save_predictions_to_txt(self, predictions: Dict[str, List[Detection]], 
                               prefix: str = 'prediction') -> List[str]:
        """
        将预测结果保存为txt文件
        
        Args:
            predictions: 预测结果字典 {图像路径: 检测结果列表}
            prefix: 文件名前缀
            
        Returns:
            生成的txt文件路径列表
        """
        txt_files = []
        
        try:
            self.logger.info(f"开始保存 {len(predictions)} 个预测结果为txt格式")
            
            for image_path, detections in predictions.items():
                # 获取图像文件名（不含扩展名）
                image_name = Path(image_path).stem
                
                # 生成txt文件名
                txt_filename = f"{prefix}_{image_name}.txt"
                txt_filepath = self.output_dir / txt_filename
                
                # 格式化检测结果
                formatted_result = self.format_single_image(image_name, detections)
                
                # 保存到文件
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(formatted_result)
                
                txt_files.append(str(txt_filepath))
                
                self.logger.debug(f"保存预测结果: {txt_filepath} ({len(detections)} 个检测)")
            
            self.logger.info(f"所有预测结果已保存，共生成 {len(txt_files)} 个txt文件")
            return txt_files
            
        except Exception as e:
            self.logger.error(f"保存预测结果失败: {e}")
            raise
    
    def create_submission_zip(self, txt_files: List[str], zip_name: str = 'submit.zip') -> str:
        """
        创建提交用的zip文件
        
        Args:
            txt_files: txt文件路径列表
            zip_name: zip文件名
            
        Returns:
            zip文件路径
        """
        try:
            zip_path = self.output_dir / zip_name
            
            self.logger.info(f"创建提交zip文件: {zip_path}")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for txt_file in txt_files:
                    # 只保存文件名，不包含路径
                    arcname = Path(txt_file).name
                    zipf.write(txt_file, arcname)
                    self.logger.debug(f"添加文件到zip: {arcname}")
            
            self.logger.info(f"提交zip文件创建完成: {zip_path} (包含 {len(txt_files)} 个文件)")
            return str(zip_path)
            
        except Exception as e:
            self.logger.error(f"创建提交zip文件失败: {e}")
            raise
    
    def process_test_predictions(self, test_predictions: Dict[str, List[Detection]], 
                               create_zip: bool = True) -> Dict[str, str]:
        """
        处理测试集预测结果，生成比赛提交文件
        
        Args:
            test_predictions: 测试集预测结果
            create_zip: 是否创建zip文件
            
        Returns:
            包含文件路径信息的字典
        """
        try:
            # 保存为txt文件
            txt_files = self.save_predictions_to_txt(test_predictions, prefix='prediction')
            
            result_info = {
                'txt_files': txt_files,
                'output_dir': str(self.output_dir),
                'total_files': len(txt_files)
            }
            
            # 创建提交zip文件
            if create_zip and txt_files:
                zip_path = self.create_submission_zip(txt_files)
                result_info['zip_file'] = zip_path
            
            # 生成统计信息
            total_detections = sum(len(dets) for dets in test_predictions.values())
            result_info['total_detections'] = total_detections
            
            # 按类别统计
            class_counts = {}
            for detections in test_predictions.values():
                for detection in detections:
                    class_name = detection.class_name
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            result_info['class_statistics'] = class_counts
            
            self.logger.info(f"测试集预测结果处理完成:")
            self.logger.info(f"  - 图像数量: {len(test_predictions)}")
            self.logger.info(f"  - 检测总数: {total_detections}")
            self.logger.info(f"  - 类别统计: {class_counts}")
            
            return result_info
            
        except Exception as e:
            self.logger.error(f"处理测试集预测结果失败: {e}")
            raise
    
    def validate_submission_format(self, zip_path: str) -> Dict[str, Any]:
        """
        验证提交文件格式
        
        Args:
            zip_path: zip文件路径
            
        Returns:
            验证结果字典
        """
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'file_count': 0,
                'total_detections': 0
            }
            
            if not os.path.exists(zip_path):
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"zip文件不存在: {zip_path}")
                return validation_result
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                validation_result['file_count'] = len(file_list)
                
                # 检查文件格式
                for filename in file_list:
                    if not filename.endswith('.txt'):
                        validation_result['warnings'].append(f"非txt文件: {filename}")
                        continue
                    
                    # 读取文件内容并验证格式
                    try:
                        content = zipf.read(filename).decode('utf-8')
                        lines = content.strip().split('\n')
                        
                        if content.strip():  # 非空文件
                            for line_num, line in enumerate(lines, 1):
                                if line.strip():  # 非空行
                                    parts = line.strip().split()
                                    if len(parts) != 6:
                                        validation_result['errors'].append(
                                            f"{filename} 第{line_num}行格式错误: 应为6个字段")
                                        validation_result['is_valid'] = False
                                    else:
                                        # 验证类别名称
                                        class_name = parts[0]
                                        valid_classes = ['car', 'truck', 'bike', 'human']
                                        if class_name not in valid_classes:
                                            validation_result['warnings'].append(
                                                f"{filename} 第{line_num}行: 未知类别 '{class_name}'")
                                        
                                        # 验证数值格式
                                        try:
                                            confidence = float(parts[1])
                                            bbox = [float(x) for x in parts[2:6]]
                                            
                                            if not (0 <= confidence <= 1):
                                                validation_result['warnings'].append(
                                                    f"{filename} 第{line_num}行: 置信度超出范围 [0,1]")
                                            
                                            validation_result['total_detections'] += 1
                                            
                                        except ValueError:
                                            validation_result['errors'].append(
                                                f"{filename} 第{line_num}行: 数值格式错误")
                                            validation_result['is_valid'] = False
                    
                    except Exception as e:
                        validation_result['errors'].append(f"读取文件 {filename} 失败: {e}")
                        validation_result['is_valid'] = False
            
            self.logger.info(f"提交文件验证完成: {'通过' if validation_result['is_valid'] else '失败'}")
            if validation_result['errors']:
                self.logger.error(f"验证错误: {validation_result['errors']}")
            if validation_result['warnings']:
                self.logger.warning(f"验证警告: {validation_result['warnings']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"验证提交文件格式失败: {e}")
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'file_count': 0,
                'total_detections': 0
            }
    
    def generate_submission_summary(self, result_info: Dict[str, Any]) -> str:
        """
        生成提交摘要报告
        
        Args:
            result_info: 结果信息字典
            
        Returns:
            摘要报告字符串
        """
        summary_lines = [
            "=" * 50,
            "车载摄像机目标检测 - 提交摘要",
            "=" * 50,
            f"输出目录: {result_info.get('output_dir', 'N/A')}",
            f"生成文件数: {result_info.get('total_files', 0)}",
            f"检测总数: {result_info.get('total_detections', 0)}",
            "",
            "类别统计:"
        ]
        
        class_stats = result_info.get('class_statistics', {})
        for class_name, count in class_stats.items():
            summary_lines.append(f"  {class_name}: {count}")
        
        if 'zip_file' in result_info:
            summary_lines.extend([
                "",
                f"提交文件: {result_info['zip_file']}",
                "请将此zip文件提交到比赛平台"
            ])
        
        summary_lines.append("=" * 50)
        
        return '\n'.join(summary_lines)


def main():
    """
    主函数 - 用于测试格式化器
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='比赛格式结果输出工具')
    parser.add_argument('--validate', type=str, help='验证zip文件格式')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        formatter = CompetitionFormatter(args.output_dir)
        
        if args.validate:
            # 验证提交文件格式
            result = formatter.validate_submission_format(args.validate)
            
            print(f"验证结果: {'通过' if result['is_valid'] else '失败'}")
            print(f"文件数量: {result['file_count']}")
            print(f"检测总数: {result['total_detections']}")
            
            if result['errors']:
                print("\n错误:")
                for error in result['errors']:
                    print(f"  - {error}")
            
            if result['warnings']:
                print("\n警告:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
        
        else:
            print("请使用 --validate 参数指定要验证的zip文件")
        
    except Exception as e:
        print(f"操作失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())