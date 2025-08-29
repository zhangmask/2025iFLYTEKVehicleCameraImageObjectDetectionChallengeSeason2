# 车载摄像机图像目标检测挑战赛

## 项目简介

本项目是基于YOLOv8的车载摄像机图像目标检测解决方案，旨在准确识别和定位车载摄像机拍摄图像中的各类目标对象。项目使用深度学习技术，通过训练优化的YOLOv8模型来实现高精度的目标检测。

## 项目目标

- 实现车载摄像机图像中目标的准确检测和定位
- 优化模型性能，提升检测精度（mAP50目标 > 0.6）
- 生成符合比赛要求的预测结果文件

## 项目结构

```
车载摄像机图像的目标检测挑战赛 赛季2/
├── config.yaml                 # 模型训练配置文件
├── fix_submission.py           # 提交文件修复脚本
├── submit.zip                  # 最终提交文件
├── submit/                     # 预测结果文件夹
│   ├── *.txt                   # 各图像的预测结果文件
├── outputs/                    # 训练输出目录
│   └── yolo_data/
│       └── models/
│           └── yolov8_vehicle_detection/
│               ├── weights/    # 模型权重文件
│               ├── results.csv # 训练结果记录
│               └── *.png       # 训练可视化图表
├── logs/                       # 训练日志
│   └── training.log
└── README.md                   # 项目说明文档
```

## 环境要求

### 系统要求
- Python 3.8+
- CUDA 11.0+ (推荐使用GPU加速)
- 至少8GB内存
- 至少10GB可用磁盘空间

### 依赖安装

```bash
# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装YOLOv8
pip install ultralytics

# 安装其他依赖
pip install opencv-python pillow numpy pandas matplotlib seaborn
```

## 数据准备

### 数据格式
- 训练数据：图像文件 + YOLO格式标注文件
- 测试数据：仅图像文件
- 支持的图像格式：jpg, jpeg, png

### 数据预处理
1. 确保标注文件格式正确（YOLO格式）
2. 检查图像和标注文件的对应关系
3. 验证类别标签的一致性

## 模型训练

### 配置文件说明

`config.yaml` 包含所有训练参数：

```yaml
model:
  name: yolov8l              # 模型类型
  size: yolov8l              # 模型大小
  input_size: 640            # 输入图像尺寸
  confidence_threshold: 0.25  # 置信度阈值
  iou_threshold: 0.45        # IoU阈值

train:
  epochs: 300                # 训练轮数
  batch_size: 4              # 批次大小
  learning_rate: 0.001       # 学习率
  optimizer: AdamW           # 优化器
  weight_decay: 0.0005       # 权重衰减
  momentum: 0.937            # 动量
  warmup_epochs: 3           # 预热轮数
  patience: 50               # 早停耐心值
  save_period: 10            # 保存周期
  device: 0                  # GPU设备ID
  workers: 4                 # 数据加载线程数
```

### 训练步骤

1. **准备配置文件**
   ```bash
   # 检查config.yaml配置是否正确
   ```

2. **启动训练**
   ```bash
   python -c "
   import yaml
   from ultralytics import YOLO
   
   # 加载配置
   with open('config.yaml', 'r', encoding='utf-8') as f:
       config = yaml.safe_load(f)
   
   # 初始化模型
   model = YOLO(config['model']['name'] + '.pt')
   
   # 开始训练
   results = model.train(
       data='path/to/dataset.yaml',
       epochs=config['train']['epochs'],
       batch=config['train']['batch_size'],
       lr0=config['train']['learning_rate'],
       optimizer=config['train']['optimizer'],
       weight_decay=config['train']['weight_decay'],
       momentum=config['train']['momentum'],
       warmup_epochs=config['train']['warmup_epochs'],
       patience=config['train']['patience'],
       save_period=config['train']['save_period'],
       device=config['train']['device'],
       workers=config['train']['workers'],
       project='outputs/yolo_data/models',
       name='yolov8_vehicle_detection'
   )
   "
   ```

3. **监控训练过程**
   - 查看训练日志：`logs/training.log`
   - 查看训练曲线：`outputs/yolo_data/models/yolov8_vehicle_detection/*.png`
   - 查看训练结果：`outputs/yolo_data/models/yolov8_vehicle_detection/results.csv`

## 推理和预测

### 使用最佳模型进行推理

```python
from ultralytics import YOLO
import os

# 加载最佳权重
model = YOLO('outputs/yolo_data/models/yolov8_vehicle_detection/weights/best.pt')

# 对测试集进行推理
test_images_dir = 'path/to/test/images'
output_dir = 'submit'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 批量推理
for image_file in os.listdir(test_images_dir):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_images_dir, image_file)
        
        # 进行预测
        results = model(image_path)
        
        # 保存预测结果
        result_file = os.path.join(output_dir, 
                                 os.path.splitext(image_file)[0] + '.txt')
        
        with open(result_file, 'w') as f:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 格式：class_id confidence x_center y_center width height
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x, y, w, h = box.xywhn[0].tolist()
                        f.write(f"{cls} {conf:.6f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
```

## 提交文件生成

### 打包预测结果

```python
import zipfile
import os

def create_submission_zip():
    """创建提交用的zip文件"""
    with zipfile.ZipFile('submit.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        submit_dir = 'submit'
        for root, dirs, files in os.walk(submit_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, submit_dir)
                    zipf.write(file_path, arcname)
    
    print(f"提交文件 submit.zip 创建完成")

# 执行打包
create_submission_zip()
```

### 验证提交文件

```python
def validate_submission():
    """验证提交文件格式"""
    import zipfile
    
    with zipfile.ZipFile('submit.zip', 'r') as zipf:
        files = zipf.namelist()
        print(f"提交文件包含 {len(files)} 个预测结果文件")
        
        # 检查文件格式
        for file in files[:5]:  # 显示前5个文件作为示例
            print(f"文件: {file}")
            with zipf.open(file) as f:
                content = f.read().decode('utf-8')
                lines = content.strip().split('\n')
                if lines and lines[0]:
                    print(f"  预测数量: {len(lines)}")
                    print(f"  示例: {lines[0]}")

# 执行验证
validate_submission()
```

## 性能指标和结果

### 当前模型性能

- **模型**: YOLOv8l
- **训练轮数**: 300 epochs
- **最佳mAP50**: 0.47085
- **最佳mAP50-95**: 0.21+
- **训练时间**: 约数小时（取决于硬件配置）

### 性能指标说明

- **mAP50**: IoU阈值为0.5时的平均精度
- **mAP50-95**: IoU阈值从0.5到0.95的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

### 结果文件

- `results.csv`: 详细的训练指标记录
- `confusion_matrix.png`: 混淆矩阵
- `F1_curve.png`: F1曲线
- `P_curve.png`: 精确率曲线
- `R_curve.png`: 召回率曲线
- `PR_curve.png`: PR曲线

## 使用示例

### 快速开始

```bash
# 1. 克隆项目
cd "车载摄像机图像的目标检测挑战赛 赛季2"

# 2. 安装依赖
pip install ultralytics opencv-python pillow numpy pandas

# 3. 准备数据（将数据放在相应目录）

# 4. 开始训练
python train_model.py

# 5. 进行推理
python inference.py

# 6. 生成提交文件
python fix_submission.py
```

### 自定义配置

修改 `config.yaml` 文件来调整训练参数：

```yaml
# 使用更大的模型
model:
  name: yolov8x  # 改为yolov8x获得更高精度

# 调整训练参数
train:
  epochs: 500    # 增加训练轮数
  batch_size: 2  # 减小批次大小（如果显存不足）
  learning_rate: 0.0005  # 调整学习率
```

## 注意事项

### 训练注意事项

1. **显存管理**
   - YOLOv8l模型需要较大显存，建议使用8GB+显存的GPU
   - 如果显存不足，可以减小batch_size或使用YOLOv8m模型

2. **数据质量**
   - 确保标注数据的准确性
   - 检查图像和标注文件的对应关系
   - 验证类别标签的一致性

3. **训练监控**
   - 定期检查训练日志和损失曲线
   - 注意过拟合现象，适当调整正则化参数
   - 使用验证集监控模型性能

### 推理注意事项

1. **模型选择**
   - 使用 `best.pt` 权重文件进行最终推理
   - 确保模型配置与训练时一致

2. **结果格式**
   - 预测结果必须符合比赛要求的格式
   - 坐标使用归一化格式（0-1之间）
   - 置信度保留6位小数

3. **文件管理**
   - 确保所有测试图像都有对应的预测文件
   - 检查文件命名的一致性
   - 验证zip文件的结构

### 性能优化建议

1. **模型优化**
   - 尝试使用YOLOv8x模型获得更高精度
   - 调整NMS参数优化检测结果
   - 使用模型集成技术

2. **数据增强**
   - 增加更多的数据增强策略
   - 调整增强强度和概率
   - 使用Mixup、CutMix等高级增强技术

3. **训练策略**
   - 使用余弦退火学习率调度
   - 实施渐进式训练
   - 尝试不同的优化器和超参数

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案：减小batch_size或使用更小的模型
   ```

2. **训练中断**
   ```
   解决方案：检查数据路径和配置文件，使用resume功能继续训练
   ```

3. **预测结果格式错误**
   ```
   解决方案：检查坐标转换和文件保存格式
   ```

### 联系信息

如有问题，请检查：
1. 配置文件是否正确
2. 数据路径是否存在
3. 依赖包是否完整安装
4. GPU驱动和CUDA版本是否兼容

---

**项目版本**: v1.0  
**最后更新**: 2024年1月  
**兼容性**: Python 3.8+, PyTorch 1.9+, CUDA 11.0+