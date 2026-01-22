# ResUNet-a PyTorch 运行指南

## 目录结构

首先，您需要准备以下目录结构：

```
resunet-a-pytorch/
├── model.py
├── loss.py
├── main.py
├── predict.py
├── batch_preprocess.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/                    # 数据目录（您需要创建）
│   ├── images/              # 训练图像目录
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── gt/                  # 对应的掩码（ground truth）目录
│       ├── image1.png       # 必须与images中的文件名对应
│       ├── image2.png
│       └── ...
│
├── test_images/             # 测试图像目录（预测时使用）
│   ├── test1.png
│   └── ...
│
└── results/                 # 预测结果保存目录（会自动创建）
    └── ...
```

## 1. 安装依赖

```bash
cd TransKalambaUnet++
pip install -r requirements.txt
```

## 2. 准备数据

### 2.1 数据格式要求

- **图像格式**: 支持 PNG, JPG, JPEG, TIF, TIFF
- **掩码格式**: 支持 PNG, JPG, JPEG, TIF, TIFF
- **图像和掩码必须同名**: 例如 `image1.png` 对应 `image1.png`（掩码）
- **掩码应该是二值图像**: 0（背景）和 255（前景），或归一化到 [0, 1]

### 2.2 数据目录结构示例

```
data/
├── images/          # 训练图像
│   ├── img_001.png
│   ├── img_002.png
│   └── img_003.png
└── gt/              # 对应的掩码
    ├── img_001.png  # 与images中的文件名相同
    ├── img_002.png
    └── img_003.png
```

### 2.3 数据读取位置

数据读取的代码位置：
- **训练数据读取**: `batch_preprocess.py` 中的 `load_dataset()` 函数
- **具体代码行**: `batch_preprocess.py` 第 123-223 行
- **读取逻辑**:
  1. 扫描 `--image_path` 目录中的所有图像文件
  2. 在 `--gt_path` 目录中查找对应的掩码文件（通过文件名匹配）
  3. 自动按 `validation_split` 比例分割训练集和验证集

## 3. 运行训练

### 3.1 基本训练命令

```bash
python main.py \
    --image_path ./data/images \
    --gt_path ./data/gt \
    --image_size 256 \
    --batch_size 8 \
    --num_classes 2 \
    --validation_split 0.2 \
    --epochs 100 \
    --model_save_path ./checkpoints \
    --checkpoint_mode epochs \
    --learning_rate 1e-4 \
    --loss_function tanimoto
```

### 3.2 参数说明

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--image_path` | 训练图像目录路径 | - | ✅ |
| `--gt_path` | 掩码（ground truth）目录路径 | - | ✅ |
| `--image_size` | 输入图像尺寸 | 256 | ❌ |
| `--batch_size` | 批次大小 | 8 | ❌ |
| `--num_classes` | 输出类别数（2=二分类） | 2 | ❌ |
| `--validation_split` | 验证集比例 | 0.2 | ❌ |
| `--epochs` | 训练轮数 | 100 | ❌ |
| `--model_save_path` | 模型保存目录 | ./ | ❌ |
| `--checkpoint_mode` | 保存模式：'epochs'或'best' | epochs | ❌ |
| `--learning_rate` | 学习率 | 1e-4 | ❌ |
| `--loss_function` | 损失函数：'bce', 'dice', 'tanimoto' | tanimoto | ❌ |
| `--layer_norm` | 归一化类型：'batch', 'instance', 'layer' | batch | ❌ |

### 3.3 训练输出

训练过程中会：
1. 自动创建 `--model_save_path` 目录
2. 保存最佳模型到 `best_model.pth`
3. 根据 `checkpoint_mode` 保存定期checkpoint
4. 生成训练历史图表 `training_history.png`

## 4. 运行预测

### 4.1 基本预测命令

```bash
python predict.py \
    --image_path ./test_images \
    --model_path ./checkpoints/best_model.pth \
    --output_path ./results \
    --image_size 256 \
    --num_classes 2
```

### 4.2 参数说明

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--image_path` | 测试图像目录路径 | - | ✅ |
| `--model_path` | 训练好的模型文件路径（.pth） | - | ✅ |
| `--output_path` | 预测结果保存目录 | - | ✅ |
| `--image_size` | 输入图像尺寸（需与训练时一致） | 256 | ❌ |
| `--num_classes` | 类别数（需与训练时一致） | 2 | ❌ |

### 4.3 预测输出

- 预测结果会保存在 `--output_path` 目录
- 文件名格式：`原文件名_pred.png`
- 例如：`test1.png` → `test1_pred.png`

## 5. 完整示例

### 示例1: 快速开始（使用默认参数）

```bash
# 1. 准备数据（假设数据在 data/images 和 data/gt）
# 2. 训练
python main.py --image_path ./data/images --gt_path ./data/gt

# 3. 预测
python predict.py \
    --image_path ./test_images \
    --model_path ./best_model.pth \
    --output_path ./results
```

### 示例2: 自定义参数训练

```bash
python main.py \
    --image_path ./data/images \
    --gt_path ./data/gt \
    --image_size 512 \
    --batch_size 4 \
    --epochs 50 \
    --model_save_path ./my_models \
    --learning_rate 0.0001 \
    --loss_function dice
```

### 示例3: 使用GPU训练

代码会自动检测GPU，如果有CUDA可用会自动使用GPU。确保：
1. 安装了CUDA版本的PyTorch
2. GPU驱动正确安装

```bash
# 检查GPU是否可用（在Python中）
python -c "import torch; print(torch.cuda.is_available())"
```

## 6. 数据读取详细说明

### 6.1 数据读取代码位置

- **文件**: `batch_preprocess.py`
- **函数**: `load_dataset()` (第 123-223 行)
- **类**: `SegmentationDataset` (第 12-87 行)

### 6.2 数据读取流程

1. **扫描图像目录** (`batch_preprocess.py` 第 149-153 行)
   ```python
   # 扫描所有支持的图像格式
   image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', ...]
   ```

2. **匹配掩码文件** (`batch_preprocess.py` 第 159-177 行)
   ```python
   # 通过文件名匹配对应的掩码
   # 例如: images/img_001.png → gt/img_001.png
   ```

3. **数据集分割** (`batch_preprocess.py` 第 179-184 行)
   ```python
   # 按 validation_split 比例分割
   # 例如: 80% 训练, 20% 验证
   ```

4. **数据加载** (`batch_preprocess.py` 第 193-221 行)
   ```python
   # 创建 PyTorch DataLoader
   # 支持数据增强、批处理等
   ```

### 6.3 数据预处理

数据在 `SegmentationDataset.__getitem__()` 中处理：
- 图像：BGR → RGB，resize，归一化，转换为张量 (C, H, W)
- 掩码：灰度图，resize，二值化，转换为张量 (1, H, W)

## 7. 常见问题

### Q1: 找不到掩码文件怎么办？

**A**: 确保：
1. 图像和掩码文件名完全一致（包括扩展名）
2. 掩码文件在 `--gt_path` 目录中
3. 支持的格式：PNG, JPG, JPEG, TIF, TIFF

### Q2: 内存不足怎么办？

**A**: 尝试：
- 减小 `--batch_size`（例如改为 4 或 2）
- 减小 `--image_size`（例如改为 128）
- 减少 `--num_workers`（在代码中修改）

### Q3: 训练很慢怎么办？

**A**: 
- 使用GPU（确保安装了CUDA版本的PyTorch）
- 增加 `batch_size`（如果内存允许）
- 使用更小的 `image_size`

### Q4: 如何查看训练进度？

**A**: 
- 训练过程中会实时打印损失和指标
- 训练结束后会生成 `training_history.png` 图表

## 8. 测试代码

运行基础测试确保代码正常工作：

```bash
python test_basic.py
```

这会测试：
- 模型前向传播
- 损失函数
- 指标计算
- 训练步骤

## 9. 数据准备检查清单

在开始训练前，请确认：

- [ ] 数据目录结构正确
- [ ] 图像和掩码文件名匹配
- [ ] 掩码是二值图像（0和255，或0和1）
- [ ] 图像格式支持（PNG/JPG/TIF等）
- [ ] 有足够的磁盘空间保存模型
- [ ] 安装了所有依赖包

## 10. 快速开始命令总结

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（创建 data/images 和 data/gt 目录，放入图像和掩码）

# 3. 训练模型
python main.py --image_path ./data/images --gt_path ./data/gt

# 4. 预测（训练完成后）
python predict.py \
    --image_path ./test_images \
    --model_path ./best_model.pth \
    --output_path ./results
```
