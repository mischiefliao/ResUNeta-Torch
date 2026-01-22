# ResUNet-a 模型训练完整指南

## 📋 目录

1. [数据准备](#数据准备)
2. [目录结构](#目录结构)
3. [训练方法](#训练方法)
4. [参数说明](#参数说明)
5. [训练示例](#训练示例)
6. [常见问题](#常见问题)

---

## 📁 数据准备

### 1. 数据格式要求

- **图像格式**: PNG, JPG, JPEG, TIF, TIFF（支持大小写）
- **掩码格式**: PNG, JPG, JPEG, TIF, TIFF
- **图像和掩码必须同名**: 
  - 例如：`image1.png` 对应 `image1.png`（掩码）
  - 或者：`image1.tif` 对应 `image1.png`（掩码，扩展名可以不同）
- **掩码应该是二值图像**: 
  - 0（背景）和 255（前景）
  - 或者归一化到 [0, 1] 范围

### 2. 数据组织

您的数据应该按以下方式组织：

```
项目根目录/
├── resunet-a-pytorch/
│   ├── main.py
│   ├── model.py
│   ├── loss.py
│   └── ...
│
└── data/                    # 数据目录（您需要创建）
    ├── images/              # 训练图像目录
    │   ├── image_001.png
    │   ├── image_002.png
    │   ├── image_003.png
    │   └── ...
    │
    └── gt/                  # 对应的掩码（ground truth）目录
        ├── image_001.png    # 必须与images中的文件名对应
        ├── image_002.png
        ├── image_003.png
        └── ...
```

**重要提示**：
- `images/` 和 `gt/` 目录中的文件名必须一一对应
- 如果图像是 `image_001.tif`，掩码可以是 `image_001.png`（扩展名可以不同）

---

## 🚀 训练方法

### 方法1：命令行训练（推荐）

#### 基本训练命令

```bash
cd resunet-a-pytorch

python main.py \
    --image_path ../data/images \
    --gt_path ../data/gt \
    --image_size 256 \
    --batch_size 8 \
    --epochs 100 \
    --model_save_path ./checkpoints
```

#### 完整参数训练命令

```bash
python main.py \
    --image_path ../data/images \
    --gt_path ../data/gt \
    --image_size 256 \
    --batch_size 8 \
    --num_classes 2 \
    --validation_split 0.2 \
    --epochs 100 \
    --layer_norm batch \
    --model_save_path ./checkpoints \
    --checkpoint_mode epochs \
    --learning_rate 1e-4 \
    --loss_function tanimoto
```

### 方法2：PyCharm 中训练

#### 步骤1：创建运行配置

1. 点击 **Run** → **Edit Configurations...**
2. 点击左上角 **+** → 选择 **Python**
3. 配置如下：

   **Name**: `Train ResUNet-a`
   
   **Script path**: `D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch\main.py`
   
   **Parameters**:
   ```
   --image_path ./Data/images --gt_path ./Data/gt --image_size 256 --batch_size 8 --epochs 100 --model_save_path ./checkpoints
   ```
   
   **Working directory**: `D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch`
   
   **Python interpreter**: 选择您的 PyTorch 环境（如 `pytorch313`）

#### 步骤2：运行训练

1. 在顶部运行配置下拉菜单中选择 `Train ResUNet-a`
2. 点击运行按钮（绿色三角形）或按 `Shift+F10`
3. 训练输出会显示在底部的 **Run** 标签页

### 方法3：从 Checkpoint 恢复训练

如果训练被中断，可以从 checkpoint 恢复：

```bash
python main.py \
    --image_path ../data/images \
    --gt_path ../data/gt \
    --epochs 100 \
    --model_save_path ./checkpoints \
    --resume ./checkpoints/interrupted_checkpoint.pth
```

---

## ⚙️ 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--image_path` | 训练图像目录路径 | `./data/images` |
| `--gt_path` | 掩码（ground truth）目录路径 | `./data/gt` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image_size` | 256 | 输入图像大小（高度和宽度） |
| `--batch_size` | 8 | 每个 batch 的图像数量 |
| `--num_classes` | 2 | 输出类别数（2 = 二分类） |
| `--validation_split` | 0.2 | 验证集比例（20%） |
| `--epochs` | 100 | 训练轮数 |
| `--layer_norm` | batch | 归一化类型：`batch`, `instance`, `layer` |
| `--model_save_path` | `./` | 模型保存目录 |
| `--checkpoint_mode` | epochs | Checkpoint 保存模式：`epochs`（每轮保存）或 `best`（只保存最佳） |
| `--learning_rate` | 1e-4 | 学习率（0.0001） |
| `--loss_function` | tanimoto | 损失函数：`bce`, `dice`, `tanimoto` |
| `--resume` | None | 从 checkpoint 恢复训练的路径 |

---

## 📝 训练示例

### 示例1：快速开始（最小配置）

```bash
python main.py \
    --image_path ./Data/images \
    --gt_path ./Data/gt
```

这将使用所有默认参数开始训练。

### 示例2：自定义参数训练

```bash
python main.py \
    --image_path ./Data/images \
    --gt_path ./Data/gt \
    --image_size 512 \
    --batch_size 4 \
    --epochs 200 \
    --learning_rate 5e-5 \
    --loss_function dice \
    --model_save_path ./my_checkpoints
```

### 示例3：使用 GPU 训练

代码会自动检测并使用 GPU（如果可用）。确保：
1. 已安装 CUDA 版本的 PyTorch
2. GPU 驱动已正确安装

训练时会显示：
```
Using device: cuda
```

如果没有 GPU，会使用 CPU：
```
Using device: cpu
```

### 示例4：恢复训练

```bash
# 第一次训练（被 Ctrl+C 中断）
python main.py \
    --image_path ./Data/images \
    --gt_path ./Data/gt \
    --epochs 100 \
    --model_save_path ./checkpoints

# 按 Ctrl+C 后，恢复训练
python main.py \
    --image_path ./Data/images \
    --gt_path ./Data/gt \
    --epochs 100 \
    --model_save_path ./checkpoints \
    --resume ./checkpoints/interrupted_checkpoint.pth
```

---

## 📊 训练输出说明

训练过程中会显示：

```
Using device: cuda
Loading dataset...
Dataset split: 7163 training, 1791 validation
Creating model...
Model parameters: 10,976,674

Starting training...

Epoch 1/100
Training: 100%|████████████| 895/895 [02:30<00:00,  5.95it/s]
Validating: 100%|██████████| 224/224 [00:15<00:00, 14.23it/s]
Train Loss: 0.4523
Val Loss: 0.3891
Train - IoU: 0.6234, Precision: 0.7123, Recall: 0.6891, F1: 0.7005
Val - IoU: 0.6789, Precision: 0.7456, Recall: 0.7234, F1: 0.7344
Best model saved (Val Loss: 0.3891)
Checkpoint saved to ./checkpoints/checkpoint_epoch_1.pth
```

### 输出指标说明

- **Train Loss / Val Loss**: 训练/验证损失（越小越好）
- **IoU**: Intersection over Union（交并比，0-1，越大越好）
- **Precision**: 精确率（0-1，越大越好）
- **Recall**: 召回率（0-1，越大越好）
- **F1**: F1 分数（精确率和召回率的调和平均，0-1，越大越好）

---

## 💾 保存的文件

训练过程中会保存以下文件：

### Checkpoint 文件

- `latest_checkpoint.pth` - 每个 epoch 后的最新状态
- `checkpoint_epoch_N.pth` - 每个 epoch 的完整 checkpoint（如果 `--checkpoint_mode epochs`）
- `best_model.pth` - 验证集上表现最好的模型
- `interrupted_checkpoint.pth` - Ctrl+C 暂停时保存的状态

### 其他文件

- `training_history.png` - 训练历史曲线图（训练完成后生成）

---

## ❓ 常见问题

### Q1: 找不到图像文件

**错误**: `No images found in ./data/images`

**解决**: 
- 检查路径是否正确
- 确保图像文件在指定目录中
- 检查文件扩展名是否支持（PNG, JPG, TIF 等）

### Q2: 找不到对应的掩码文件

**错误**: `Mask not found for image: ...`

**解决**:
- 确保掩码文件名与图像文件名对应
- 检查掩码文件是否在 `gt` 目录中
- 扩展名可以不同，但基础文件名必须相同

### Q3: 内存不足（Out of Memory）

**解决**:
- 减小 `--batch_size`（例如从 8 改为 4 或 2）
- 减小 `--image_size`（例如从 256 改为 128）
- 使用 CPU 训练（虽然较慢，但内存占用更少）

### Q4: 训练很慢

**解决**:
- 确保使用 GPU（检查是否显示 `Using device: cuda`）
- 增大 `--batch_size`（如果内存允许）
- 减少 `--image_size`
- 使用更少的 `--epochs` 先测试

### Q5: 如何查看训练进度？

训练过程中会显示：
- 进度条（每个 epoch）
- 实时损失和指标
- 最佳模型保存提示

训练完成后会生成 `training_history.png` 图表。

### Q6: 如何停止训练？

- 按 `Ctrl+C` 可以安全停止训练
- 会自动保存当前状态到 `interrupted_checkpoint.pth`
- 可以使用 `--resume` 参数恢复训练

---

## 🎯 训练建议

### 对于小数据集（< 1000 张图像）

```bash
python main.py \
    --image_path ./data/images \
    --gt_path ./data/gt \
    --batch_size 4 \
    --epochs 50 \
    --learning_rate 1e-4
```

### 对于中等数据集（1000-10000 张图像）

```bash
python main.py \
    --image_path ./data/images \
    --gt_path ./data/gt \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 1e-4
```

### 对于大数据集（> 10000 张图像）

```bash
python main.py \
    --image_path ./data/images \
    --gt_path ./data/gt \
    --batch_size 16 \
    --epochs 200 \
    --learning_rate 5e-5
```

### 损失函数选择建议

- **`tanimoto`**（默认）: 适合大多数语义分割任务
- **`dice`**: 适合类别不平衡的数据
- **`bce`**: 传统的二分类交叉熵，简单但有效

---

## 📞 获取帮助

如果遇到问题：

1. 检查数据路径和格式是否正确
2. 查看错误信息，通常会有详细说明
3. 确保所有依赖已正确安装：`pip install -r requirements.txt`
4. 检查 PyTorch 和 CUDA 是否正确安装

---

## ✅ 快速检查清单

开始训练前，请确认：

- [ ] 数据已准备好（图像和掩码）
- [ ] 目录结构正确（`images/` 和 `gt/`）
- [ ] 文件名对应（图像和掩码同名）
- [ ] 依赖已安装（`pip install -r requirements.txt`）
- [ ] 路径正确（相对路径或绝对路径）
- [ ] 有足够的磁盘空间保存 checkpoint

现在您可以开始训练了！🚀
