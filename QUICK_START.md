# 快速开始 - 如何运行训练

## ⚠️ 重要：不要使用 Python Console！

**Python Console** 是交互式控制台，用于测试代码片段，**不能运行需要命令行参数的脚本**。

## ✅ 正确的运行方法

### 方法1：使用 PyCharm 运行配置（推荐）

#### 步骤1：创建运行配置

1. **关闭 Python Console**（如果打开了）
2. 点击 **Run** → **Edit Configurations...**
3. 点击左上角 **+** → 选择 **Python**
4. 配置如下：

   **Name**: `Train ResUNet-a`
   
   **Script path**: 
   ```
   D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch\main.py
   ```
   
   **Parameters**:
   ```
   --image_path ./Data/images --gt_path ./Data/gt --image_size 256 --batch_size 8 --epochs 100 --model_save_path ./checkpoints
   ```
   
   **Working directory**:
   ```
   D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch
   ```
   
   **Python interpreter**: 选择 `pytorch313` 环境

#### 步骤2：运行

1. 在顶部运行配置下拉菜单中选择 `Train ResUNet-a`
2. 点击运行按钮（绿色三角形）或按 `Shift+F10`
3. **查看底部 "Run" 标签页**（不是 Python Console）

### 方法2：使用 Terminal（命令行）

1. 打开 Terminal：
   - View → Tool Windows → Terminal
   - 或点击底部 **Terminal** 标签

2. 切换到项目目录：
   ```bash
   cd "D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch"
   ```

3. 运行训练命令：
   ```bash
   python main.py --image_path ./Data/images --gt_path ./Data/gt --image_size 256 --batch_size 8 --epochs 100 --model_save_path ./checkpoints
   ```

## 📋 参数说明

### 必需参数

- `--image_path`: 训练图像目录
  - 示例：`./Data/images`
  
- `--gt_path`: 掩码（ground truth）目录
  - 示例：`./Data/gt`

### 可选参数（有默认值）

- `--image_size 256`: 图像大小
- `--batch_size 8`: 批次大小
- `--epochs 100`: 训练轮数
- `--model_save_path ./checkpoints`: 模型保存目录

## 🔍 如何判断是否在运行？

### ✅ 正确的运行方式会显示：

```
Using device: cpu
Loading dataset...
Dataset split: 7163 training, 1791 validation
Creating model...
Model parameters: 10,976,674

Starting training...

Epoch 1/100
Training: 0%|          | 0/895 [00:00<?, ?it/s]
```

### ❌ Python Console 只会显示：

```
>>> 
```

（等待您输入代码，不会运行脚本）

## ❓ 常见问题

### Q1: 为什么没有反应？

**A**: 您可能是在 Python Console 中运行，而不是使用运行配置。

**解决**：
1. 关闭 Python Console
2. 使用运行配置或 Terminal 运行

### Q2: 提示缺少参数？

**错误**: `main.py: error: the following arguments are required: --image_path, --gt_path`

**解决**: 确保在 Parameters 中添加了 `--image_path` 和 `--gt_path`

### Q3: 找不到数据？

**错误**: `No images found in ./Data/images`

**解决**: 
- 检查路径是否正确
- 确保数据在 `./Data/images` 和 `./Data/gt` 目录中

### Q4: 如何查看训练输出？

训练输出会显示在：
- **Run 标签页**（使用运行配置时）
- **Terminal**（使用命令行时）

**不是** Python Console！

## 🎯 快速检查清单

开始训练前，确认：

- [ ] 已关闭 Python Console
- [ ] 已创建运行配置或使用 Terminal
- [ ] Parameters 中包含 `--image_path` 和 `--gt_path`
- [ ] 数据已准备好
- [ ] 查看 **Run** 标签页（不是 Python Console）

## 📝 完整示例

### 最小配置（使用默认参数）

```bash
python main.py --image_path ./Data/images --gt_path ./Data/gt
```

### 完整配置

```bash
python main.py \
    --image_path ./Data/images \
    --gt_path ./Data/gt \
    --image_size 256 \
    --batch_size 8 \
    --epochs 100 \
    --model_save_path ./checkpoints \
    --learning_rate 1e-4 \
    --loss_function tanimoto
```

---

**记住：使用运行配置或 Terminal，不要使用 Python Console！** 🚀
