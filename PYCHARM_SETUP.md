# PyCharm 运行配置设置指南

## 快速设置步骤

### 步骤 1: 打开运行配置

1. 点击 PyCharm 顶部菜单：**Run → Edit Configurations...**
2. 或者点击运行配置下拉菜单旁边的齿轮图标

### 步骤 2: 创建新的运行配置

1. 点击左上角的 **+** 按钮
2. 选择 **Python**

### 步骤 3: 配置参数

**Name**: `Train ResUNet-a`

**Script path**: 
```
D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch\main.py
```

**Parameters** (复制以下内容):
```
--image_path ./data/images --gt_path ./data/gt --image_size 256 --batch_size 8 --epochs 100 --model_save_path ./checkpoints --learning_rate 1e-4 --loss_function tanimoto
```

**Working directory**:
```
D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch
```

**Python interpreter**: 选择您的 Anaconda 环境
```
D:\Tools\Anaconda\envs\pytorch313\python.exe
```

### 步骤 4: 环境变量（可选）

如果需要设置环境变量（如 `KMP_DUPLICATE_LIB_OK=TRUE`）：
1. 展开 "Environment variables"
2. 点击右侧的文件夹图标
3. 添加：
   - Name: `KMP_DUPLICATE_LIB_OK`
   - Value: `TRUE`

### 步骤 5: 保存并运行

1. 点击 **OK** 保存配置
2. 选择配置并点击运行按钮（绿色三角形）

## 参数说明

### 必需参数
- `--image_path`: 训练图像目录
- `--gt_path`: 掩码（ground truth）目录

### 可选参数（有默认值）
- `--image_size`: 图像尺寸（默认：256）
- `--batch_size`: 批次大小（默认：8）
- `--num_classes`: 类别数（默认：2）
- `--epochs`: 训练轮数（默认：100）
- `--model_save_path`: 模型保存路径（默认：`./`）
- `--learning_rate`: 学习率（默认：1e-4）
- `--loss_function`: 损失函数（默认：tanimoto）

## 数据路径设置

### 相对路径（推荐）
如果数据在项目目录下：
```
--image_path ./data/images
--gt_path ./data/gt
```

### 绝对路径
如果数据在其他位置：
```
--image_path D:/data/images
--gt_path D:/data/gt
```

## 常见问题

### Q: 如何修改数据路径？
A: 在 Parameters 中修改 `--image_path` 和 `--gt_path` 的值

### Q: 如何调整批次大小？
A: 修改 `--batch_size` 参数（例如：`--batch_size 16`）

### Q: 如何只训练几个epoch测试？
A: 修改 `--epochs` 参数（例如：`--epochs 5`）

### Q: 如何保存到不同目录？
A: 修改 `--model_save_path` 参数（例如：`--model_save_path ./my_models`）

## 示例配置

### 最小配置（使用所有默认值）
```
--image_path ./data/images --gt_path ./data/gt
```

### 完整配置
```
--image_path ./data/images --gt_path ./data/gt --image_size 512 --batch_size 4 --epochs 50 --model_save_path ./checkpoints --learning_rate 0.0001 --loss_function dice
```

### 快速测试配置（少量epoch）
```
--image_path ./data/images --gt_path ./data/gt --epochs 5 --batch_size 4
```
