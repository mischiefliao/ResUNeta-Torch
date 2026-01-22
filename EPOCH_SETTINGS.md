# Epoch 设置位置完整说明

## 📋 项目中所有与 Epoch 相关的设置位置

### 1. 命令行参数设置（主要设置位置）

**文件**: `main.py`  
**位置**: 第 641 行

```python
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
```

**说明**:
- 这是**主要的 epoch 设置位置**
- 默认值：`100`
- 可以通过命令行参数修改：`--epochs 200`

**使用方法**:
```bash
python main.py --image_path ./Data/images --gt_path ./Data/gt --epochs 200
```

---

### 2. 训练循环中的 Epoch 控制

**文件**: `main.py`  
**位置**: 第 500 行

```python
for epoch in range(start_epoch, args.epochs):
    print(f"\nEpoch {epoch+1}/{args.epochs}")
```

**说明**:
- `start_epoch`: 起始 epoch（新训练时为 0，恢复训练时为上次停止的 epoch+1）
- `args.epochs`: 总 epoch 数（从命令行参数获取）
- 训练范围：`range(start_epoch, args.epochs)`

**示例**:
- 新训练：`range(0, 100)` → 训练 epoch 0-99（显示为 1-100）
- 从第 50 个 epoch 恢复：`range(50, 100)` → 训练 epoch 50-99（显示为 51-100）

---

### 3. Checkpoint 中保存的 Epoch 信息

**文件**: `utils.py`  
**位置**: 第 216 行

```python
checkpoint = {
    'epoch': epoch,  # 当前完成的 epoch 编号（从 0 开始）
    ...
}
```

**说明**:
- Checkpoint 文件中保存的是**已完成的 epoch 编号**（从 0 开始）
- 例如：第 100 个 epoch 完成后，checkpoint 中保存 `epoch = 99`

---

### 4. 从 Checkpoint 恢复时的 Epoch 计算

**文件**: `main.py`  
**位置**: 第 471 行

```python
start_epoch = checkpoint.get('epoch', 0) + 1
```

**说明**:
- 从 checkpoint 中读取 `epoch` 值（已完成的 epoch）
- `+1` 表示从下一个 epoch 开始继续训练
- 例如：checkpoint 中 `epoch = 99`（已完成第 100 个 epoch），则 `start_epoch = 100`（从第 101 个 epoch 开始）

---

### 5. Checkpoint 文件名中的 Epoch

**文件**: `main.py`  
**位置**: 第 550 行

```python
checkpoint_path = os.path.join(args.model_save_path, f'checkpoint_epoch_{epoch+1}.pth')
```

**说明**:
- Checkpoint 文件名使用 `epoch+1`（因为显示给用户的是从 1 开始）
- 例如：`checkpoint_epoch_100.pth` 表示第 100 个 epoch 的 checkpoint

---

### 6. 批处理文件中的 Epoch 设置

**文件**: `run_train.bat`  
**位置**: 第 10 行

```batch
--epochs 100 ^
```

**说明**:
- Windows 批处理文件中的默认 epoch 设置
- 可以修改此文件来改变默认值

---

### 7. 训练历史记录中的 Epoch

**文件**: `utils.py`  
**位置**: 第 371, 384 行

```python
axes[0].set_xlabel('Epoch')
axes[idx].set_xlabel('Epoch')
```

**说明**:
- 训练历史图表中的 X 轴标签
- 显示的是 epoch 编号（从 1 开始）

---

## 🎯 Epoch 编号说明

### Epoch 编号规则

1. **内部编号**（Python 代码中）：
   - 从 `0` 开始：`epoch = 0, 1, 2, ..., 99`
   - `range(start_epoch, args.epochs)` 使用内部编号

2. **显示编号**（给用户看的）：
   - 从 `1` 开始：`Epoch 1/100, Epoch 2/100, ..., Epoch 100/100`
   - `print(f"Epoch {epoch+1}/{args.epochs}")` 显示时 +1

3. **Checkpoint 文件名**：
   - 使用显示编号：`checkpoint_epoch_1.pth, checkpoint_epoch_2.pth, ...`
   - `f'checkpoint_epoch_{epoch+1}.pth'`

4. **Checkpoint 内容**：
   - 保存内部编号：`checkpoint['epoch'] = 99`（表示完成了第 100 个 epoch）

---

## 📝 修改 Epoch 数量的方法

### 方法1：命令行参数（推荐）

```bash
python main.py --image_path ./Data/images --gt_path ./Data/gt --epochs 200
```

### 方法2：修改默认值

**文件**: `main.py` 第 641 行

```python
# 修改前
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

# 修改后（例如改为 200）
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
```

### 方法3：修改批处理文件

**文件**: `run_train.bat` 第 10 行

```batch
# 修改前
--epochs 100 ^

# 修改后（例如改为 200）
--epochs 200 ^
```

### 方法4：PyCharm 运行配置

在 PyCharm 的 Parameters 中添加：
```
--epochs 200
```

---

## 🔍 关键代码位置总结

| 位置 | 文件 | 行号 | 作用 |
|------|------|------|------|
| **主要设置** | `main.py` | 641 | 命令行参数定义（默认 100） |
| **训练循环** | `main.py` | 500 | `for epoch in range(start_epoch, args.epochs)` |
| **Epoch 显示** | `main.py` | 501 | `print(f"Epoch {epoch+1}/{args.epochs}")` |
| **Checkpoint 保存** | `utils.py` | 216 | `'epoch': epoch` |
| **Checkpoint 恢复** | `main.py` | 471 | `start_epoch = checkpoint.get('epoch', 0) + 1` |
| **文件名生成** | `main.py` | 550 | `checkpoint_epoch_{epoch+1}.pth` |
| **批处理文件** | `run_train.bat` | 10 | `--epochs 100` |

---

## ⚠️ 重要注意事项

### 1. Epoch 编号的转换

- **内部编号**（代码中）：从 0 开始
- **显示编号**（用户看到）：从 1 开始
- **转换公式**：`显示编号 = 内部编号 + 1`

### 2. 恢复训练时的 Epoch 计算

```python
# Checkpoint 中保存的是已完成的 epoch（内部编号）
checkpoint['epoch'] = 99  # 表示完成了第 100 个 epoch

# 恢复时从下一个 epoch 开始
start_epoch = checkpoint['epoch'] + 1  # = 100（从第 101 个 epoch 开始）
```

### 3. 总 Epoch 数的含义

- `--epochs 100` 表示**总共训练 100 个 epoch**
- 如果从第 50 个 epoch 恢复，设置 `--epochs 100` 会继续训练到第 100 个 epoch
- 如果想再训练 50 个 epoch，应该设置 `--epochs 150`

---

## 💡 常见问题

### Q1: 如何只训练几个 epoch 进行测试？

```bash
python main.py --image_path ./Data/images --gt_path ./Data/gt --epochs 5
```

### Q2: 如何修改默认 epoch 数？

修改 `main.py` 第 641 行的 `default=100` 为您想要的值。

### Q3: 恢复训练时，epoch 数如何设置？

`--epochs` 参数是**总 epoch 数**，不是剩余 epoch 数。

- 如果从第 50 个 epoch 恢复，想训练到第 100 个 epoch：`--epochs 100`
- 如果想再训练 50 个 epoch：`--epochs 150`

---

## 📊 完整示例

### 示例1：新训练 200 个 epoch

```bash
python main.py --image_path ./Data/images --gt_path ./Data/gt --epochs 200
```

### 示例2：从第 50 个 epoch 恢复，训练到第 200 个 epoch

```bash
python main.py \
    --image_path ./Data/images \
    --gt_path ./Data/gt \
    --epochs 200 \
    --resume ./checkpoints/checkpoint_epoch_50.pth
```

### 示例3：快速测试（只训练 5 个 epoch）

```bash
python main.py --image_path ./Data/images --gt_path ./Data/gt --epochs 5
```

---

**总结：主要的 epoch 设置位置是 `main.py` 第 641 行的命令行参数 `--epochs`，默认值为 100。**
