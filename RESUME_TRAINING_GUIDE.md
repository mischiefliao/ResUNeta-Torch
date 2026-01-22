# 训练暂停和恢复功能使用指南

## 功能概述

现在项目支持以下功能：

1. ✅ **Ctrl+C 优雅暂停**：在训练过程中按 `Ctrl+C` 可以安全地暂停训练
2. ✅ **自动保存状态**：暂停时会自动保存当前训练状态到 checkpoint
3. ✅ **热启动恢复**：可以从保存的 checkpoint 继续训练，完全恢复训练状态

## 使用方法

### 1. 正常训练

```bash
python main.py \
    --image_path ./data/images \
    --gt_path ./data/gt \
    --image_size 256 \
    --batch_size 8 \
    --epochs 100 \
    --model_save_path ./checkpoints
```

### 2. 暂停训练（Ctrl+C）

在训练过程中，在 PyCharm 终端中按 `Ctrl+C`：

- 训练会立即停止当前 batch
- 自动保存当前状态到 `interrupted_checkpoint.pth`
- 显示恢复训练的命令

**示例输出：**
```
Training interrupted by user (Ctrl+C)
Saving current state...
Checkpoint saved to ./checkpoints/interrupted_checkpoint.pth
You can resume training using: --resume ./checkpoints/interrupted_checkpoint.pth
Training stopped.
```

### 3. 恢复训练

使用 `--resume` 参数从 checkpoint 恢复：

```bash
python main.py \
    --image_path ./data/images \
    --gt_path ./data/gt \
    --image_size 256 \
    --batch_size 8 \
    --epochs 100 \
    --model_save_path ./checkpoints \
    --resume ./checkpoints/interrupted_checkpoint.pth
```

**恢复时会：**
- ✅ 加载模型权重
- ✅ 恢复优化器状态（包括动量等）
- ✅ 恢复学习率调度器状态
- ✅ 恢复训练历史（loss、metrics）
- ✅ 从上次停止的 epoch 继续训练

## Checkpoint 文件说明

训练过程中会自动保存以下 checkpoint 文件：

1. **`latest_checkpoint.pth`** - 每个 epoch 后保存的最新状态（用于快速恢复）
2. **`checkpoint_epoch_N.pth`** - 每个 epoch 的完整 checkpoint（如果 `--checkpoint_mode epochs`）
3. **`best_model.pth`** - 验证集上表现最好的模型
4. **`interrupted_checkpoint.pth`** - Ctrl+C 暂停时保存的状态

## 恢复训练示例

### 场景1：从 Ctrl+C 暂停恢复

```bash
# 第一次训练（被 Ctrl+C 中断）
python main.py --image_path ./data/images --gt_path ./data/gt --epochs 100 --model_save_path ./checkpoints

# 按 Ctrl+C 后，恢复训练
python main.py --image_path ./data/images --gt_path ./data/gt --epochs 100 --model_save_path ./checkpoints --resume ./checkpoints/interrupted_checkpoint.pth
```

### 场景2：从特定 epoch 恢复

```bash
# 从第 50 个 epoch 的 checkpoint 恢复
python main.py --image_path ./data/images --gt_path ./data/gt --epochs 100 --model_save_path ./checkpoints --resume ./checkpoints/checkpoint_epoch_50.pth
```

### 场景3：从最新 checkpoint 恢复

```bash
# 从最新的 checkpoint 恢复
python main.py --image_path ./data/images --gt_path ./data/gt --epochs 100 --model_save_path ./checkpoints --resume ./checkpoints/latest_checkpoint.pth
```

## 注意事项

1. **参数一致性**：恢复训练时，建议使用相同的训练参数（batch_size、learning_rate 等），除非您明确想要修改它们

2. **Epoch 数量**：`--epochs` 参数是总 epoch 数。如果从第 50 个 epoch 恢复，设置 `--epochs 100` 会继续训练到第 100 个 epoch

3. **数据路径**：恢复训练时仍需要提供 `--image_path` 和 `--gt_path`，因为需要重新创建 DataLoader

4. **Checkpoint 兼容性**：确保 checkpoint 文件与当前模型架构兼容

## 保存的内容

每个 checkpoint 包含：

- ✅ 模型权重 (`model_state_dict`)
- ✅ 优化器状态 (`optimizer_state_dict`)
- ✅ 学习率调度器状态 (`scheduler_state_dict`)
- ✅ 训练历史：
  - `train_losses` - 训练损失历史
  - `val_losses` - 验证损失历史
  - `train_metrics_history` - 训练指标历史（IoU, Precision, Recall, F1）
  - `val_metrics_history` - 验证指标历史
- ✅ 当前 epoch 编号
- ✅ 最佳验证损失

## 故障排除

### 问题1：找不到 checkpoint 文件

**错误：** `FileNotFoundError: Checkpoint file not found: ...`

**解决：** 检查 checkpoint 路径是否正确，确保文件存在

### 问题2：模型架构不匹配

**错误：** `RuntimeError: Error(s) in loading state_dict`

**解决：** 确保 checkpoint 是用相同模型架构保存的

### 问题3：恢复后训练历史丢失

**解决：** 确保使用最新版本的代码，旧版本的 checkpoint 可能不包含训练历史

## 最佳实践

1. **定期检查 checkpoint**：确保 checkpoint 文件正常保存
2. **使用 `latest_checkpoint.pth`**：这是最方便的恢复点
3. **保留多个 checkpoint**：不要只依赖一个 checkpoint
4. **测试恢复功能**：在长时间训练前，先测试恢复功能是否正常

## 技术细节

- 使用 Python 的 `KeyboardInterrupt` 异常处理 Ctrl+C
- Checkpoint 使用 PyTorch 的 `torch.save()` 和 `torch.load()`
- 训练历史以列表形式保存，恢复时完全重建
- 支持在 Windows 和 Linux 上使用
