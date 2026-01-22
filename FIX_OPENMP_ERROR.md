# OpenMP 库冲突错误解决方案

## 错误信息

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

## 错误原因

这个错误通常发生在 Anaconda 环境中，因为：
1. **多个OpenMP运行时库**：Anaconda、PyTorch、NumPy等可能都包含OpenMP库
2. **库冲突**：多个库尝试初始化同一个OpenMP运行时
3. **常见于Windows + Anaconda环境**

## 解决方案

### ✅ 方案1: 代码中自动修复（已实现）

我已经在所有相关文件中添加了以下代码：

```python
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

**修改的文件**：
- `main.py` - 训练脚本
- `predict.py` - 预测脚本
- `test_basic.py` - 测试脚本

现在直接运行应该不会再出现这个错误。

### 方案2: 设置系统环境变量（永久解决）

1. **Windows设置**：
   - 右键"此电脑" → 属性
   - 高级系统设置 → 环境变量
   - 在"用户变量"或"系统变量"中点击"新建"
   - 变量名：`KMP_DUPLICATE_LIB_OK`
   - 变量值：`TRUE`
   - 确定保存

2. **重启PyCharm**使环境变量生效

### 方案3: 在PyCharm运行配置中设置

1. Run → Edit Configurations...
2. 选择您的运行配置
3. 在"Environment variables"中添加：
   - Name: `KMP_DUPLICATE_LIB_OK`
   - Value: `TRUE`
4. 应用并运行

### 方案4: 在命令行中设置（临时）

在运行Python脚本前设置环境变量：

**PowerShell:**
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main.py --image_path ./data/images --gt_path ./data/gt
```

**CMD:**
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
python main.py --image_path ./data/images --gt_path ./data/gt
```

## 更好的长期解决方案

### 方案A: 重新安装相关包

```bash
# 在Anaconda环境中
conda install nomkl numpy scipy scikit-learn numexpr
conda remove mkl mkl-service
conda install mkl mkl-service
```

### 方案B: 使用conda-forge

```bash
conda install -c conda-forge numpy scipy scikit-learn
```

### 方案C: 创建新的干净环境

```bash
conda create -n pytorch_env python=3.10
conda activate pytorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python albumentations tqdm matplotlib scikit-learn
```

## 注意事项

⚠️ **警告**：设置 `KMP_DUPLICATE_LIB_OK=TRUE` 是一个**临时解决方案**，虽然通常可以解决问题，但：
- 可能导致轻微的性能下降
- 在极少数情况下可能产生不正确的结果
- 最佳实践是确保只有一个OpenMP运行时

✅ **推荐**：代码中已自动设置，这是最简单且安全的方案。

## 验证修复

运行训练脚本，如果不再出现OpenMP错误，说明修复成功：

```bash
python main.py --image_path ./data/images --gt_path ./data/gt
```

应该看到正常的训练输出，而不是OpenMP错误。
