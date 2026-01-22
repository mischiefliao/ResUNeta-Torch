# Windows 11 + PyCharm 运行指南

## ✅ 兼容性说明

**这个项目完全可以在 Windows 11 上使用 PyCharm 运行！** 代码已经针对 Windows 进行了兼容性处理。

## 🔧 Windows 特定修改

### 1. 多进程数据加载

**问题**: Windows 上 PyTorch 的 `DataLoader` 使用多进程（`num_workers > 0`）时可能出现问题。

**解决方案**: 
- 代码已自动检测 Windows 系统，并将 `num_workers` 设置为 0
- 在 Windows 上使用单进程数据加载（速度稍慢但更稳定）

**修改位置**:
- `main.py`: 第 103-104 行自动设置 `num_workers=0`（Windows）
- `batch_preprocess.py`: 添加了 Windows 多进程兼容性处理

### 2. 多进程启动方法

**问题**: Windows 不支持 Unix 的 'fork' 方式，需要使用 'spawn'。

**解决方案**: 
- 代码已自动设置 `multiprocessing.set_start_method('spawn')`
- 在所有相关文件中添加了 Windows 检测和设置

**修改位置**:
- `main.py`: 第 15-22 行
- `batch_preprocess.py`: 第 15-20 行
- `predict.py`: 第 15-20 行

## 📋 PyCharm 配置步骤

### 步骤 1: 打开项目

1. 打开 PyCharm
2. File → Open → 选择 `resunet-a-pytorch` 目录
3. 等待 PyCharm 索引完成

### 步骤 2: 配置 Python 解释器

1. File → Settings → Project → Python Interpreter
2. 点击齿轮图标 → Add...
3. 选择：
   - **Existing environment**: 选择已安装 Python 3.7+ 的环境
   - **New environment**: 创建新的虚拟环境（推荐）
4. 确保 Python 版本 ≥ 3.7

### 步骤 3: 安装依赖

**方法1: 使用 PyCharm 终端**

1. View → Tool Windows → Terminal
2. 在终端中运行：
```bash
pip install -r requirements.txt
```

**方法2: 使用 PyCharm 包管理器**

1. File → Settings → Project → Python Interpreter
2. 点击 `+` 按钮
3. 搜索并安装：`torch`, `torchvision`, `opencv-python`, `albumentations` 等

### 步骤 4: 配置运行参数

#### 训练脚本配置

1. Run → Edit Configurations...
2. 点击 `+` → Python
3. 配置如下：

**Name**: `Train ResUNet-a`

**Script path**: `D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch\main.py`

**Parameters**:
```
--image_path ./data/images --gt_path ./data/gt --image_size 256 --batch_size 8 --epochs 100 --model_save_path ./checkpoints
```

**Working directory**: `D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch`

#### 预测脚本配置

1. Run → Edit Configurations...
2. 点击 `+` → Python
3. 配置如下：

**Name**: `Predict ResUNet-a`

**Script path**: `D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch\predict.py`

**Parameters**:
```
--image_path ./test_images --model_path ./checkpoints/best_model.pth --output_path ./results --image_size 256 --num_classes 2
```

**Working directory**: `D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch`

### 步骤 5: 准备数据目录

在项目根目录创建以下结构：

```
resunet-a-pytorch/
├── data/
│   ├── images/      # 训练图像
│   └── gt/          # 掩码
├── test_images/     # 测试图像
└── checkpoints/     # 模型保存（会自动创建）
```

### 步骤 6: 运行

1. 选择配置好的运行配置（Train ResUNet-a 或 Predict ResUNet-a）
2. 点击运行按钮（绿色三角形）或按 `Shift+F10`

## 🐛 常见 Windows 问题及解决方案

### 问题 1: "RuntimeError: An attempt has been made to start a new process..."

**原因**: Windows 多进程问题

**解决**: 
- ✅ 代码已自动处理，`num_workers` 在 Windows 上自动设为 0
- 如果仍有问题，在 `main.py` 中手动设置 `num_workers=0`

### 问题 2: "FileNotFoundError: [WinError 2] 系统找不到指定的文件"

**原因**: 路径问题

**解决**:
- 使用正斜杠 `/` 或双反斜杠 `\\` 在路径中
- 使用相对路径：`./data/images` 而不是 `D:\...`
- 确保路径中的目录存在

### 问题 3: CUDA/GPU 不可用

**原因**: Windows 上 CUDA 配置问题

**解决**:
1. 检查 CUDA 安装：
```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
```

2. 如果返回 False：
   - 安装 CUDA 版本的 PyTorch：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   - 检查 NVIDIA 驱动是否最新

### 问题 4: 内存不足

**解决**:
- 减小 `batch_size`（例如改为 4 或 2）
- 减小 `image_size`（例如改为 128）
- 关闭其他占用内存的程序

### 问题 5: 路径中包含中文或特殊字符

**解决**:
- 避免在路径中使用中文
- 使用英文路径：`D:\Projects\resunet\data`

## 💡 PyCharm 使用技巧

### 1. 调试模式

- 在代码行号左侧点击设置断点
- 使用 `Shift+F9` 启动调试
- 可以查看变量值、单步执行等

### 2. 查看变量

- 运行/调试时，在 Variables 窗口查看变量值
- 鼠标悬停在变量上查看值

### 3. 终端使用

- View → Tool Windows → Terminal
- 可以直接运行命令行命令
- 支持 PowerShell 和 CMD

### 4. 代码补全

- PyCharm 会自动提供代码补全
- `Ctrl+Space` 手动触发补全

## 📝 快速测试

运行测试脚本确保一切正常：

1. 创建运行配置：
   - **Script path**: `test_basic.py`
   - **Parameters**: (留空)
   - **Working directory**: `resunet-a-pytorch`

2. 运行测试，应该看到：
```
==================================================
Running basic tests for ResUNet-a PyTorch implementation
==================================================
Testing model...
Input shape: torch.Size([2, 3, 256, 256])
Output shape: torch.Size([2, 2, 256, 256])
✓ Model forward pass successful!
...
✓ All tests passed successfully!
```

## ✅ 检查清单

在运行前确认：

- [ ] Python 3.7+ 已安装
- [ ] PyCharm 已配置 Python 解释器
- [ ] 所有依赖已安装（`pip install -r requirements.txt`）
- [ ] 数据目录已创建（`data/images` 和 `data/gt`）
- [ ] 运行配置已设置
- [ ] 测试脚本运行成功

## 🚀 开始训练

一切就绪后：

1. 确保数据已准备好
2. 选择 "Train ResUNet-a" 配置
3. 点击运行
4. 观察控制台输出

训练过程中会显示：
- 每个 epoch 的训练和验证损失
- IoU、Precision、Recall、F1 指标
- 最佳模型自动保存

## 📞 获取帮助

如果遇到问题：

1. 检查控制台错误信息
2. 查看 `RUN_GUIDE.md` 获取详细说明
3. 运行 `test_basic.py` 验证环境配置
4. 检查数据路径和格式是否正确
