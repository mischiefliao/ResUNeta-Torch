# 安装依赖包指南

## 方法1: 使用PyCharm自动安装（最简单）

当您看到"不满足软件包要求"的警告时：

1. **点击"安装要求"按钮**
   - PyCharm会自动安装requirements.txt中的所有包
   - 这是最简单的方法

2. **或者手动操作**：
   - File → Settings → Project → Python Interpreter
   - 点击包列表上方的 `+` 按钮
   - 搜索并安装每个包

## 方法2: 使用终端安装（推荐）

在PyCharm的Terminal中运行：

```bash
cd "D:\Work\MajorJob\AI\Remote Sensing\resunetp-main\resunet-a-pytorch"
pip install -r requirements.txt
```

## 方法3: 逐个安装

如果批量安装失败，可以逐个安装：

```bash
pip install torch>=1.9.0
pip install torchvision>=0.10.0
pip install numpy>=1.19.0
pip install opencv-python>=4.5.0
pip install scikit-learn>=0.24.0
pip install tqdm>=4.62.0
pip install matplotlib>=3.3.0
pip install Pillow>=8.0.0
pip install albumentations>=1.0.0
```

## 注意事项

### 如果安装PyTorch时遇到问题

**CPU版本**（如果没有GPU）：
```bash
pip install torch torchvision torchaudio
```

**CUDA版本**（如果有GPU，根据CUDA版本选择）：

CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 如果scikit-learn安装失败

由于scipy DLL问题，scikit-learn可能无法安装。不用担心，代码已经使用numpy实现替代方案，可以正常运行。

## 验证安装

安装完成后，在Python Console中运行：

```python
import torch
import torchvision
import numpy
import cv2
import tqdm
import matplotlib
import albumentations

print("所有依赖已安装！")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

## 快速安装命令

**完整安装（推荐）**：
```bash
pip install torch torchvision torchaudio numpy opencv-python tqdm matplotlib Pillow albumentations
```

**最小安装（如果scikit-learn有问题）**：
```bash
pip install torch torchvision torchaudio numpy opencv-python tqdm matplotlib Pillow albumentations
# scikit-learn 可选，代码已使用numpy替代
```
