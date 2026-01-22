# 在PyCharm中安装CUDA版本的PyTorch

## 步骤 1: 检查CUDA版本

首先需要确定您的系统CUDA版本：

### 方法1: 使用命令行检查
在PowerShell或CMD中运行：
```powershell
nvidia-smi
```
查看右上角显示的CUDA版本（例如：CUDA Version: 12.1）

### 方法2: 检查已安装的CUDA
```powershell
nvcc --version
```

## 步骤 2: 在PyCharm中安装CUDA版本的PyTorch

### 方法A: 使用PyCharm的包管理器（推荐）

1. **打开PyCharm设置**
   - File → Settings（或按 `Ctrl+Alt+S`）
   - 或者：PyCharm → Preferences（Mac）

2. **导航到Python解释器**
   - Project → Python Interpreter
   - 选择您的虚拟环境（`.venv`）

3. **卸载CPU版本的PyTorch**
   - 在包列表中搜索 `torch`
   - 点击 `torch`、`torchvision`、`torchaudio` 右侧的 `-` 按钮卸载
   - 或者点击 `-` 按钮旁边的下拉菜单 → Uninstall

4. **安装CUDA版本的PyTorch**
   - 点击 `+` 按钮（Install Package）
   - 在搜索框中输入：`torch`
   - **不要直接安装**，需要指定CUDA版本

5. **使用命令行安装（在PyCharm终端中）**
   - View → Tool Windows → Terminal
   - 在终端中运行以下命令（根据您的CUDA版本选择）：

**CUDA 11.8:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 方法B: 直接使用PyCharm终端

1. **打开终端**
   - View → Tool Windows → Terminal
   - 确保激活了正确的虚拟环境（应该显示 `.venv`）

2. **卸载旧版本**
   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```

3. **安装CUDA版本**
   根据您的CUDA版本选择对应的命令（见上面的命令）

## 步骤 3: 验证安装

在PyCharm的Python Console中运行：

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

**期望输出**（如果安装成功）：
```
PyTorch版本: 2.x.x+cu118
CUDA可用: True
CUDA版本: 11.8
GPU设备: NVIDIA GeForce RTX ...
```

## 常见问题

### Q1: 如何知道应该安装哪个CUDA版本？
- 运行 `nvidia-smi` 查看CUDA Version
- 或者访问 https://pytorch.org/get-started/locally/ 查看兼容性

### Q2: 安装后仍然显示CPU？
- 检查是否在正确的虚拟环境中
- 重启PyCharm
- 确认PyTorch版本包含 `+cu` 后缀（例如：`2.1.0+cu118`）

### Q3: 找不到CUDA版本？
- 确保NVIDIA驱动已安装
- 检查CUDA Toolkit是否安装
- 某些GPU可能不支持较新的CUDA版本

### Q4: 安装失败？
- 检查网络连接
- 尝试使用国内镜像源：
  ```bash
  pip install torch torchvision torchaudio --index-url https://mirrors.aliyun.com/pypi/simple/
  ```
  然后手动安装CUDA版本

## 完整安装命令参考

### CUDA 11.8（最常用）
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 12.4
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 注意事项

1. **虚拟环境**: 确保在正确的虚拟环境中安装（`.venv`）
2. **版本匹配**: CUDA版本需要与NVIDIA驱动兼容
3. **重启**: 安装后建议重启PyCharm
4. **验证**: 安装后务必运行验证代码确认GPU可用
