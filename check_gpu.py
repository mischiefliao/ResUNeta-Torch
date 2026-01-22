"""
检查GPU/CUDA是否可用
"""
import torch

print("=" * 50)
print("PyTorch GPU/CUDA 检查")
print("=" * 50)

print(f"\nPyTorch 版本: {torch.__version__}")

print(f"\nCUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU 可用！")
    print(f"\nGPU 设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print(f"\n当前使用的GPU: cuda:{torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU计算
    print("\n测试GPU计算...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU计算测试成功！")
else:
    print("✗ GPU 不可用")
    print("\n可能的原因：")
    print("1. 没有安装CUDA版本的PyTorch")
    print("2. 没有NVIDIA GPU或GPU驱动未安装")
    print("3. CUDA版本不匹配")
    print("\n解决方案：")
    print("1. 检查是否有NVIDIA GPU:")
    print("   - 打开设备管理器 -> 显示适配器")
    print("   - 查看是否有NVIDIA显卡")
    print("\n2. 安装CUDA版本的PyTorch:")
    print("   访问: https://pytorch.org/get-started/locally/")
    print("   选择您的CUDA版本，例如:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n3. 检查CUDA版本:")
    print("   nvidia-smi  # 在命令行运行")

print("\n" + "=" * 50)
