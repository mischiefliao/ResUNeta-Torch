"""
Training script for ResUNet-a
PyTorch implementation converted from TensorFlow/Keras version
Maintains the same command-line interface as the original TensorFlow version
"""

# IMPORTANT: Set environment variable BEFORE importing torch/numpy
# Fix OpenMP library conflict (common in Anaconda environments)
# This is a workaround for the "libiomp5md.dll already initialized" error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Windows compatibility: Set multiprocessing start method
if sys.platform == 'win32':
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

from model import create_resunet_a
from loss import get_loss_function
from batch_preprocess import load_dataset
from utils import (
    calculate_metrics, save_checkpoint, load_checkpoint, count_parameters, 
    set_seed, plot_training_history
)
import glob
import shutil


def find_checkpoint_files(checkpoint_dir):
    """
    Find all checkpoint files in the directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
    
    Returns:
        List of checkpoint file paths
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoint_patterns = [
        '*.pth',
        '*.pt',
        'checkpoint*.pth',
        'best_model.pth',
        'latest_checkpoint.pth',
        'interrupted_checkpoint.pth'
    ]
    
    checkpoint_files = []
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))
    
    # Remove duplicates and sort
    checkpoint_files = sorted(list(set(checkpoint_files)))
    return checkpoint_files


def clear_checkpoints(checkpoint_dir):
    """
    Clear all checkpoint files in the directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Number of files deleted
    """
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if not checkpoint_files:
        print("No checkpoint files found to delete.")
        return 0
    
    print(f"\n找到 {len(checkpoint_files)} 个 checkpoint 文件:")
    for f in checkpoint_files:
        print(f"  - {os.path.basename(f)}")
    
    # Also remove training history image if exists
    history_file = os.path.join(checkpoint_dir, 'training_history.png')
    if os.path.exists(history_file) and history_file not in checkpoint_files:
        checkpoint_files.append(history_file)
        print(f"  - training_history.png")
    
    deleted_count = 0
    failed_files = []
    for file_path in checkpoint_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_count += 1
                print(f"  ✓ 已删除: {os.path.basename(file_path)}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                deleted_count += 1
                print(f"  ✓ 已删除目录: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ✗ 无法删除 {os.path.basename(file_path)}: {e}")
            failed_files.append(file_path)
    
    print(f"\n已删除 {deleted_count} 个文件/目录。")
    
    if failed_files:
        print(f"警告: {len(failed_files)} 个文件删除失败，请手动检查。")
        return deleted_count
    
    # Verify deletion
    remaining = find_checkpoint_files(checkpoint_dir)
    if remaining:
        print(f"警告: 仍有 {len(remaining)} 个 checkpoint 文件残留:")
        for f in remaining:
            print(f"  - {os.path.basename(f)}")
    else:
        print("✓ 所有 checkpoint 文件已成功清空。")
    
    return deleted_count


def interactive_resume_selection(checkpoint_dir):
    """
    Interactive function to let user choose between new training or resume
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Tuple (should_resume: bool, checkpoint_path: str or None)
    """
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if not checkpoint_files:
        print("\n未找到任何 checkpoint 文件，将开始新的训练。")
        return False, None
    
    print("\n" + "="*60)
    print("检测到已存在的训练 checkpoint 文件")
    print("="*60)
    print(f"\n找到 {len(checkpoint_files)} 个 checkpoint 文件:")
    
    # Group checkpoints by type
    latest_checkpoint = None
    best_checkpoint = None
    interrupted_checkpoint = None
    epoch_checkpoints = []
    
    for f in checkpoint_files:
        basename = os.path.basename(f)
        if basename == 'latest_checkpoint.pth':
            latest_checkpoint = f
        elif basename == 'best_model.pth':
            best_checkpoint = f
        elif basename == 'interrupted_checkpoint.pth':
            interrupted_checkpoint = f
        elif 'checkpoint_epoch_' in basename:
            epoch_checkpoints.append(f)
    
    # Display available checkpoints
    print("\n可用的 checkpoint:")
    checkpoint_options = []
    
    if latest_checkpoint:
        print(f"  [1] latest_checkpoint.pth (最新)")
        checkpoint_options.append(('latest', latest_checkpoint))
    
    if interrupted_checkpoint:
        print(f"  [2] interrupted_checkpoint.pth (中断保存)")
        checkpoint_options.append(('interrupted', interrupted_checkpoint))
    
    if best_checkpoint:
        print(f"  [3] best_model.pth (最佳模型)")
        checkpoint_options.append(('best', best_checkpoint))
    
    if epoch_checkpoints:
        # Show last few epoch checkpoints
        epoch_checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        print(f"  [4] checkpoint_epoch_*.pth (共 {len(epoch_checkpoints)} 个)")
        checkpoint_options.append(('epoch', epoch_checkpoints[-1]))  # Latest epoch
    
    print(f"\n  [N] 重新开始训练 (清空所有 checkpoint)")
    print(f"  [Q] 退出程序")
    
    while True:
        try:
            choice = input("\n请选择 [1-4/N/Q]: ").strip().upper()
            
            if choice == 'Q':
                print("退出程序。")
                sys.exit(0)
            
            elif choice == 'N':
                print("\n您选择了重新开始训练。")
                confirm = input("确认要清空所有 checkpoint 文件吗？(y/n): ").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    deleted_count = clear_checkpoints(checkpoint_dir)
                    if deleted_count > 0:
                        print(f"\n✓ 已清空 {deleted_count} 个 checkpoint 文件")
                    # Verify deletion one more time
                    remaining = find_checkpoint_files(checkpoint_dir)
                    if remaining:
                        print(f"⚠ 警告: 仍有 {len(remaining)} 个文件残留，但将继续新训练...")
                    print("\n✓ 将开始全新的训练（从 epoch 1 开始）")
                    return False, None
                else:
                    print("取消操作，重新选择...")
                    continue
            
            elif choice in ['1', '2', '3', '4']:
                idx = int(choice) - 1
                if idx < len(checkpoint_options):
                    checkpoint_type, checkpoint_path = checkpoint_options[idx]
                    print(f"\n您选择了: {os.path.basename(checkpoint_path)}")
                    if os.path.exists(checkpoint_path):
                        return True, checkpoint_path
                    else:
                        print(f"错误: checkpoint 文件不存在: {checkpoint_path}")
                        continue
                else:
                    print("无效的选择，请重新输入。")
                    continue
            
            else:
                print("无效的选择，请输入 1-4, N, 或 Q。")
                continue
                
        except KeyboardInterrupt:
            print("\n\n用户中断，退出程序。")
            sys.exit(0)
        except Exception as e:
            print(f"输入错误: {e}，请重新输入。")


def check_gradient_health(grad_norms, epoch):
    """
    Check for gradient explosion and vanishing gradients
    
    Args:
        grad_norms: List of gradient norms from an epoch
        epoch: Current epoch number
    
    Returns:
        Dictionary with gradient health status
    """
    if not grad_norms:
        return {
            'has_explosion': False,
            'has_vanishing': False,
            'status': 'unknown',
            'avg_norm': 0.0,
            'max_norm': 0.0,
            'min_norm': 0.0
        }
    
    avg_norm = sum(grad_norms) / len(grad_norms)
    max_norm = max(grad_norms)
    min_norm = min(grad_norms)
    
    # Gradient explosion threshold: norm > 10.0
    explosion_threshold = 10.0
    has_explosion = max_norm > explosion_threshold
    
    # Gradient vanishing threshold: norm < 1e-6
    vanishing_threshold = 1e-6
    has_vanishing = avg_norm < vanishing_threshold
    
    # Determine status
    if has_explosion:
        status = '⚠️ 梯度爆炸 (Gradient Explosion)'
    elif has_vanishing:
        status = '⚠️ 梯度消失 (Gradient Vanishing)'
    elif avg_norm < 1e-5:
        status = '⚠️ 梯度过小 (Very Small Gradients)'
    elif avg_norm > 1.0:
        status = '⚠️ 梯度较大 (Large Gradients)'
    else:
        status = '✓ 梯度正常 (Normal Gradients)'
    
    return {
        'has_explosion': has_explosion,
        'has_vanishing': has_vanishing,
        'status': status,
        'avg_norm': avg_norm,
        'max_norm': max_norm,
        'min_norm': min_norm,
        'grad_norms': grad_norms
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    num_processed_batches = 0  # Track actually processed batches
    
    # Gradient monitoring
    grad_norms = []  # Store gradient norms for analysis
    
    for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc="Training")):
        try:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            # Gradient clipping to prevent gradient explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Collect gradient norms for analysis
            grad_norms.append(grad_norm.item())
            
            # Monitor gradient (only print occasionally to avoid spam)
            if batch_idx == 0 and num_processed_batches == 0:
                print(f"  [Debug] Initial gradient norm: {grad_norm:.6f}, Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            total_loss += loss.item()
            num_processed_batches += 1
            
            # Calculate metrics
            with torch.no_grad():
                # Handle model output shape: (B, C, H, W) -> (B, H, W)
                if outputs.dim() == 4 and outputs.shape[1] == 2:
                    # Two channel output - take foreground channel
                    pred = torch.softmax(outputs, dim=1)[:, 1, :, :]  # (B, H, W)
                elif outputs.dim() == 4 and outputs.shape[1] == 1:
                    pred = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)
                else:
                    pred = torch.sigmoid(outputs)  # (B, H, W)
                pred = (pred > 0.5).float()
                # Ensure masks are (B, H, W) for metrics calculation
                masks_for_metrics = masks.squeeze(1) if masks.dim() == 4 and masks.shape[1] == 1 else masks
                batch_metrics = calculate_metrics(pred, masks_for_metrics)
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print("Skipping this batch and continuing...")
            continue
    
    # Average metrics (use actually processed batches)
    if num_processed_batches == 0:
        raise ValueError("No batches were successfully processed. Check your dataset and model.")
    avg_loss = total_loss / num_processed_batches
    avg_metrics = {key: value / num_processed_batches for key, value in total_metrics.items()}
    
    # Calculate gradient statistics
    if grad_norms:
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)
        min_grad_norm = min(grad_norms)
        avg_metrics['grad_norm'] = avg_grad_norm
        avg_metrics['max_grad_norm'] = max_grad_norm
        avg_metrics['min_grad_norm'] = min_grad_norm
        avg_metrics['grad_norms'] = grad_norms  # Store all norms for detailed analysis
    else:
        avg_metrics['grad_norm'] = 0.0
        avg_metrics['max_grad_norm'] = 0.0
        avg_metrics['min_grad_norm'] = 0.0
        avg_metrics['grad_norms'] = []
    
    return avg_loss, avg_metrics


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    num_processed_batches = 0  # Track actually processed batches
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Validating")):
            try:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at validation batch {batch_idx}, skipping...")
                    continue
                
                total_loss += loss.item()
                num_processed_batches += 1
                
                # Calculate metrics
                # Handle model output shape: (B, C, H, W) -> (B, H, W)
                if outputs.dim() == 4 and outputs.shape[1] == 2:
                    # Two channel output - take foreground channel
                    pred = torch.softmax(outputs, dim=1)[:, 1, :, :]  # (B, H, W)
                elif outputs.dim() == 4 and outputs.shape[1] == 1:
                    pred = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)
                else:
                    pred = torch.sigmoid(outputs)  # (B, H, W)
                pred = (pred > 0.5).float()
                # Ensure masks are (B, H, W) for metrics calculation
                masks_for_metrics = masks.squeeze(1) if masks.dim() == 4 and masks.shape[1] == 1 else masks
                batch_metrics = calculate_metrics(pred, masks_for_metrics)
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]
            except Exception as e:
                print(f"Error processing validation batch {batch_idx}: {e}")
                print("Skipping this batch and continuing...")
                continue
    
    # Average metrics (use actually processed batches)
    if num_processed_batches == 0:
        raise ValueError("No batches were successfully processed in validation. Check your dataset and model.")
    avg_loss = total_loss / num_processed_batches
    avg_metrics = {key: value / num_processed_batches for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def main(args):
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Interactive resume selection (if --resume is not explicitly provided and not in no_interactive mode)
    print(f"\n{'='*60}")
    print(f"检查训练模式...")
    print(f"{'='*60}")
    print(f"训练模式参数:")
    print(f"  args.resume = {args.resume}")
    print(f"  args.no_interactive = {args.no_interactive}")
    print(f"{'='*60}")
    
    if args.resume is None and not args.no_interactive:
        should_resume, checkpoint_path = interactive_resume_selection(args.model_save_path)
        print(f"\n交互式选择结果:")
        print(f"  should_resume = {should_resume}")
        print(f"  checkpoint_path = {checkpoint_path}")
        
        if should_resume and checkpoint_path:
            args.resume = checkpoint_path
            print(f"\n✓ 将使用 checkpoint: {os.path.basename(checkpoint_path)}")
        elif not should_resume:
            # User chose to start fresh - ensure args.resume is explicitly None
            args.resume = None
            print("\n✓ 用户选择了重新开始训练（已清空所有 checkpoint）...")
            # Double check: verify no checkpoint files remain
            remaining_checkpoints = find_checkpoint_files(args.model_save_path)
            if remaining_checkpoints:
                print(f"⚠ 警告: 仍有 {len(remaining_checkpoints)} 个 checkpoint 文件残留，将自动清理...")
                clear_checkpoints(args.model_save_path)
                # Final verification
                remaining_checkpoints = find_checkpoint_files(args.model_save_path)
                if remaining_checkpoints:
                    print(f"❌ 错误: 仍有 {len(remaining_checkpoints)} 个文件无法删除，请手动检查:")
                    for f in remaining_checkpoints:
                        print(f"   - {f}")
                    raise RuntimeError("无法清空所有 checkpoint 文件，请手动删除后重试。")
                else:
                    print("✓ 所有 checkpoint 文件已成功清空。")
            # CRITICAL: Force args.resume to None to prevent any accidental loading
            args.resume = None
            print(f"✓ args.resume 已强制设置为: {args.resume}")
            print(f"✓ 确认: 将不会加载任何 checkpoint")
    elif args.resume is not None:
        print(f"\n使用指定的 checkpoint: {args.resume}")
    else:
        print("\n非交互模式：开始新的训练...")
        # Ensure no checkpoint is loaded
        args.resume = None
        print(f"✓ args.resume 已明确设置为: {args.resume}")
    
    # Load data
    print("Loading dataset...")
    # Windows compatibility: Set num_workers to 0 on Windows to avoid multiprocessing issues
    num_workers = 0 if sys.platform == 'win32' else 4
    train_loader, val_loader = load_dataset(
        image_dir=args.image_path,
        gt_dir=args.gt_path,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=num_workers,
        use_augmentation=True
    )
    
    # Create model
    print("Creating model...")
    model = create_resunet_a(
        in_channels=3,
        out_channels=args.num_classes,
        depth=7,
        layer_norm=args.layer_norm
    )
    model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    criterion = get_loss_function(args.loss_function)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    # Note: verbose parameter removed in newer PyTorch versions
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Print training configuration
    print(f"\n{'='*60}")
    print(f"训练配置信息:")
    print(f"  学习率: {args.learning_rate}")
    print(f"  损失函数: {args.loss_function}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  图像尺寸: {args.image_size}")
    print(f"  优化器: Adam")
    print(f"  学习率调度器: ReduceLROnPlateau (patience=5, factor=0.5)")
    print(f"{'='*60}")
    
    # Warning if learning rate is too low
    if args.learning_rate < 5e-4:
        print(f"\n⚠ 警告: 当前学习率 {args.learning_rate} 可能偏小，建议尝试:")
        print(f"  --learning_rate 5e-4 或 --learning_rate 1e-3")
        print(f"  如果Loss下降很慢，可以尝试提高学习率\n")
    
    # Training history
    train_losses = []
    val_losses = []
    train_metrics_history = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
    val_metrics_history = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
    gradient_history = []  # Store gradient health information
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Load checkpoint if resuming (explicitly check for None to avoid any issues)
    print(f"\n{'='*60}")
    print(f"检查是否需要加载 checkpoint...")
    print(f"{'='*60}")
    print(f"Checkpoint 加载检查:")
    print(f"  args.resume = {args.resume}")
    print(f"  args.resume is not None = {args.resume is not None}")
    print(f"{'='*60}")
    
    # CRITICAL: Only load checkpoint if args.resume is explicitly set and not None/empty
    # Add extra safety check to prevent accidental checkpoint loading
    resume_path = args.resume
    if resume_path is not None:
        resume_path = str(resume_path).strip()
        if not resume_path:
            resume_path = None
    
    if resume_path is not None and resume_path:
        # Double check: ensure this is really what user wants
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"✓ 找到 checkpoint 文件，正在加载...")
        print(f"⚠ 警告: 将从 checkpoint 恢复训练，而不是从头开始！")
        checkpoint = load_checkpoint(model, optimizer, resume_path, device, scheduler)
        checkpoint_epoch = checkpoint.get('epoch', 0)
        start_epoch = checkpoint_epoch + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_metrics_history = checkpoint.get('train_metrics_history', {'iou': [], 'precision': [], 'recall': [], 'f1': []})
        val_metrics_history = checkpoint.get('val_metrics_history', {'iou': [], 'precision': [], 'recall': [], 'f1': []})
        gradient_history = checkpoint.get('gradient_history', [])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"✓ 从 epoch {start_epoch} 恢复训练（共 {args.epochs} 个 epoch）")
        print(f"⚠ 注意: Checkpoint 中保存的 epoch = {checkpoint_epoch} (已完成第 {checkpoint_epoch+1} 个 epoch)")
    else:
        print(f"✓ 未指定 checkpoint，将从头开始训练")
        # CRITICAL: Explicitly ensure start_epoch is 0 for new training
        # Force reset to prevent any accidental checkpoint loading
        start_epoch = 0
        # Reset all training history to ensure fresh start
        train_losses = []
        val_losses = []
        train_metrics_history = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        val_metrics_history = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        gradient_history = []
        best_val_loss = float('inf')
        print(f"✓ start_epoch 已强制设置为: {start_epoch}")
        print(f"✓ 所有训练历史已重置")
        print(f"✓ 确认: 不会加载任何 checkpoint，完全从头开始")
    
    # Training loop - 显示训练参数信息（所有训练方式都会显示）
    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"{'='*60}")
    print(f"训练参数信息:")
    print(f"  start_epoch = {start_epoch}")
    print(f"  总 epoch 数 = {args.epochs}")
    print(f"  训练范围: range({start_epoch}, {args.epochs}) = [{start_epoch}, {start_epoch+1}, ..., {args.epochs-1}]")
    print(f"  将显示: Epoch {start_epoch+1}/{args.epochs} 到 Epoch {args.epochs}/{args.epochs}")
    print(f"{'='*60}\n")
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            for key in train_metrics_history:
                if key in train_metrics:
                    train_metrics_history[key].append(train_metrics[key])
            
            # Collect gradient information
            grad_norms = train_metrics.get('grad_norms', [])
            if grad_norms:
                grad_health = check_gradient_health(grad_norms, epoch)
                gradient_history.append({
                    'epoch': epoch,
                    'avg_grad_norm': grad_health['avg_norm'],
                    'max_grad_norm': grad_health['max_norm'],
                    'min_grad_norm': grad_health['min_norm'],
                    'has_explosion': grad_health['has_explosion'],
                    'has_vanishing': grad_health['has_vanishing'],
                    'status': grad_health['status']
                })
            
            # Validate
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            for key in val_metrics_history:
                val_metrics_history[key].append(val_metrics[key])
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print metrics with higher precision (12 decimal places, 3x of current 4)
            print(f"Train Loss: {train_loss:.12f}")
            print(f"Val Loss: {val_loss:.12f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Train - IoU: {train_metrics['iou']:.4f}, Precision: {train_metrics['precision']:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val - IoU: {val_metrics['iou']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Display diagnostic information if metrics are zero (especially for first epoch or when issues occur)
            if (train_metrics.get('precision', 0) == 0.0 and train_metrics.get('recall', 0) == 0.0 and train_metrics.get('f1', 0) == 0.0) or \
               (val_metrics.get('precision', 0) == 0.0 and val_metrics.get('recall', 0) == 0.0 and val_metrics.get('f1', 0) == 0.0) or \
               (epoch == 0):  # Always show diagnostics for first epoch
                print(f"\n{'='*60}")
                print(f"指标诊断报告 (Epoch {epoch+1})")
                print(f"{'='*60}")
                
                if '_diagnostics' in train_metrics:
                    diag = train_metrics['_diagnostics']
                    print(f"【训练集诊断】")
                    print(f"  TP (真阳性): {diag['tp']:,} | FP (假阳性): {diag['fp']:,} | FN (假阴性): {diag['fn']:,} | TN (真阴性): {diag['tn']:,}")
                    print(f"  预测统计: 1的数量={diag['pred_ones']:,} ({diag['pred_ones_ratio']*100:.2f}%), 0的数量={diag['pred_zeros']:,} ({100-diag['pred_ones_ratio']*100:.2f}%)")
                    print(f"  真实标签: 1的数量={diag['target_ones']:,} ({diag['target_ones_ratio']*100:.2f}%), 0的数量={diag['target_zeros']:,} ({100-diag['target_ones_ratio']*100:.2f}%)")
                    
                    # 诊断问题
                    if diag['pred_ones'] == 0:
                        print(f"  ⚠️  问题: 模型预测全部为背景(0)！")
                        print(f"     可能原因: 模型输出全部<0.5，或模型未学习到任何特征")
                    elif diag['pred_zeros'] == 0:
                        print(f"  ⚠️  问题: 模型预测全部为前景(1)！")
                        print(f"     可能原因: 模型输出全部>0.5，或阈值设置不当")
                    if diag['target_ones'] == 0:
                        print(f"  ⚠️  警告: 真实标签全部为背景(0)！")
                        print(f"     可能原因: 数据标签格式问题，或数据集中没有前景像素")
                    elif diag['target_zeros'] == 0:
                        print(f"  ⚠️  警告: 真实标签全部为前景(1)！")
                        print(f"     可能原因: 数据标签格式问题")
                    if diag['tp'] == 0 and diag['fp'] == 0 and diag['fn'] > 0:
                        print(f"  ⚠️  问题: 模型预测全部为0，但真实标签有1！")
                    elif diag['tp'] == 0 and diag['fp'] > 0 and diag['fn'] == 0:
                        print(f"  ⚠️  问题: 模型预测全部为1，但真实标签全部为0！")
                
                if '_diagnostics' in val_metrics:
                    diag = val_metrics['_diagnostics']
                    print(f"\n【验证集诊断】")
                    print(f"  TP (真阳性): {diag['tp']:,} | FP (假阳性): {diag['fp']:,} | FN (假阴性): {diag['fn']:,} | TN (真阴性): {diag['tn']:,}")
                    print(f"  预测统计: 1的数量={diag['pred_ones']:,} ({diag['pred_ones_ratio']*100:.2f}%), 0的数量={diag['pred_zeros']:,} ({100-diag['pred_ones_ratio']*100:.2f}%)")
                    print(f"  真实标签: 1的数量={diag['target_ones']:,} ({diag['target_ones_ratio']*100:.2f}%), 0的数量={diag['target_zeros']:,} ({100-diag['target_ones_ratio']*100:.2f}%)")
                    
                    # 诊断问题
                    if diag['pred_ones'] == 0:
                        print(f"  ⚠️  问题: 模型预测全部为背景(0)！")
                        print(f"     可能原因: 模型输出全部<0.5，或模型未学习到任何特征")
                    elif diag['pred_zeros'] == 0:
                        print(f"  ⚠️  问题: 模型预测全部为前景(1)！")
                        print(f"     可能原因: 模型输出全部>0.5，或阈值设置不当")
                    if diag['target_ones'] == 0:
                        print(f"  ⚠️  警告: 真实标签全部为背景(0)！")
                        print(f"     可能原因: 数据标签格式问题，或数据集中没有前景像素")
                    elif diag['target_zeros'] == 0:
                        print(f"  ⚠️  警告: 真实标签全部为前景(1)！")
                        print(f"     可能原因: 数据标签格式问题")
                    if diag['tp'] == 0 and diag['fp'] == 0 and diag['fn'] > 0:
                        print(f"  ⚠️  问题: 模型预测全部为0，但真实标签有1！")
                    elif diag['tp'] == 0 and diag['fp'] > 0 and diag['fn'] == 0:
                        print(f"  ⚠️  问题: 模型预测全部为1，但真实标签全部为0！")
                
                print(f"{'='*60}\n")
            
            # Display gradient health report every 10 epochs
            if (epoch + 1) % 10 == 0 and grad_norms:
                print(f"\n{'='*60}")
                print(f"梯度健康报告 (Epoch {epoch+1})")
                print(f"{'='*60}")
                grad_health = check_gradient_health(grad_norms, epoch)
                print(f"状态: {grad_health['status']}")
                print(f"平均梯度范数: {grad_health['avg_norm']:.6f}")
                print(f"最大梯度范数: {grad_health['max_norm']:.6f}")
                print(f"最小梯度范数: {grad_health['min_norm']:.6f}")
                if grad_health['has_explosion']:
                    print(f"⚠️  检测到梯度爆炸！最大梯度范数: {grad_health['max_norm']:.6f} > 10.0")
                    print(f"   建议: 降低学习率或使用梯度裁剪")
                if grad_health['has_vanishing']:
                    print(f"⚠️  检测到梯度消失！平均梯度范数: {grad_health['avg_norm']:.6f} < 1e-6")
                    print(f"   建议: 提高学习率、使用残差连接或调整初始化")
                print(f"{'='*60}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.model_save_path, 'best_model.pth')
                try:
                    save_checkpoint(
                        model, optimizer, epoch, val_loss,
                        best_model_path,
                        scheduler=scheduler,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        train_metrics_history=train_metrics_history,
                        val_metrics_history=val_metrics_history,
                        gradient_history=gradient_history,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics
                    )
                    print(f"Best model saved (Val Loss: {val_loss:.4f})")
                except Exception as e:
                    print(f"Warning: Failed to save best model: {e}")
            
            # Save checkpoint based on mode
            if args.checkpoint_mode == 'epochs':
                checkpoint_path = os.path.join(args.model_save_path, f'checkpoint_epoch_{epoch+1}.pth')
                try:
                    save_checkpoint(
                        model, optimizer, epoch, val_loss,
                        checkpoint_path,
                        scheduler=scheduler,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        train_metrics_history=train_metrics_history,
                        val_metrics_history=val_metrics_history,
                        gradient_history=gradient_history,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics
                    )
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint: {e}")
            
            # Save latest checkpoint for resume capability
            latest_checkpoint_path = os.path.join(args.model_save_path, 'latest_checkpoint.pth')
            try:
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    latest_checkpoint_path,
                    scheduler=scheduler,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_metrics_history=train_metrics_history,
                    val_metrics_history=val_metrics_history,
                    gradient_history=gradient_history,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )
            except Exception as e:
                print(f"Warning: Failed to save latest checkpoint: {e}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Saving current state...")
        # Save checkpoint before exiting
        interrupt_checkpoint_path = os.path.join(args.model_save_path, 'interrupted_checkpoint.pth')
        try:
            # Get the last epoch and loss values if available
            last_epoch = start_epoch - 1 if epoch == start_epoch else epoch
            last_val_loss = val_losses[-1] if val_losses else best_val_loss
            last_train_loss = train_losses[-1] if train_losses else 0.0
            last_train_metrics = {k: v[-1] if v else 0.0 for k, v in train_metrics_history.items()} if train_metrics_history else {}
            last_val_metrics = {k: v[-1] if v else 0.0 for k, v in val_metrics_history.items()} if val_metrics_history else {}
            
            save_checkpoint(
                model, optimizer, last_epoch, last_val_loss,
                interrupt_checkpoint_path,
                scheduler=scheduler,
                train_losses=train_losses,
                val_losses=val_losses,
                train_metrics_history=train_metrics_history,
                val_metrics_history=val_metrics_history,
                gradient_history=gradient_history,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                train_metrics=last_train_metrics,
                val_metrics=last_val_metrics
            )
            print(f"Checkpoint saved to {interrupt_checkpoint_path}")
            print(f"You can resume training using: --resume {interrupt_checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to save interrupted checkpoint: {e}")
        print("Training stopped.")
        return
    
    # Save training history plot
    history_path = os.path.join(args.model_save_path, 'training_history.png')
    plot_training_history(
        train_losses, val_losses,
        train_metrics_history, val_metrics_history,
        save_path=history_path
    )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResUNet-a model")
    
    # Arguments matching the original TensorFlow version
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--image_path', type=str, required=True, help='Path to images directory')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth masks directory')
    parser.add_argument('--layer_norm', type=str, default='batch', 
                       choices=['batch', 'instance', 'layer'], help='Layer normalization type')
    parser.add_argument('--model_save_path', type=str, default='./', help='Path to save model')
    parser.add_argument('--checkpoint_mode', type=str, default='epochs',
                       choices=['epochs', 'best'], help='Checkpoint saving mode')
    
    # Additional PyTorch-specific arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--loss_function', type=str, default='tanimoto',
                       choices=['bce', 'dice', 'tanimoto'], help='Loss function')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from. If not provided, will prompt interactively.')
    parser.add_argument('--no_interactive', action='store_true',
                       help='Disable interactive checkpoint selection (use --resume or start fresh)')
    
    args = parser.parse_args()
    
    # Override interactive mode if --no_interactive is set
    if args.no_interactive and args.resume is None:
        print("非交互模式：未指定 --resume，将开始新的训练。")
    
    main(args)
