@echo off
REM 训练脚本 - 快速启动
REM 请根据您的实际数据路径修改下面的路径

python main.py ^
    --image_path ./data/images ^
    --gt_path ./data/gt ^
    --image_size 256 ^
    --batch_size 8 ^
    --epochs 100 ^
    --model_save_path ./checkpoints ^
    --learning_rate 1e-4 ^
    --loss_function tanimoto

pause
