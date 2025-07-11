#!/bin/bash
#SBATCH --job-name=fraud_detection_test
#SBATCH --time=10:00:00          # 预计运行时间1小时
#SBATCH --mem=32G                # 请求16GB内存
#SBATCH --cpus-per-task=32        # 请求8个CPU核心
#SBATCH --output=fraud_detection_%j.log  # 输出日志文件

# 加载必要的模块
module load miniforge

# 激活conda环境（如果有）
conda activate c7y_env

# 运行Python脚本
python fraud_detection_stacking.py
