#!/bin/bash
#SBATCH --job-name=fraud_detection_test
#SBATCH --time=01:00:00          # 预计运行时间1小时
#SBATCH --mem=16G                # 请求16GB内存
#SBATCH --cpus-per-task=8        # 请求8个CPU核心
#SBATCH --output=base_rf%j.log  # 输出日志文件

# 加载必要的模块
module load miniforge

# 激活conda环境（如果有）
conda activate c7y_env

# 运行Python脚本
python run_rf.py
