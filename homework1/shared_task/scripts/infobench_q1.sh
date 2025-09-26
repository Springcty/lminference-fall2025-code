#!/bin/sh 
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=general
#SBATCH --mem=48Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=infobench_q1
#SBATCH --error=logs/infobench_q1.%j.err
#SBATCH --output=logs/infobench_q1.%j.out

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
source /usr/share/Modules/init/bash
module load cuda-12.5

python3 ../shared_task_hw1.py