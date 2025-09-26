#!/bin/sh 
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=general
#SBATCH --mem=48Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=mmlu_med_bad_case
#SBATCH --error=logs/mmlu_med_bad_case.%j.err
#SBATCH --output=logs/mmlu_med_bad_case.%j.out

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
source /usr/share/Modules/init/bash
module load cuda-12.5

python3 ../shared_task_hw1.py