#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1          # allocate one TITAN RTX
#SBATCH --time=15:00:00
#SBATCH --job-name=dummy_diff_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --chdir=/home/robine/dummy_diff

set -e
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"

# conda hook (portable)
eval "$($(which conda) shell.bash hook)"
conda activate my_env

# run *inside* an srun step that owns the GPU
srun --gpu-bind=single:1 python -u src/eval_mine.py
