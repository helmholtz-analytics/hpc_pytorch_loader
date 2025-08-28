#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --account="hrfmri2"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time="02:00:00"
#SBATCH --job-name=train_resnet18
#SBATCH --gres="gpu:4"
#SBATCH --partition="dc-gpu-devel"

# Activate virtual environment
source ../../dataloadenv/bin/activate

# Set up distributed training environment
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=6000

# Run the training script
srun --cpu-bind=socket python train_resnet18.py \
    --dataset_path "path/to/converted/dataset" \
    --output_model_path "resnet18_model.pth" \
    --batch_size 64 \
    --num_workers 4 \
    --epochs 20 \
    --lr 0.001 \
    --dist \