#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --account="hrfmri2"
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time="01:59:00"
#SBATCH --job-name=dataloadeg
#SBATCH --gres="gpu:1"
#SBATCH --partition="dc-gpu-devel"


#activate virtual environemnt
source ../../dataloadenv/bin/activate

# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${{MASTER_ADDR}}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=6000

# Run the program
time srun --cpu-bind=socket python imagenet1k_conversion.py \
    --path "." \
    --format "memmap" \
    --img_per_file 10000 \
    --batch_size 1000 \
    --num_worker 4
    

