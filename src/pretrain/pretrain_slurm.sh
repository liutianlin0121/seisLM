#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:4             # Number of GPUs to reserve
#SBATCH --job-name=pretrain   # Name of your job
#SBATCH --mem-per-cpu=8G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=4      # Total number of tasks per node
#SBATCH --output=pretrain.o%j # Path and name to the file for the STDOUT
#SBATCH --partition=a100         # Partition to allocate your job
#SBATCH --qos=gpu1week              # Selected queue to allocate your job
#SBATCH --time=7-00:00:00       # Maximum allocated time

source ~/miniconda3/etc/profile.d/conda.sh

conda activate /scicore/home/dokman0000/liu0003/miniconda3/envs/seisbench

srun python3 pretrain_run.py \
  --config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/pretrain_config_std_norm_two_axes.json \
