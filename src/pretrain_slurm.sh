#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:4             # Number of GPUs to reserve
#SBATCH --job-name=pretrain   # Name of your job
#SBATCH --mem-per-cpu=8G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=4      # Total number of tasks per node
#SBATCH --output=pretrain.o%j # Path and name to the file for the STDOUT
#SBATCH --partition=a100         # Partition to allocate your job
#SBATCH --qos=1week              # Selected queue to allocate your job
#SBATCH --time=7-00:00:00       # Maximum allocated time

source ~/anaconda3/etc/profile.d/conda.sh

conda activate /scicore/home/dokman0000/liu0003/anaconda3/envs/seisbench

# no sinkhorn with scaled logits in quantization
srun python3 pretrain_run.py \
  --config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/pretrain_config_rmsnorm_std_nomean_reduce_codevectors.json \
