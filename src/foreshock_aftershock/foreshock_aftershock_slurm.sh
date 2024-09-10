#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:2             # Number of GPUs to reserve
#SBATCH --job-name=foreshock_aftershock     # Name of your job
#SBATCH --mem-per-cpu=4G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=2      # Total number of tasks per node
#SBATCH --output=foreshock_aftershock.o%j   # Path and name to the file for the STDOUT
#SBATCH --partition=rtx4090         # Partition to allocate your job
#SBATCH --qos=gpu6hours             # Selected queue to allocate your job
#SBATCH --time=0-01:00:00        # Maximum allocated time

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scicore/home/dokman0000/liu0003/miniconda3/envs/seisbench

srun python3 foreshock_aftershock_run.py --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/foreshock_aftershock/seisLM_shock_classifier.json --num_classes 9
