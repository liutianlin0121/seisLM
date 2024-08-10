#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:2             # Number of GPUs to reserve
#SBATCH --job-name=phasepick     # Name of your job
#SBATCH --mem-per-cpu=4G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=2      # Total number of tasks per node
#SBATCH --output=phasepick.o%j   # Path and name to the file for the STDOUT
#SBATCH --partition=a100         # Partition to allocate your job
#SBATCH --qos=1day             # Selected queue to allocate your job
#SBATCH --time=1-00:00:00        # Maximum allocated time

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /scicore/home/dokman0000/liu0003/anaconda3/envs/seisbench

# seisLM with convolutional pos encoding
# srun python3 phasepick_run.py \
#   --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/seisLM_convpos.json \
#   --save_checkpoints

# seisLM with rope
# srun python3 phasepick_run.py \
#   --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/seisLM_rope.json \
#   --save_checkpoints


# phasenet
srun python3 phasepick_run.py --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/phasenet.json\
  --save_checkpoints
