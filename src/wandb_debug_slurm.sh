#!/bin/bash

#SBATCH --job-name=wandb_debug     #Name of your job
#SBATCH --cpus-per-task=4    #Number of cores to reserve
#SBATCH --mem=1G
#SBATCH --time=0-00:30:00      #Maximum allocated time
#SBATCH --qos=30min         #Selected queue to allocate your job
#SBATCH --output=wandb_debug.o%j   #Path and name to the file for the STDOUT
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --partition=scicore

source ~/anaconda3/etc/profile.d/conda.sh

conda activate /scicore/home/dokman0000/liu0003/anaconda3/envs/seisbench


srun python3 wandb_debug_run.py
