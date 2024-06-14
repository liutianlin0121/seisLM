#!/bin/bash     

#SBATCH --job-name=seisLM_finetune     #Name of your job
#SBATCH --cpus-per-task=8    #Number of cores to reserve
#SBATCH --mem-per-cpu=4G     #Amount of RAM/core to reserve
#SBATCH --time=4-00:00:00      #Maximum allocated time
#SBATCH --qos=1week         #Selected queue to allocate your job
#SBATCH --output=seisLM_finetune.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=seisLM_finetune.e%j    #Path and name to the file for the STDERR
#SBATCH --gres=gpu:2         #Number of GPUs to reserve
#SBATCH --partition=a100

source ~/anaconda3/etc/profile.d/conda.sh

conda activate /scicore/home/dokman0000/liu0003/anaconda3/envs/seisbench 

python3 finetune_run.py --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/finetune/ethz_seisLM.json

