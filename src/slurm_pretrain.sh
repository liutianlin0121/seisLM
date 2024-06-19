#!/bin/bash

#SBATCH --job-name=seisLM_pretrain     #Name of your job
#SBATCH --cpus-per-task=10    #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve
#SBATCH --time=7-00:00:00      #Maximum allocated time
#SBATCH --qos=gpu1week         #Selected queue to allocate your job
#SBATCH --output=seisLM_pretrain.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=seisLM_pretrain.e%j    #Path and name to the file for the STDERR
#SBATCH --gres=gpu:4         #Number of GPUs to reserve
#SBATCH --partition=a100

source ~/anaconda3/etc/profile.d/conda.sh

conda activate /scicore/home/dokman0000/liu0003/anaconda3/envs/seisbench

# sinkhorn
# srun python3 pretrain_run.py --model_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/model_config_4xdownsample_sinkhorn.json --training_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/training_config.json

# no sinkhorn
# srun python3 pretrain_run.py --model_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/model_config_4xdownsample.json --training_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/training_config.json

# no sinkhorn with scaled logits in quantization
srun python3 pretrain_run.py --model_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/model_config_4xdownsample_scale_logits_quantization.json --training_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/training_config.json
