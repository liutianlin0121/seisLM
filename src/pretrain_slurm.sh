#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:2             # Number of GPUs to reserve
#SBATCH --job-name=pretrain     # Name of your job
#SBATCH --mem-per-cpu=4G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=2      # Total number of tasks per node
#SBATCH --output=pretrain.o%j   # Path and name to the file for the STDOUT
#SBATCH --partition=rtx8000,a100         # Partition to allocate your job
#SBATCH --qos=1day             # Selected queue to allocate your job
#SBATCH --time=1-00:00:00        # Maximum allocated time

source ~/anaconda3/etc/profile.d/conda.sh

conda activate /scicore/home/dokman0000/liu0003/anaconda3/envs/seisbench

# sinkhorn
# srun python3 pretrain_run.py --model_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/model_config_4xdownsample_sinkhorn.json --training_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/training_config.json

# no sinkhorn
# srun python3 pretrain_run.py --model_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/model_config_4xdownsample.json --training_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/training_config.json

# no sinkhorn with scaled logits in quantization
# srun python3 pretrain_run.py --model_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/model_config_4xdownsample_scale_logits_quantization.json --training_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/training_config.json

# 3 encoder layers with scaled logits in quantization
srun python3 pretrain_run.py --model_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/model_config_3encoder_layers_scale_logits_quantization.json --training_config_path /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/training_config.json
