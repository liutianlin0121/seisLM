#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:2        # Number of GPUs to reserve
#SBATCH --job-name=shock-classifier     # Name of your job
#SBATCH --mem-per-cpu=4G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=2      # Total number of tasks per node
#SBATCH --output=shock.o%j   # Path and name to the file for the STDOUT
#SBATCH --partition=a100,rtx4090       # Partition to allocate your job
#SBATCH --qos=gpu6hours             # Selected queue to allocate your job
#SBATCH --time=0-6:00:00         # Maximum allocated time
#SBATCH --array=0-9             # Array job with indices


source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scicore/home/dokman0000/liu0003/miniconda3/envs/seisbench

# Define arrays for configs, training fractions, and num_classes
configs=(
  "/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/foreshock_aftershock/seisLM_base_shock_classifier.json"
  "/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/foreshock_aftershock/seisLM_large_shock_classifier.json"
  # "/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/foreshock_aftershock/conv1d_shock_classifier.json"
)

training_fractions=(0.05 0.1 0.2 0.5 1.0)
# training_fractions=(1.0)
num_classes=(9)

# Calculate the total number of combinations
num_training_fractions=${#training_fractions[@]}
num_class_configurations=${#num_classes[@]}
num_configs=${#configs[@]}

# Ensure the array indices are correct
total_combinations=$((num_configs * num_training_fractions * num_class_configurations))

# Calculate the indices for configs, training_fractions, and num_classes
config_index=$((SLURM_ARRAY_TASK_ID / (num_training_fractions * num_class_configurations)))
remainder=$((SLURM_ARRAY_TASK_ID % (num_training_fractions * num_class_configurations)))
training_fraction_index=$((remainder / num_class_configurations))
num_class_index=$((remainder % num_class_configurations))

# Get the actual config, training_fraction, and num_class for this job
config=${configs[$config_index]}
training_fraction=${training_fractions[$training_fraction_index]}
num_class=${num_classes[$num_class_index]}

# Run the Python script with the specific config, training_fraction, and num_class
srun python3 foreshock_aftershock_run.py \
  --config $config \
  --training_fraction $training_fraction \
  --num_classes $num_class \
  --save_checkpoints
