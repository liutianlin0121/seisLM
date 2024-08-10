#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:2             # Number of GPUs to reserve
#SBATCH --job-name=phasepick     # Name of your job
#SBATCH --mem-per-cpu=4G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=2      # Total number of tasks per node
#SBATCH --output=phasepick.o%j   # Path and name to the file for the STDOUT
#SBATCH --partition=a100         # Partition to allocate your job
#SBATCH --qos=30min               # Selected queue to allocate your job
#SBATCH --time=0-00:30:00        # Maximum allocated time
#SBATCH --array=0-9              # Array job with indices


# TODO: REMEMBER TO EDIT THE ARRAY INDICES!!!
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /scicore/home/dokman0000/liu0003/anaconda3/envs/seisbench

# Define arrays for data names and training fractions
data_names=('GEOFON' 'ETHZ')
training_fractions=(0.05 0.1 0.2 0.5 1.0)

# Calculate the total number of combinations
num_data_names=${#data_names[@]}
num_training_fractions=${#training_fractions[@]}

# Calculate the indices for data_name and training_fraction
data_name_index=$((SLURM_ARRAY_TASK_ID / num_training_fractions))
training_fraction_index=$((SLURM_ARRAY_TASK_ID % num_training_fractions))

# Get the actual data_name and training_fraction for this job
data_name=${data_names[$data_name_index]}
training_fraction=${training_fractions[$training_fraction_index]}

# Run the Python script with the specific data_name and training_fraction
srun python3 phasepick_run.py \
  --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/phasenet.json \
  --data_name $data_name \
  --training_fraction $training_fraction \
  --save_checkpoints \
