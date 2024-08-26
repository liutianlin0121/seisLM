#!/bin/bash

JOB_NAME="pretrain"
# TEST_RUN=true
TEST_RUN=false
CONFIG_DIR="/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs"
SCRIPT_NAME="pretrain_slurm.sh"

# Print the value of test_run
echo "TEST_RUN is set to $TEST_RUN"

# Create the SLURM script
cat <<EOT > $SCRIPT_NAME
#!/bin/bash

#SBATCH --cpus-per-task=8        # Number of cores to reserve
#SBATCH --gres=gpu:4             # Number of GPUs to reserve
#SBATCH --job-name=${JOB_NAME}   # Name of your job
#SBATCH --mem-per-cpu=8G         # Amount of RAM/core to reserve
#SBATCH --nodes=1                # Node count
#SBATCH --ntasks-per-node=4      # Total number of tasks per node
#SBATCH --output=${JOB_NAME}.o%j # Path and name to the file for the STDOUT
#SBATCH --partition=a100         # Partition to allocate your job
EOT

if [ "$TEST_RUN" = true ]; then
  cat <<EOT >> $SCRIPT_NAME
#SBATCH --qos=gpu1day             # Selected queue to allocate your job
#SBATCH --time=0-24:00:00       # Maximum allocated time
#SBATCH --partition=a100         # Partition to allocate your job
#SBATCH --gres=gpu:2             # Number of GPUs to reserve
#SBATCH --ntasks-per-node=2      # Total number of tasks per node
EOT
else
  cat <<EOT >> $SCRIPT_NAME
#SBATCH --qos=gpu1week              # Selected queue to allocate your job
#SBATCH --time=7-00:00:00       # Maximum allocated time
EOT
fi

cat <<EOT >> $SCRIPT_NAME

source ~/miniconda3/etc/profile.d/conda.sh

conda activate /scicore/home/dokman0000/liu0003/miniconda3/envs/seisbench

srun python3 pretrain_run.py \\
  --config_path ${CONFIG_DIR}/${JOB_NAME}/pretrain_config_std_norm_two_axes.json \\
EOT

if [ "$TEST_RUN" = true ]; then
  echo "  --test_run" >> $SCRIPT_NAME
fi

# Submit the generated script
sbatch $SCRIPT_NAME
