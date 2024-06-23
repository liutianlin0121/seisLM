import os
import random
import time
import wandb
from seisLM.utils.wandb_utils import shutdown_cleanup_thread

# Initialize a new WandB run
wandb.init(project="debugging_project")

slurm_job_id = os.getenv('SLURM_JOB_ID')

# Log SLURM job ID
if slurm_job_id:
  wandb.config.update({"slurm_job_id": slurm_job_id})

# Log some random metrics
for i in range(10):
  wandb.log({
      "metric_1": random.random(),
      "metric_2": random.uniform(0, 100),
      "metric_3": random.randint(0, 10)
  })
  time.sleep(1)  # Simulate some delay


shutdown_cleanup_thread.start()
