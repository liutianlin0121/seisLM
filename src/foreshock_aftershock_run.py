"""Training script for the phase picking.

Adapted from:
  https://github.com/seisbench/pick-benchmark/blob/main/benchmark/train.py


"""
import argparse
import traceback
import os
import json
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch import seed_everything
from seisLM.data_pipeline import foreshock_aftershock_dataloaders as dataloaders
from seisLM.model.task_specific import foreshock_aftershock_models
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread


def train_foreshock_aftershock(config, task_name):
  """
  Runs the model training defined by the config.
  """
  seed = config.get("seed", 42)
  seed_everything(seed)

  model_name = config["model"]

  model = foreshock_aftershock_models.__getattribute__(model_name + "Lit")(
      **config["model_args"]
  )

  config['data_args']['num_classes'] = config["model_args"]['num_classes']
  loaders = dataloaders.prepare_foreshock_aftershock_dataloaders(
      **config["data_args"],
  )


  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )

  run_name = f"seed_{seed}" + f"_time_{formatted_time}"

  logger = WandbLogger(
      # Groups related experiments together
      project=task_name,
      # Describes a specific experiment within the project
      name=run_name,
      # Filter runs based on keywords or categories.
      tags=[f"model_{model_name}",],
      # A unique identifier for the run
      id=run_name,
      save_code=True,
      save_dir=project_path.MODEL_SAVE_DIR,
  )

  slurm_job_id = os.getenv('SLURM_JOB_ID')
  if slurm_job_id:
    logger.log_hyperparams({"slurm_job_id": slurm_job_id})

  logger.log_hyperparams(model.hparams)
  logger.log_hyperparams(config)

  checkpoint_callback = ModelCheckpoint(
      monitor="val/loss",
      save_top_k=1,
      mode='min',
      filename="{epoch}-{step}",
  )

  callbacks = [checkpoint_callback]

  # Training loop
  trainer = L.Trainer(
      profiler="simple",
      default_root_dir=project_path.MODEL_SAVE_DIR,
      logger=logger,
      callbacks=callbacks,
      **config.get("trainer_args", {}),
  )

  trainer.fit(model, loaders['train'], loaders['val'])


if __name__ == "__main__":
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  torch.set_float32_matmul_precision('high')

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  args = parser.parse_args()


  with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)

  task_name = os.path.basename(__file__)[: -len(".py")]

  try:
    train_foreshock_aftershock(config, task_name)
  except Exception as e:
    traceback.print_exc()
  finally:
    shutdown_cleanup_thread.start()
