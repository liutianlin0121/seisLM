"""Training script for the phase picking.

Adapted from:
  https://github.com/seisbench/pick-benchmark/blob/main/benchmark/train.py


"""
import argparse
import os
import json
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch import seed_everything
from seisLM.data_pipeline import dataloaders
from seisLM.model.task_specific import phasepick_models
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread


def train_phasepick(config, task_name):
  """
  Runs the model training defined by the config.

  Config parameters:

      - model: Model used as in the models.py file, but without the Lit suffix
      - data: Dataset used, as in seisbench.data
      - model_args: Arguments passed to the constructor of the model lightning
           module
      - trainer_args: Arguments passed to the lightning trainer
      - batch_size: Batch size for training and validation
      - num_workers: Number of workers for data loading.
        If not set, uses environment variable BENCHMARK_DEFAULT_WORKERS

  :param config: Configuration parameters for training
  """
  seed = config.get("seed", 42)
  seed_everything(seed)

  model_name = config["model"]
  training_fraction = config.get("training_fraction", 1.0)

  model = phasepick_models.__getattribute__(model_name + "Lit")(
      **config.get("model_args", {})
  )

  train_loader, dev_loader = dataloaders.prepare_seisbench_dataloaders(
      model=model,
      data_names=config["data"],
      batch_size=config.get("batch_size", 1024),
      num_workers=config.get("num_workers", 8),
      training_fraction=training_fraction,
      cache=config.get("cache", None),
      prefetch_factor=config.get("prefetch_factor", 2),
  )



  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )

  run_name = config['data'] + f"_train_frac_{training_fraction}" \
        + f"_model_{model_name}" + f"_seed_{seed}" + f"_time_{formatted_time}"

  logger = WandbLogger(
      # Groups related experiments together
      project=task_name,
      # Describes a specific experiment within the project
      name=run_name,
      # Filter runs based on keywords or categories.
      tags=[
        f"data_{config['data']}",
        f"model_{model_name}",
        f"train_frac_{training_fraction}"
      ],
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
      monitor="val_loss",
      save_top_k=1,
      mode='min',
      filename="{epoch}-{step}",
  )

  callbacks = [checkpoint_callback]

  # TODO: Add the option to freeze transformer layers for several epochs.

  # Training loop
  trainer = L.Trainer(
      profiler="simple",
      default_root_dir=project_path.MODEL_SAVE_DIR,
      logger=logger,
      callbacks=callbacks,
      **config.get("trainer_args", {}),
  )

  trainer.fit(model, train_loader, dev_loader)


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
    train_phasepick(config, task_name)
  except:
    print("Something went wrong")
  finally:
    shutdown_cleanup_thread.start()

