"""Training script for the phase picking.

Adapted from:
  https://github.com/seisbench/pick-benchmark/blob/main/benchmark/train.py

Example usage:
python src/phasepick_run.py --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/ethz_seisLM.json

"""
import argparse
import traceback
import os
import json
import time
import ml_collections
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch import seed_everything
from seisLM.data_pipeline import seisbench_dataloaders as dataloaders
from seisLM.model.task_specific import phasepick_models
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread


def train_phasepick(
  config: ml_collections.ConfigDict,
  task_name: str,
  save_checkpoint: bool = False,
  ) -> None:
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

  model_name = config.model_name
  data_name = config.data_args.data_name
  training_fraction = config.data_args.get("training_fraction", 1.0)

  model = phasepick_models.__getattribute__(model_name + "Lit")(
      **config.get("model_args", {})
  )

  train_loader, dev_loader = dataloaders.prepare_seisbench_dataloaders(
      model=model,
      data_names=data_name,
      batch_size=config.data_args.batch_size,
      num_workers=config.get("num_workers", 8),
      training_fraction=training_fraction,
      cache=config.get("cache", None),
      prefetch_factor=config.get("prefetch_factor", 2),
  )



  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )

  run_name = data_name + f"_train_frac_{training_fraction}" \
        + f"_model_{model_name}" + f"_seed_{seed}" + f"_time_{formatted_time}"

  logger = WandbLogger(
      # Groups related experiments together
      project=task_name,
      # Describes a specific experiment within the project
      name=run_name,
      # Filter runs based on keywords or categories.
      tags=[
        f"data_{data_name}",
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

  if save_checkpoint:
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        save_last=True,
        mode='min',
        filename="{epoch}-{step}",
    )
    callbacks = [checkpoint_callback]
    enable_checkpointing = True
  else:
    callbacks = None
    enable_checkpointing = False
    print('Checkpoints will not be saved.')

  # TODO: Add the option to freeze transformer layers for several epochs.

  # Training loop
  trainer = L.Trainer(
      profiler="simple",
      default_root_dir=project_path.MODEL_SAVE_DIR,
      logger=logger,
      callbacks=callbacks,
      enable_checkpointing=enable_checkpointing,
      **config.get("trainer_args", {}),
  )

  trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  torch.set_float32_matmul_precision('high')

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument(
      "--save_checkpoints", action="store_true",
      help="Run in test mode for profiling purposes"
  )

  args = parser.parse_args()


  with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)

  config = ml_collections.ConfigDict(config)
  task_name = os.path.basename(__file__)[: -len(".py")]

  try:
    train_phasepick(config, task_name)
  except Exception as e:
    traceback.print_exc()
  finally:
    shutdown_cleanup_thread.start()
