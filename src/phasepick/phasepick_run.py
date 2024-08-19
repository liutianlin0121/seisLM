"""Training script for the phase picking.

Adapted from:
  https://github.com/seisbench/pick-benchmark/blob/main/benchmark/train.py

Example usage:
python src/phasepick_run.py --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/phasepick/ethz_seisLM.json
python phasepick_run.py --config /home/liu0003/Desktop/projects/seisLM/seisLM/configs/phasepick/ethz_phasenet.json --save_checkpoints

"""
import logging
import argparse
import json
import os
import time
import traceback

import lightning as L
import ml_collections
import torch

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
  LearningRateMonitor, ModelCheckpoint
)
from lightning.pytorch.loggers import WandbLogger

from seisLM.data_pipeline import seisbench_dataloaders as dataloaders
from seisLM.model.task_specific import phasepick_models
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread
from seisLM.evaluation import pick_eval


def train_phasepick(
  config: ml_collections.ConfigDict,
  task_name: str,
  save_checkpoints: bool = False,
  run_name_prefix: str = "",
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

  if hasattr(config.model_args, "layerdrop"):
    config.training_args.strategy = "ddp_find_unused_parameters_true"


  model_name = config.model_name
  data_name = config.data_args.data_name
  training_fraction = config.data_args.get("training_fraction", 1.0)

  model_cls = phasepick_models.__getattribute__(model_name + "Lit")
  model = model_cls(
    config.model_args, config.training_args
  )

  train_loader, dev_loader = dataloaders.prepare_seisbench_dataloaders(
      model=model,
      data_names=data_name,
      batch_size=config.data_args.batch_size,
      num_workers=config.data_args.get("num_workers", 8),
      training_fraction=training_fraction,
      cache=config.get("cache", None),
      prefetch_factor=config.get("prefetch_factor", 2),
  )

  max_train_steps = config.training_args.max_epochs * len(train_loader)
  config.training_args.max_train_steps = max_train_steps


  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )

  run_name = data_name + f"_train_frac_{training_fraction}" \
        + f"_model_{model_name}" + f"_seed_{seed}" + f"_time_{formatted_time}"

  logger = WandbLogger(
      # Groups related experiments together
      project=task_name,
      # Describes a specific experiment within the project
      name=f"{run_name_prefix}_{run_name}",
      # Filter runs based on keywords or categories.
      tags=[
        f"data_{data_name}_train_frac_{training_fraction}",
        f"model_{model_name}",
      ],
      # A unique identifier for the run
      id=f"{run_name_prefix}_{run_name}",
      save_code=True,
      save_dir=project_path.MODEL_SAVE_DIR,
  )

  slurm_job_id = os.getenv('SLURM_JOB_ID')
  if slurm_job_id:
    logger.log_hyperparams({"slurm_job_id": slurm_job_id})

  logger.log_hyperparams(config.to_dict())
  logger.log_hyperparams(model.model_config.to_dict())

  lr_monitor = LearningRateMonitor(logging_interval='step')
  callbacks = [lr_monitor]

  if save_checkpoints:
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        save_last=True,
        mode='min',
        filename="{epoch}-{step}",
    )
    callbacks.append(checkpoint_callback)
    enable_checkpointing = True
  else:
    enable_checkpointing = False
    logging.warning("Checkpoints will not be saved.")


  log_every_n_steps = max(
      1,
      min(50, int(len(train_loader) / config.training_args.devices))
  )

  # Training loop
  trainer = L.Trainer(
      profiler="simple",
      log_every_n_steps=log_every_n_steps,
      check_val_every_n_epoch=5,
      default_root_dir=project_path.MODEL_SAVE_DIR,
      logger=logger,
      callbacks=callbacks,
      enable_checkpointing=enable_checkpointing,
      devices=config.training_args.devices,
      strategy=config.training_args.strategy,
      accelerator=config.training_args.accelerator,
      max_epochs=config.training_args.max_epochs
  )

  trainer.fit(model, train_loader, dev_loader)

  if trainer.checkpoint_callback is not None:
    best_model_path = trainer.checkpoint_callback.best_model_path
    if model_name == 'MultiDimWav2Vec2ForFrameClassification':
      model = model_cls.load_from_checkpoint(
        best_model_path,
        load_pretrained=False
      )
    else:
      model = model_cls.load_from_checkpoint(
        best_model_path
      )

    target_path = project_path.gitdir() +\
      f'/data/targets/{config.data_args.data_name}/'

    pick_eval.save_pick_predictions(
        model=model,
        target_path=target_path,
        sets=config.eval_args.sets,
        batch_size=config.eval_args.batch_size,
        save_tag=f"{run_name_prefix}_{run_name}",
    )


if __name__ == "__main__":
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  torch.set_float32_matmul_precision('high')

  parser = argparse.ArgumentParser()
  parser.add_argument("--config_path", type=str, required=True)

  parser.add_argument(
      "--training_fraction", type=float, required=True,
      help="Fraction of the training set to use"
  )
  parser.add_argument(
      "--save_checkpoints", action="store_true",
      help="Run in test mode for profiling purposes"
  )

  args = parser.parse_args()


  with open(args.config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

  config = ml_collections.ConfigDict(config)
  task_name = os.path.basename(__file__)[: -len(".py")]
  run_name_prefix = args.config_path.split("/")[-1].split(".")[0]


  target_path = project_path.gitdir() +\
    f'/data/targets/{config.data_args.data_name}/'

  assert os.path.exists(target_path), (
    f"Target path {target_path} does not exist."
    )


  config.data_args.training_fraction = args.training_fraction


  slurm_job_id = os.getenv('SLURM_JOB_ID')

  if config.training_args.devices == 1:
    config.training_args.strategy = 'auto' # disable ddp for single GPU

  try:
    train_phasepick(
      config=config,
      task_name=task_name,
      save_checkpoints=args.save_checkpoints,
      run_name_prefix=run_name_prefix,
    )


  except Exception as e:
    traceback.print_exc()
  finally:
    shutdown_cleanup_thread.start()
