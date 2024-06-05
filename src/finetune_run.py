"""Training script for the phase picking.

Simplified from:
  https://github.com/seisbench/pick-benchmark/blob/main/benchmark/train.py
"""

import argparse
import json
import os
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from seisLM.data_pipeline import dataloaders
import seisLM.model.supervised_models as models
from seisLM.utils import project_path


def train(config, experiment_name):
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

  model = models.__getattribute__(config["model"] + "Lit")(
      **config.get("model_args", {})
  )

  train_loader, dev_loader = dataloaders.prepare_seisbench_dataloaders(
      model=model,
      data_names=config["data"],
      batch_size=config.get("batch_size", 1024),
      num_workers=config.get("num_workers", 8),
  )

  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )
  run_name = f"{seed}__{formatted_time}"

  project_path.create_folder_if_not_exists(project_path.MODEL_SAVE_DIR)

  logger = WandbLogger(
      project=experiment_name,
      save_dir=project_path.MODEL_SAVE_DIR,
      name=run_name,
      id=run_name,
  )


  logger.log_hyperparams(model.hparams)
  logger.log_hyperparams(config)

  checkpoint_callback = ModelCheckpoint(
      monitor="val_loss",
      save_top_k=1,
      mode='min',
      filename="{epoch}-{step}",
  )

  callbacks = [checkpoint_callback]

  # TODO: Freeze the transformer layers for the first several epochs
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

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  args = parser.parse_args()


  with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)

  experiment_name = os.path.basename(args.config)[:-5]
  train(config, experiment_name)
