"""Training script for the foreshock-aftershock classification.

Example usage:

python foreshock_aftershock_run.py --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/foreshock_aftershock/seisLM_shock_classifier.json

python foreshock_aftershock_run.py --config /scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/foreshock_aftershock/conv1d_shock_classifier.json

"""
import argparse
import json
import os
import time
import traceback


import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (LearningRateMonitor, ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
import ml_collections


from seisLM.data_pipeline import \
    foreshock_aftershock_dataloaders as dataloaders
from seisLM.model.task_specific import foreshock_aftershock_models
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread
from seisLM.model.task_specific import shared_task_specific


# The ratio 0.7 here is the base fraction of training dataset
# (out of the whole dataset that contains training, validation,
# and testing sets).
BASE_TRAINING_FRACTION = 0.7

def train_foreshock_aftershock(
  config: ml_collections.ConfigDict,
  task_name: str,
  save_checkpoint: bool = False,
  ) -> None:
  """Runs the model training defined by the config.
  """
  seed = config.get("seed", 42)
  seed_everything(seed)


  loaders = dataloaders.prepare_foreshock_aftershock_dataloaders(
      num_classes=config.model_args.num_classes,
      **config.data_args,
  )

  max_train_steps = config.trainer_args.max_epochs * len(
    loaders['train'])

  config.trainer_args.max_train_steps = max_train_steps


  if config.model_name == 'Wav2Vec2ForSequenceClassification':
    LitModel = foreshock_aftershock_models.Wav2vec2ShockClassifierLit
  elif config.model_name == 'Conv1DShockClassifier':
    LitModel = foreshock_aftershock_models.Conv1DShockClassifierLit
  else:
    raise ValueError(f"Model {config.model_name} not supported")

  model = LitModel(
      model_config=config.model_args,
      training_config=config.trainer_args,
  )

  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )

  relative_fraction = round(
    config.data_args.train_frac / BASE_TRAINING_FRACTION, 3
  )

  run_name = f"num_classes_{config.model_args.num_classes}_seed_{seed}"\
    + f"_training_fraction_{relative_fraction}"\
    + f"_model_{config.model_name}"\
    + f"_time_{formatted_time}"

  logger = WandbLogger(
      # Groups related experiments together
      project=task_name,
      # Describes a specific experiment within the project
      name=run_name,
      # Filter runs based on keywords or categories.
      tags=[f"model_{config.model_name}",
            f"num_classes_{config.model_args.num_classes}"],
      # A unique identifier for the run
      id=run_name,
      save_code=True,
      offline=config.get("wandb_offline", True),
      save_dir=project_path.MODEL_SAVE_DIR,
      config=config,
  )

  slurm_job_id = os.getenv('SLURM_JOB_ID')
  if slurm_job_id:
    logger.log_hyperparams({"slurm_job_id": slurm_job_id})

  logger.log_hyperparams(config.to_dict())
  logger.log_hyperparams(model.model_config.to_dict())

  lr_monitor = LearningRateMonitor(logging_interval='step')
  callbacks = [lr_monitor]

  if save_checkpoint:
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
    print('Checkpoints will not be saved.')


  if (config.model_name == "Wav2Vec2ForSequenceClassification" and
    config.trainer_args.unfreeze_base_at_epoch > 0):
    callbacks.append(
      shared_task_specific.BaseModelUnfreeze(
        unfreeze_at_epoch=config.trainer_args.unfreeze_base_at_epoch
      )
    )

  log_every_n_steps = min(
    50, len(loaders['train']) // config.trainer_args.devices
  )

  # Training loop
  trainer = L.Trainer(
      profiler="simple",
      default_root_dir=project_path.MODEL_SAVE_DIR,
      logger=logger,
      callbacks=callbacks,
      log_every_n_steps=log_every_n_steps,
      devices=config.trainer_args.devices,
      strategy=config.trainer_args.strategy,
      accelerator=config.trainer_args.accelerator,
      max_epochs=config.trainer_args.max_epochs,
      enable_checkpointing=enable_checkpointing,
  )

  trainer.fit(model, loaders['train'], loaders['test'])
  # trainer.test(ckpt_path="best", dataloaders=loaders['test'])
  trainer.test(ckpt_path="last", dataloaders=loaders['test'])

if __name__ == "__main__":
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  torch.set_float32_matmul_precision('high')

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument(
    "--training_fraction", type=float, default=1.0, required=False,
    help="Fraction of the training set to use (default: 1.0)"
  )
  parser.add_argument(
    "--num_classes", type=int, choices=[2, 4, 8, 9],
    default=4, required=False,
    help="Number of classes in the dataset (2, 4, 8, or 9). (default: 4)"
  )

  parser.add_argument(
      "--save_checkpoints", action="store_true",
      help="Run in test mode for profiling purposes"
  )

  args = parser.parse_args()

  with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)
  config = ml_collections.ConfigDict(config)

  if hasattr(config.model_args, "layerdrop") and (
    config.model_args.layerdrop > 0):
    config.trainer_args.strategy = "ddp_find_unused_parameters_true"




  config.data_args.train_frac = args.training_fraction * BASE_TRAINING_FRACTION

  config.model_args.num_classes = args.num_classes
  task_name = os.path.basename(__file__)[: -len(".py")]

  try:
    train_foreshock_aftershock(config, task_name, args.save_checkpoints)

  except Exception as e:
    traceback.print_exc()
  finally:
    shutdown_cleanup_thread.start()
