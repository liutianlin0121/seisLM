"""Training script for the phase picking.

Adapted from:
  https://github.com/seisbench/pick-benchmark/blob/main/benchmark/train.py


"""
import argparse
import json
import os
import time
import traceback


import lightning as L
import torch
import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import ml_collections


from seisLM.data_pipeline import \
    foreshock_aftershock_dataloaders as dataloaders
from seisLM.model.task_specific import foreshock_aftershock_models
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread
from seisLM.model.task_specific import shared_task_specific

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

  model = foreshock_aftershock_models.ShockClassifierLit(
      model_name=config.model_name,
      model_config=config.model_args,
      training_config=config.trainer_args,
  )

  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )

  run_name = f"num_classes_{config.model_args.num_classes}_seed_{seed}"\
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
      save_dir=project_path.MODEL_SAVE_DIR,
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
  wandb.finish()

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
    # for num_classes in [4, 9, 8, 2]:
    for num_classes in [4]:
      config.model_args.num_classes = num_classes
      train_foreshock_aftershock(config, task_name, args.save_checkpoints)

  except Exception as e:
    traceback.print_exc()
  finally:
    shutdown_cleanup_thread.start()
