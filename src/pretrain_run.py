"""Training of earthquake language model."""
import argparse
import traceback
import os
import json
import time
from ml_collections import config_dict
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from transformers import Wav2Vec2Config
from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2
from seisLM.data_pipeline import collator
from seisLM.data_pipeline import dataloaders
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread

DEFAULT_NUM_WORKERS = 4
def train_self_supervised(model_config, training_config, test_run):
  """
  Args:
    model_config: Wav2Vec2Config object
    training_config: config_dict.ConfigDict object
    test_run: bool
  """

  seed_everything(training_config.seed)
  model = LitMultiDimWav2Vec2(model_config, training_config)


  data_collator = \
    collator.DataCollatorForWav2Vec2PretrainingConcatChannelsNoPadding(
        model=model.model,
        mask_time_prob=training_config.mask_time_prob,
        mask_time_length=training_config.mask_time_length,
    )


  training_config.num_workers = int(
    os.environ.get('SLURM_CPUS_PER_TASK', DEFAULT_NUM_WORKERS))

  train_loader, dev_loader = dataloaders.prepare_seisbench_dataloaders(
    model=model,
    training_fraction=training_config.training_fraction,
    data_names=training_config.data_name,
    batch_size=training_config.local_batch_size,
    num_workers=training_config.num_workers,
    prefetch_factor=training_config.prefetch_factor,
    collator=data_collator,
    cache=training_config.cache_dataset,
  )

  training_config.max_train_steps = training_config.num_train_epochs * len(
    train_loader)

  checkpoint_callback = ModelCheckpoint(
      monitor='val/loss',
      save_top_k=1,
      mode='min',
      filename="{epoch}-{step}",
  )

  lr_monitor = LearningRateMonitor(logging_interval='step')
  callbacks = [checkpoint_callback, lr_monitor]

  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )

  project = 'pretrained_seisLM' if not test_run else 'test_pretrained_seisLM'

  logger = WandbLogger(
      project=project,
      save_dir=project_path.MODEL_SAVE_DIR,
      name=f"{training_config.seed}__{formatted_time}",
      id=f"{training_config.seed}__{formatted_time}",
      save_code=True,
  )

  slurm_job_id = os.getenv('SLURM_JOB_ID')

  if slurm_job_id:
    logger.log_hyperparams({"slurm_job_id": slurm_job_id})

  logger.log_hyperparams(model.hparams)
  logger.log_hyperparams(training_config.to_dict())

  trainer = L.Trainer(
      profiler='simple',
      logger=logger,
      log_every_n_steps=training_config.log_every_n_steps,
      devices=training_config.devices,
      accelerator='gpu',
      strategy='ddp',
      max_epochs=training_config.num_train_epochs,
      callbacks=callbacks,
      default_root_dir=project_path.MODEL_SAVE_DIR,
      precision=training_config.precision,
  )

  # Start training
  trainer.fit(
      model,
      train_dataloaders=train_loader,
      val_dataloaders=dev_loader,
  )

if __name__ == '__main__':
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  torch.set_float32_matmul_precision('high')

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_config_path", type=str, required=True)
  parser.add_argument("--training_config_path", type=str, required=True)
  # Add the boolean argument with a default value of False
  parser.add_argument(
      "--test_run", action="store_true",
      help="Run in test mode for profiling purposes"
  )
  args = parser.parse_args()

  model_config = Wav2Vec2Config.from_pretrained(
    args.model_config_path,
    attn_implementation="sdpa"
  )

  with open(args.training_config_path, "r", encoding="utf-8") as f:
    training_config = json.load(f)
  training_config = config_dict.ConfigDict(training_config)

  if args.test_run:
    training_config.num_train_epochs = 1

  try:
    train_self_supervised(model_config, training_config, args.test_run)
  except Exception as e:
    traceback.print_exc()
  finally:
    shutdown_cleanup_thread.start()
