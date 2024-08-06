"""Training of earthquake language model.

python src/pretrain_run.py \
  --config_path=/scicore/home/dokman0000/liu0003/projects/seisLM/seisLM/configs/pretrain/pretrain_config_rmsnorm_std_nomean_reduce_codevectors_rope.json \
  --test_run


"""
import argparse
import traceback
import os
import json
import time
import ml_collections
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2
from seisLM.data_pipeline import collator
from seisLM.data_pipeline import pretrain_dataloaders as dataloaders
from seisLM.utils import project_path
from seisLM.utils.wandb_utils import shutdown_cleanup_thread

DEFAULT_NUM_WORKERS = 4
def train_self_supervised(
  *,
  config: ml_collections.ConfigDict,
  project_name: str,
  run_name_prefix: str,
  ) -> None:
  """
  Args:
    model_config: Wav2Vec2Config object
    training_config: config_dict.ConfigDict object
    test_run: str
  """

  seed_everything(config.seed)
  model = LitMultiDimWav2Vec2(config)


  data_collator = \
    collator.DataCollatorForWav2Vec2PretrainingConcatChannelsNoPadding(
        config=config.model_config,
        mask_time_prob=config.training_config.mask_time_prob,
        mask_time_length=config.training_config.mask_time_length,
    )


  config.data_config.num_workers = int(
    os.environ.get('SLURM_CPUS_PER_TASK', DEFAULT_NUM_WORKERS))

  train_loader, dev_loaders = dataloaders.prepare_pretrain_dataloaders(
    model=model,
    training_fraction=config.data_config.training_fraction,
    data_names=config.data_config.data_name,
    batch_size=config.data_config.local_batch_size,
    num_workers=config.data_config.num_workers,
    prefetch_factor=config.data_config.prefetch_factor,
    collator=data_collator,
    cache=config.data_config.cache_dataset,
  )

  dev_loaders_iterable = list(dev_loaders.values())

  config.training_config.max_train_steps = (
    config.training_config.max_epochs * len(train_loader)
  )

  checkpoint_callback = ModelCheckpoint(
      monitor='train/loss',
      save_top_k=1,
      save_last=True,
      mode='min',
      filename="{epoch}-{step}",
  )

  lr_monitor = LearningRateMonitor(logging_interval='step')
  callbacks = [checkpoint_callback, lr_monitor]

  formatted_time = time.strftime(
    "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
  )


  logger = WandbLogger(
      project=project_name,
      save_dir=project_path.MODEL_SAVE_DIR,
      name=f"{run_name_prefix}_{config.seed}__{formatted_time}",
      id=f"{run_name_prefix}_{config.seed}__{formatted_time}",
      save_code=True,
  )

  slurm_job_id = os.getenv('SLURM_JOB_ID')

  if slurm_job_id:
    logger.log_hyperparams({"slurm_job_id": slurm_job_id})

  trainer = L.Trainer(
      profiler='simple',
      logger=logger,
      log_every_n_steps=config.training_config.log_every_n_steps,
      devices=config.training_config.devices,
      accelerator='gpu',
      strategy='ddp',
      detect_anomaly=config.training_config.detect_anomaly,
      max_epochs=config.training_config.max_epochs,
      callbacks=callbacks,
      default_root_dir=project_path.MODEL_SAVE_DIR,
      precision=config.training_config.precision,
  )

  # Start training
  trainer.fit(
      model,
      train_dataloaders=train_loader,
      val_dataloaders=dev_loaders_iterable,
  )

if __name__ == '__main__':
  # TODO: Try the following settings:
  # Enable flash attention
  torch.backends.cuda.enable_flash_sdp(True)
  # Set cuDNN backend flags
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = False
  torch.set_float32_matmul_precision('high')

  parser = argparse.ArgumentParser()
  parser.add_argument("--config_path", type=str, required=True)
  # Add the boolean argument with a default value of False
  parser.add_argument(
      "--test_run", action="store_true",
      help="Run in test mode for profiling purposes"
  )
  args = parser.parse_args()

  with open(args.config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
  config = ml_collections.ConfigDict(config)
  run_name_prefix = args.config_path.split("/")[-1].split(".")[0]

  if args.test_run:
    # if test_run is True, train for only 1 epoch w/ a small batchsize.
    print("Running in test mode")
    config.training_config.max_epochs = 2
    # config.data_config.local_batch_size = 4 #8
    config.data_config.data_name = ['ETHZ']
    config.training_config.detect_anomaly = True
    # config.training_config.devices=2
    project_name = "test_pretrained_seisLM"
  else:
    config.training_config.detect_anomaly = False
    project_name = "pretrained_seisLM"

  try:
    train_self_supervised(
      config=config,
      project_name=project_name,
      run_name_prefix=run_name_prefix
    )

  except Exception as e:
    traceback.print_exc()
  finally:
    shutdown_cleanup_thread.start()
