"""Training of earthquake language model."""
import time
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from seisLM.model.pretrained_models import LitMultiDimWav2Vec2
from seisLM.data_pipeline import collator
from seisLM.utils import project_path
from seisLM.data_pipeline import dataloaders
from seisLM.configs.pretrain import pretrain_config


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def main():
  model_config = pretrain_config.get_model_config()
  training_config = pretrain_config.get_training_config()

  seed_everything(training_config.seed)
  model = LitMultiDimWav2Vec2(model_config, training_config)


  data_collator = \
    collator.DataCollatorForWav2Vec2PretrainingConcatChannelsNoPadding(
        model=model.model,
        mask_time_prob=training_config.mask_time_prob,
        mask_time_length=training_config.mask_time_length,
    )


  train_loader, dev_loader = dataloaders.prepare_seisbench_dataloaders(
    model=model,
    data_names=training_config.data_name,
    batch_size=training_config.global_batch_size,
    num_workers=training_config.num_workers,
    collator=data_collator,
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
  run_name = f"{training_config.seed}__{formatted_time}"

  logger = WandbLogger(
      project=training_config.logger_project_name,
      save_dir=project_path.MODEL_SAVE_DIR,
      name=run_name,
      id=run_name,
  )

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
    main()