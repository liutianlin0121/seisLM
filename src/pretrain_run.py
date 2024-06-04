"""Training of earthquake language model."""
import time
from ml_collections import config_dict
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from transformers import Wav2Vec2Config
from seisLM.model.pretrained_models import LitMultiDimWav2Vec2
from seisLM.data_pipeline import collator
from seisLM.utils.project_path import gitdir
from seisLM.data_pipeline import dataloaders

# model_name_or_path = "patrickvonplaten/wav2vec2-base-v2"

model_config_path = gitdir() + '/configs/pretrain/pretrained_model_config.json'
model_config = Wav2Vec2Config.from_pretrained(model_config_path)

training_config = config_dict.ConfigDict()
training_config.data_name = 'ETHZ'
training_config.mask_time_prob = 0.65
training_config.mask_time_length = 10
training_config.global_batch_size = 8
training_config.seed = 42
training_config.warmup_frac_step = 0.2
training_config.learning_rate = 1e-4
training_config.weight_decay = 1e-4
training_config.num_train_epochs = 20
training_config.adam_beta1 = 0.9
training_config.adam_beta2 = 0.999
training_config.adam_epsilon = 1e-8
training_config.max_gumbel_temperature = 2.0
training_config.min_gumbel_temperature = 0.5
training_config.log_every_n_steps = 100
training_config.logger_project_name = 'seisLM'
training_config.num_workers = 8
training_config.model_save_dir = \
  '/home/liu0003/Desktop/projects/seisLM/results/models'
training_config.precision = "32"
training_config.gpu_devices = [0, 1]
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
  data_name=training_config.data_name,
  batch_size=training_config.global_batch_size,
  num_workers=training_config.num_workers,
  collator=data_collator,
)

training_config.max_train_steps = training_config.num_train_epochs * len(
  train_loader)

# Training loop
checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',
    save_top_k=1,
    mode='min',
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval='step')

formatted_time = time.strftime(
  "%Y-%m-%d-%Hh-%Mm-%Ss", time.localtime(time.time())
)
run_name = f"{training_config.seed}__{formatted_time}"

logger = WandbLogger(
    project=training_config.logger_project_name,
    save_dir=training_config.model_save_dir,
    name=run_name,
    id=run_name,
)


logger.log_hyperparams(model.hparams)
logger.log_hyperparams(training_config.to_dict())


trainer = L.Trainer(
    profiler='simple',
    logger=logger,
    log_every_n_steps=training_config.log_every_n_steps,
    devices=training_config.gpu_devices,
    accelerator='gpu',
    strategy='ddp',
    max_epochs=training_config.num_train_epochs,
    callbacks=[
      checkpoint_callback, lr_monitor,
    ],
    default_root_dir=training_config.model_save_dir,
    precision=training_config.precision,
)


trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=dev_loader,
)

