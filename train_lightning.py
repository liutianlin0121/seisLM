"""Training of earthquake language model."""
import time
from ml_collections import config_dict
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from transformers import Wav2Vec2Config
from earthquakeLM.model import LitWav2Vec2
from earthquakeLM.utils import datadir
from earthquakeLM.data_pipeline import dataset as costa_rica_dataset


model_name_or_path = "patrickvonplaten/wav2vec2-base-v2"

model_config = Wav2Vec2Config.from_pretrained(model_name_or_path)
model_config.conv_dim = [a//8 for a in model_config.conv_dim]
model_config.conv_stride = [a * 2 for a in model_config.conv_stride]
model_config.conv_kernel = [a * 2 for a in model_config.conv_kernel]
model_config.num_attention_heads = 8
model_config.diversity_loss_weight = 0.15


training_config = config_dict.ConfigDict()
training_config.dataset_path = datadir('processed/pt')
training_config.mask_time_prob = 0.65
training_config.mask_time_length = 10
training_config.global_batch_size = 4
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
training_config.logger_project_name = 'earthquake-LM'
training_config.num_workers = 1
training_config.model_save_dir = \
  '/home/liu0003/Desktop/projects/earthquake-LM/saved_models'
training_config.num_train_fraction = 0.8
training_config.num_val_fraction = 0.1
training_config.num_test_fraction = 0.1
training_config.precision = "32"
training_config.data_normalization_type = 'root_mean_square'
training_config.gpu_devices = [0, 1]

seed_everything(training_config.seed)


# Prepare data collator and dataloader
dataset = costa_rica_dataset.EarthquakeRecordingDataset(
    training_config.dataset_path
)

model = LitWav2Vec2(model_config, training_config)


data_collator = \
  costa_rica_dataset.DataCollatorForWav2Vec2PretrainingConcatChannelsNoPadding(
      model=model.model,
      mask_time_prob=training_config.mask_time_prob,
      mask_time_length=training_config.mask_time_length,
      normalization_type=training_config.data_normalization_type,
  )

dataloaders = costa_rica_dataset.get_dataloaders_costa_rica(
    dataset,
    data_collator=data_collator,
    train_batch_size=training_config.global_batch_size,
    eval_batch_size=training_config.global_batch_size,
    num_train_fraction=training_config.num_train_fraction,
    num_val_fraction=training_config.num_val_fraction,
    num_test_fraction=training_config.num_test_fraction,
    num_workers=training_config.num_workers
)

training_config.max_train_steps = training_config.num_train_epochs * len(
  dataloaders['train'] )


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
    train_dataloaders=dataloaders['train'],
    val_dataloaders=dataloaders['val']
)

