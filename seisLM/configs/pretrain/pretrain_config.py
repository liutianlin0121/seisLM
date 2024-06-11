"""config for pretraining"""
from ml_collections import config_dict
from seisLM.utils import project_path
from transformers import Wav2Vec2Config

def get_training_config():
    config = config_dict.ConfigDict()
    config.data_name = ['ETHZ', 'GEOFON', 'STEAD', 'NEIC']
    config.mask_time_prob = 0.65
    config.mask_time_length = 10
    config.global_batch_size = 4 #16
    config.seed = 42
    config.warmup_frac_step = 0.2
    config.learning_rate = 1e-4
    config.weight_decay = 1e-4
    config.num_train_epochs = 20
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_epsilon = 1e-8
    config.max_gumbel_temperature = 2.0
    config.min_gumbel_temperature = 0.5
    config.log_every_n_steps = 100
    config.logger_project_name = "pretrained_seisLM"
    config.num_workers = 8
    config.model_save_dir = project_path.MODEL_SAVE_DIR
    config.precision = "32"
    config.devices = 4
    
    project_path.create_folder_if_not_exists(config.model_save_dir)
    return config

def get_model_config():
    model_config_path = project_path.gitdir(
    ) + '/seisLM/configs/pretrain/pretrained_model_config.json'
    model_config = Wav2Vec2Config.from_pretrained(model_config_path)
    return model_config
