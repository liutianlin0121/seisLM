import warnings
from ml_collections import ConfigDict
from transformers import Wav2Vec2Config

def merge_configs(config1: ConfigDict, config2: ConfigDict) -> ConfigDict:
  """Merges two ConfigDicts."""
  merged_config = ConfigDict()

  for key, value in config1.items():
    merged_config[key] = value

  for key, value in config2.items():
    if key in merged_config and merged_config[key] != value:
      warnings.warn(f'Conflict for key "{key}": '
                   f'{merged_config[key]} (config1) vs {value} (config2)')
    merged_config[key] = value

  return merged_config

def wav2vec2_config_to_configdict(
  wav2vec2_config: Wav2Vec2Config) -> ConfigDict:
  config_dict = ConfigDict()

  for key, value in wav2vec2_config.to_dict().items():
    print(key, value)
    config_dict[key] = value

  return config_dict

class ConfigTracker:
    def __init__(self, config):
        self.config = config
        self.accessed_keys = set()

    def __getattr__(self, key):
        if key in self.config:
            self.accessed_keys.add(key)
            return self.config[key]
        else:
            raise AttributeError(f"Config has no attribute '{key}'")

    def get_accessed_keys(self):
        return self.accessed_keys

    def get_unaccessed_keys(self):
        return set(self.config.keys()) - self.accessed_keys
