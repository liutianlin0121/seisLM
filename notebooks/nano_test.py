# %%
import numpy as np
import seisbench.data as sbd
import torch
from lightning.pytorch import seed_everything
from transformers import Wav2Vec2ForPreTraining as RefWav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices, _sample_negative_indices)

from seisLM.configs.pretrain import pretrain_config
from seisLM.model.nano_wav2vec2 import Wav2Vec2ForPreTraining

# %%
data = sbd.STEAD()
waveforms = data.get_waveforms(1265656)

input_values = torch.Tensor(waveforms[0]).unsqueeze(0)
batch_size, raw_sequence_length = input_values.shape
print('input_values.shape: ', input_values.shape)

# %%

config = pretrain_config.get_model_config()
ref_model = RefWav2Vec2ForPreTraining(config)
nano_model = Wav2Vec2ForPreTraining(config)
nano_model.load_state_dict(ref_model.state_dict())

model_dict = {'ref_model': ref_model, 'nano_model': nano_model}
output_dict = {}

for mode in ['train', 'eval']:
  for model_name, model in model_dict.items():
    if mode == 'train':
      model.train()
    else:
      model.eval()

    sequence_length = model._get_feat_extract_output_lengths(
      raw_sequence_length).item()

    seed_everything(0)
    mask_time_indices = _compute_mask_indices(
        shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
    )
    sampled_negative_indices = _sample_negative_indices(
        features_shape=(batch_size, sequence_length),
        num_negatives=model.config.num_negatives,
        mask_time_indices=mask_time_indices,
    )
    mask_time_indices = torch.tensor(
      data=mask_time_indices, device=input_values.device, dtype=torch.long)
    sampled_negative_indices = torch.tensor(
        data=sampled_negative_indices, device=input_values.device,
        dtype=torch.long
    )

    with torch.no_grad():
      outputs = model(input_values, mask_time_indices=mask_time_indices,
                      sampled_negative_indices=sampled_negative_indices)
    output_dict[model_name] = outputs


  for field in output_dict['ref_model']:
    value1 = getattr(output_dict['ref_model'], field)
    value2 = getattr(output_dict['nano_model'], field)
    print('passed: ', field)
    assert np.allclose(value1.cpu().numpy(), value2.cpu().numpy())

  print(f'mode_train={model.training}: all fields are equal!')
