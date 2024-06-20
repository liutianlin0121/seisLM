"""testing the multidim wav2vec model against the reference model"""
import torch
import numpy as np
from lightning.pytorch import seed_everything
import seisbench.data as sbd

from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers import Wav2Vec2Config
from transformers import Wav2Vec2ForPreTraining as RefWav2Vec2ForPreTraining
from seisLM.model.foundation.multidim_wav2vec2 import MultiDimWav2Vec2ForPreTraining

data = sbd.STEAD()
waveforms = data.get_waveforms(1265656)
input_values = torch.Tensor(waveforms[0]).unsqueeze(0)


MODEL_NAMES = ["patrickvonplaten/wav2vec2-base-v2", "facebook/wav2vec2-base"]

for evaluate in [True, False]:
  model_output = {}
  for model_name in MODEL_NAMES:
    config = Wav2Vec2Config.from_pretrained(model_name)

    for model_type in ['ref', 'new']:
      seed_everything(0)
      if model_type == 'ref':
        model = RefWav2Vec2ForPreTraining(config)
      else:
        ref_model = RefWav2Vec2ForPreTraining(config)
        model = MultiDimWav2Vec2ForPreTraining(config)
        model.load_state_dict(ref_model.state_dict())
        del ref_model

      if evaluate:
        model.eval()
      else:
        model.train()
      # compute masked indices
      batch_size, raw_sequence_length = input_values.shape
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

      model_output[f'{model_name}_{model_type}'] = outputs


  for name in MODEL_NAMES:
    new_output = model_output[f'{name}_new']
    ref_output = model_output[f'{name}_ref']

    for field in ref_output:
      value1 = getattr(new_output, field)
      value2 = getattr(ref_output, field)
      print('passed: ', field)
      assert np.allclose(value1.cpu().numpy(), value2.cpu().numpy())

    print(f'mode_train={model.training}: all fields are equal!')
