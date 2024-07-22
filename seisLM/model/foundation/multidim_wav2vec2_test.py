"""
Unit tests for the multidim wav2vec model against the reference model.

TODO: Add tests and refactor code that correspond to padding and masking.
"""

import unittest
from typing import Any

import numpy as np
import torch
from lightning.pytorch import seed_everything
from transformers import Wav2Vec2Config
from transformers import Wav2Vec2ForPreTraining as RefWav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
  _compute_mask_indices, _sample_negative_indices)

from seisLM.model.foundation.multidim_wav2vec2 import \
  MultiDimWav2Vec2ForPreTraining

from seisLM.model.foundation.mask_utils import get_feat_extract_output_lengths


def compare_model_params(model: Any, ref_model: Any) -> bool:
  """Compare the parameters of the model and the reference model."""
  model_params = dict(model.named_parameters())
  ref_model_params = dict(ref_model.named_parameters())

  if model_params.keys() != ref_model_params.keys():
    return False

  for key in model_params:
    if not torch.equal(model_params[key], ref_model_params[key]):
      return False

  return True


class TestMultiDimWav2Vec2(unittest.TestCase):
  """Unit tests for the MultiDimWav2Vec2ForPreTraining class."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.input_values = torch.randn(1, 30000)

    cls.SEED = 42
    cls.MODEL_NAMES = [
      "patrickvonplaten/wav2vec2-base-v2",
      "facebook/wav2vec2-base"
    ]
    cls.num_negatives = 100

  def test_model_params_at_initialization(self) -> None:
    for model_name in self.MODEL_NAMES:
      config = Wav2Vec2Config.from_pretrained(model_name)
      seed_everything(self.SEED)
      model = RefWav2Vec2ForPreTraining(config)

      seed_everything(self.SEED)
      ref_model = RefWav2Vec2ForPreTraining(config)

      self.assertTrue(compare_model_params(model, ref_model))

  def test_model_outputs(self) -> None:
    for evaluate in [True, False]:
      model_output = {}
      for model_name in self.MODEL_NAMES:
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

          batch_size, raw_sequence_length = self.input_values.shape

          if model_type == 'ref':
            sequence_length = model._get_feat_extract_output_lengths(
              raw_sequence_length).item()  # pylint: disable=protected-access
          else:
            sequence_length = get_feat_extract_output_lengths(
              config, raw_sequence_length).item()

          seed_everything(0)
          mask_time_indices = _compute_mask_indices(
            shape=(batch_size, sequence_length),
            mask_prob=config.mask_time_prob,
            mask_length=config.mask_time_length,
          )
          sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, sequence_length),
            num_negatives=self.num_negatives,
            mask_time_indices=mask_time_indices,
          )
          mask_time_indices = torch.tensor(
            data=mask_time_indices,
            device=self.input_values.device,
            dtype=torch.long
          )
          sampled_negative_indices = torch.tensor(
            data=sampled_negative_indices,
            device=self.input_values.device,
            dtype=torch.long
          )

          with torch.no_grad():
            outputs = model(
              self.input_values,
              mask_time_indices=mask_time_indices,
              sampled_negative_indices=sampled_negative_indices
            )

          model_output[f'{model_name}_{model_type}'] = outputs

      for name in self.MODEL_NAMES:
        new_output = model_output[f'{name}_new']
        ref_output = model_output[f'{name}_ref']

        for field in ref_output:
          value1 = getattr(new_output, field)
          value2 = getattr(ref_output, field)
          self.assertTrue(
            np.allclose(value1.cpu().numpy(), value2.cpu().numpy()),
            f'Outputs for field {field} do not match:'
            f'new {field}: {value1}, ref {field}: {value2}'
          )


if __name__ == '__main__':
  unittest.main()
