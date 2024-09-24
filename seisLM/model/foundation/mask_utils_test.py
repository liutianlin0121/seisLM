"""Unit tests for mask_utils.py"""

import unittest

import ml_collections
import numpy as np
import torch
from lightning.pytorch import seed_everything
from transformers.models.wav2vec2.modeling_wav2vec2 import (
  _compute_mask_indices,
  _sample_negative_indices,
)

from seisLM.model.foundation import mask_utils


class TestMaskUtils(unittest.TestCase):
  def setUp(self) -> None:
    self.batch_size = 8
    self.sequence_length = 128
    self.mask_prob = 0.2
    self.mask_length = 10
    self.num_negatives = 100

  def test_compute_mask_indices(self) -> None:
    """Test compute_mask_indices function against reference implementation"""
    seed_everything(0)
    ref_mask_time_indices = _compute_mask_indices(
      shape=(self.batch_size, self.sequence_length),
      mask_prob=self.mask_prob,
      mask_length=self.mask_length,
    )

    seed_everything(0)
    test_mask_time_indices = mask_utils.compute_mask_indices(
      shape=(self.batch_size, self.sequence_length),
      mask_prob=self.mask_prob,
      mask_length=self.mask_length,
    )

    np.testing.assert_array_equal(ref_mask_time_indices, test_mask_time_indices)

  def test_sample_negative_indices(self) -> None:
    """Test sample_negative_indices function against reference implementation"""
    seed_everything(0)
    mask_time_indices = _compute_mask_indices(
      shape=(self.batch_size, self.sequence_length),
      mask_prob=self.mask_prob,
      mask_length=self.mask_length,
    )

    seed_everything(0)
    ref_sampled_negative_indices = _sample_negative_indices(
      features_shape=(self.batch_size, self.sequence_length),
      num_negatives=self.num_negatives,
      mask_time_indices=mask_time_indices,
    )

    seed_everything(0)
    test_sampled_negative_indices = mask_utils.sample_negative_indices(
      features_shape=(self.batch_size, self.sequence_length),
      num_negatives=self.num_negatives,
      mask_time_indices=mask_time_indices,
    )

    np.testing.assert_array_equal(
      ref_sampled_negative_indices, test_sampled_negative_indices
    )

  def test_get_feat_extract_output_lengths(self) -> None:
    """Test get_feat_extract_output_lengths function"""
    config = ml_collections.ConfigDict()
    config.conv_kernel = [10, 3, 3]
    config.conv_stride = [5, 2, 2]

    expected_output_length = 5
    computed_output_length = mask_utils.get_feat_extract_output_lengths(
      config, self.sequence_length
    )

    self.assertEqual(computed_output_length, expected_output_length)

  def test_get_feature_vector_attention_mask(self) -> None:
    config = ml_collections.ConfigDict()
    config.conv_kernel = [8]
    config.conv_stride = [8]
    sequence_length = 24
    batch_size = 2

    attention_mask = torch.ones(batch_size, sequence_length)
    attention_mask[0, 16:] = 0
    attention_mask[1, 8:] = 0
    attention_mask = attention_mask.bool()

    reduced_attention_mask = mask_utils.get_feature_vector_attention_mask(
      config,
      feature_vector_length=3,
      attention_mask=attention_mask,
    )

    assert torch.equal(
      reduced_attention_mask,
      torch.tensor([[True, True, False], [True, False, False]]),
    )


if __name__ == "__main__":
  unittest.main()
