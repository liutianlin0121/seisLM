"""Data collator for Wav2Vec2ForPreTraining model."""
from typing import Optional, Dict
from dataclasses import dataclass
import ml_collections
import torch
import numpy as np
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from seisLM.model.foundation import mask_utils


@dataclass
class DataCollatorForWav2Vec2PretrainingConcatChannelsNoPadding:
  """
  Data collator that prepare masked indices for self-supervised pretraining.

  Args:
    config: config dict
    mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
        Percentage (between 0 and 1) of all feature vectors along the time axis
        which will be masked for the contrastive task.
        Note that overlap between masked sequences may decrease the actual
        percentage of masked vectors.
        The default value is taken from the original wav2vec 2.0 article
        (https://arxiv.org/abs/2006.11477),
        and results in about 49 percent of each sequence being masked on
        average.
    mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
        Length of each vector mask span to mask along the time axis in the
        contrastive task. The default value
        originates from the original wav2vec 2.0 article and corresponds
        to the ``M`` variable mentioned there.
  """

  config: ml_collections.ConfigDict
  mask_time_prob: Optional[float] = 0.65
  mask_time_length: Optional[int] = 10

  def __call__(self, sample: Dict[str, np.ndarray]
               ) -> Dict[str, torch.Tensor]:

    features = np.stack([s['X'] for s in sample])
    features = torch.from_numpy(features)

    batch_size, _, seq_length = features.shape

    batch = {}

    batch["input_values"] = features
    device = batch["input_values"].device

    # computes the output length of the convoluional layers
    mask_indices_seq_length = mask_utils.get_feat_extract_output_lengths(
      self.config,
      seq_length
    )

    mask_indices_seq_length = int(mask_indices_seq_length)

    features_shape = (batch_size, mask_indices_seq_length)

    # sample randomly masked indices
    mask_time_indices = _compute_mask_indices(
        shape=features_shape,
        mask_prob=self.mask_time_prob,
        mask_length=self.mask_time_length,
    )

    # sample negative indices
    sampled_negative_indices = _sample_negative_indices(
        features_shape,
        self.config.num_negatives,
        mask_time_indices=mask_time_indices,
    )
    batch["attention_mask"] = torch.ones(
      features_shape,
      device=device
    )
    batch["mask_time_indices"] = torch.tensor(
      mask_time_indices,
      dtype=torch.long,
      device=device
    )
    batch["sampled_negative_indices"] = torch.tensor(
      sampled_negative_indices,
      dtype=torch.long,
      device=device
    )

    return batch
