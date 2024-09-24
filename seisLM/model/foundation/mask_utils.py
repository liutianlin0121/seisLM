"""Masking utilities."""

from typing import Optional, Tuple, Union

import ml_collections
import numpy as np
import torch


def get_feat_extract_output_lengths(
  config: ml_collections.ConfigDict,
  input_lengths: Union[torch.Tensor, int],
) -> Union[torch.Tensor, int]:
  """
  Computes the output length of the convolutional layers

  Args:
    config: ml_collections.ConfigDict object
    input_lengths: Tensor object of shape [N, ]

  Returns:
    output_lengths: Tensor object of shape [N, ]
  """

  def _conv_out_length(
    input_length: Union[torch.Tensor, int], kernel_size: int, stride: int
  ) -> Union[torch.Tensor, int]:
    # 1D convolutional layer output length formula taken
    # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # return torch.div(
    #   input_length - kernel_size, stride, rounding_mode="floor") + 1
    return (
      torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
    )

  for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
    input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

  return input_lengths


def get_feature_vector_attention_mask(
  config: ml_collections.ConfigDict,
  feature_vector_length: int,
  attention_mask: torch.Tensor,
) -> torch.Tensor:
  """Reduce attention mask of raw input to that of extracted features."""
  # Effectively attention_mask.sum(-1), but not inplace to be able to run
  # on inference mode.
  non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

  output_lengths = get_feat_extract_output_lengths(config, non_padded_lengths)
  if isinstance(output_lengths, int):
    output_lengths = torch.tensor(output_lengths, dtype=torch.long)
  else:
    output_lengths = output_lengths.to(torch.long)

  batch_size = attention_mask.shape[0]

  attention_mask = torch.zeros(
    (batch_size, feature_vector_length),
    dtype=attention_mask.dtype,
    device=attention_mask.device,
  )
  # these two operations makes sure that all values
  # before the output lengths idxs are attended to
  attention_mask[
    (
      torch.arange(attention_mask.shape[0], device=attention_mask.device),
      output_lengths - 1,
    )
  ] = 1
  attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
  return attention_mask


def compute_mask_indices(
  shape: Tuple[int, int],
  mask_prob: float,
  mask_length: int,
  attention_mask: Optional[torch.LongTensor] = None,
  min_masks: int = 0,
) -> np.ndarray:
  """
  Computes random mask spans for a given shape.

  Args:
    shape: The shape for which to compute masks. This should be of a tuple
      of size 2 where the first element is the batch size and the second
      element is the length of the axis to span.
    mask_prob:  The percentage of the whole axis (between 0 and 1) which will
      be masked. The number of independently generated mask spans of length
      `mask_length` is computed by `mask_prob*shape[1]/mask_length`.
      Note that due to overlaps, `mask_prob` is an upper bound and the
      actual percentage will be smaller.
    mask_length: size of the mask
    min_masks: minimum number of masked spans
    attention_mask: A (right-padded) attention mask which
        independently shortens the feature axis of each batch dimension.
  """
  batch_size, sequence_length = shape

  if mask_length < 1:
    raise ValueError("`mask_length` has to be bigger than 0.")

  if mask_length > sequence_length:
    raise ValueError(
      f"`mask_length` has to be smaller than `sequence_length`, "
      f"but got `mask_length`: {mask_length}"
      f" and `sequence_length`: {sequence_length}`"
    )

  # epsilon is used for probabilistic rounding
  epsilon = np.random.rand(1).item()

  def compute_num_masked_span(input_length: int) -> int:
    """Given input length, compute how many spans should be masked"""
    num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
    num_masked_span = max(num_masked_span, min_masks)

    # make sure num masked span <= sequence_length
    if num_masked_span * mask_length > sequence_length:
      num_masked_span = sequence_length // mask_length

    # make sure num_masked span is also <= input_length - (mask_length - 1)
    if input_length - (mask_length - 1) < num_masked_span:
      num_masked_span = max(input_length - (mask_length - 1), 0)

    return num_masked_span

  # compute number of masked spans in batch
  input_lengths = (
    attention_mask.sum(-1).detach().tolist()
    if attention_mask is not None
    else [sequence_length for _ in range(batch_size)]
  )

  # SpecAugment mask to fill
  spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
  spec_aug_mask_idxs = []

  max_num_masked_span = compute_num_masked_span(sequence_length)

  if max_num_masked_span == 0:
    return spec_aug_mask

  for input_length in input_lengths:
    # compute num of masked spans for tVhis input
    num_masked_span = compute_num_masked_span(input_length)

    # get random indices to mask
    spec_aug_mask_idx = np.random.choice(
      np.arange(input_length - (mask_length - 1)),
      num_masked_span,
      replace=False,
    )

    # pick first sampled index that will serve as a dummy index to pad vector
    # to ensure same dimension for all batches due to probabilistic rounding
    # Picking first sample just pads those vectors twice.
    if len(spec_aug_mask_idx) == 0:
      # this case can only happen if `input_length` is strictly smaller then
      # `sequence_length` in which case the last token has to be a padding
      # token which we can use as a dummy mask id
      dummy_mask_idx = sequence_length - 1
    else:
      dummy_mask_idx = spec_aug_mask_idx[0]

    spec_aug_mask_idx = np.concatenate(
      [
        spec_aug_mask_idx,
        np.ones(max_num_masked_span - num_masked_span, dtype=np.int32)
        * dummy_mask_idx,
      ]
    )
    spec_aug_mask_idxs.append(spec_aug_mask_idx)

  spec_aug_mask_idxs_array = np.array(spec_aug_mask_idxs)

  # expand masked indices to masked spans
  spec_aug_mask_idxs_array = np.broadcast_to(
    spec_aug_mask_idxs_array[:, :, None],
    (batch_size, max_num_masked_span, mask_length),
  )
  spec_aug_mask_idxs_array = spec_aug_mask_idxs_array.reshape(
    batch_size, max_num_masked_span * mask_length
  )

  # add offset to the starting indexes so that indexes now create a span
  offsets = np.arange(mask_length)[None, None, :]
  offsets = np.broadcast_to(
    offsets, (batch_size, max_num_masked_span, mask_length)
  ).reshape(batch_size, max_num_masked_span * mask_length)
  spec_aug_mask_idxs_array = spec_aug_mask_idxs_array + offsets

  # ensure that we cannot have indices larger than sequence_length
  if spec_aug_mask_idxs_array.max() > sequence_length - 1:
    spec_aug_mask_idxs_array[spec_aug_mask_idxs_array > sequence_length - 1] = (
      sequence_length - 1
    )

  # scatter indices to mask
  np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs_array, 1, -1)

  return spec_aug_mask


def sample_negative_indices(
  features_shape: Tuple,
  num_negatives: int,
  mask_time_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
  """
  Sample `num_negatives` vectors from feature vectors.
  """
  batch_size, sequence_length = features_shape

  # generate indices of the positive vectors themselves, repeat them `num_negatives` times
  sequence_length_range = np.arange(sequence_length)

  # get `num_negatives` random vector indices from the same utterance
  sampled_negative_indices = np.zeros(
    shape=(batch_size, sequence_length, num_negatives), dtype=np.int32
  )

  mask_time_indices = (
    mask_time_indices.astype(bool)
    if mask_time_indices is not None
    else np.ones(features_shape, dtype=bool)
  )

  for batch_idx in range(batch_size):
    high = mask_time_indices[batch_idx].sum() - 1
    mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

    feature_indices = np.broadcast_to(
      np.arange(high + 1)[:, None], (high + 1, num_negatives)
    )
    sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
    # avoid sampling the same positive vector, but keep the distribution uniform
    sampled_indices[sampled_indices >= feature_indices] += 1

    # remap to actual indices
    sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = (
      mapped_masked_indices[sampled_indices]
    )

    # correct for batch size
    sampled_negative_indices[batch_idx] += batch_idx * sequence_length

  return sampled_negative_indices
