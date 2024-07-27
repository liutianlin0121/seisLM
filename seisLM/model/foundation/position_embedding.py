""" Positional embeddings for Wav2Vec2. """
from typing import Tuple
import ml_collections
import torch
from torch import nn, Tensor
import einops

class Wav2Vec2PositionalConvEmbedding(nn.Module):
  """Use a convolutional layer, which acts as relative positional embedding."""
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.conv = nn.Conv1d(
        config.hidden_size,
        config.hidden_size,
        kernel_size=config.num_conv_pos_embeddings,
        padding=config.num_conv_pos_embeddings // 2,
        groups=config.num_conv_pos_embedding_groups,
    )

    weight_norm = nn.utils.weight_norm
    if hasattr(nn.utils.parametrizations, "weight_norm"):
      weight_norm = nn.utils.parametrizations.weight_norm

    self.conv = weight_norm(self.conv, name="weight", dim=2)

    self.activation = nn.functional.gelu

    # With a kernel size k, padding k//2, and stride 1, the output of the
    # conv layer has a length of (input_length + 2 (k//2) - k + 1).
    # So if k is even, the output is 1 element longer than the input;
    # we remove the last element to ensure that the
    # position embeddings have the same size as the input sequence.
    # If k is odd, the output has the same length as the input, so we don't
    # need to remove any elements.
    self.remove_one_right = (
      True if config.num_conv_pos_embeddings % 2 == 0 else False
    )

  def forward(self, hidden_states: Tensor) -> Tensor:
    hidden_states = einops.rearrange(hidden_states, "b t c -> b c t")
    hidden_states = self.conv(hidden_states)
    if self.remove_one_right:
      hidden_states = hidden_states[:, :, :-1]

    hidden_states = self.activation(hidden_states)
    hidden_states = einops.rearrange(hidden_states, "b c t -> b t c")
    return hidden_states



def precompute_freqs_cis(
  dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
  """
  Precompute the frequency tensor for complex exponentials (cis)
  with given dimensions.

  This function calculates a frequency tensor with complex exponentials using
  the given dimension 'dim'
  and the end index 'end'. The 'theta' parameter scales the frequencies.
  The returned tensor contains complex values in complex64 data type.

  Args:
      dim (int): Dimension of the frequency tensor.
      end (int): End index for precomputing frequencies.
      theta (float, optional): Scaling factor for frequency computation.
      Defaults to 10000.0.

  Returns:
      torch.Tensor: Precomputed frequency tensor with complex exponentials.

  """
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


def reshape_for_broadcast(
  freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  """
  Reshape frequency tensor for broadcasting it with another tensor.

  This function reshapes the frequency tensor to have the same shape as
  the target tensor 'x'
  for the purpose of broadcasting the frequency tensor during element-wise
  operations.

  Args:
      freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
      x (torch.Tensor): Target tensor for broadcasting compatibility.

  Returns:
      torch.Tensor: Reshaped frequency tensor.

  Raises:
      AssertionError: If the frequency tensor doesn't match the expected shape.
      AssertionError: If the target tensor 'x' doesn't have the expected number
      of dimensions.
  """
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Apply rotary embeddings to input tensors using the given frequency tensor.

  This function applies rotary embeddings to the given query 'xq' and key
    'xk' tensors using the provided
  frequency tensor 'freqs_cis'. The input tensors are reshaped as complex
    numbers, and the frequency tensor
  is reshaped for broadcasting compatibility. The resulting tensors contain
    rotary embeddings and are
  returned as real tensors.

  Args:
      xq (torch.Tensor): Query tensor to apply rotary embeddings.
      xk (torch.Tensor): Key tensor to apply rotary embeddings.
      freqs_cis (torch.Tensor): Precomputed frequency tensor for
        complex exponentials.

  Returns:
      Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key
      tensor with rotary embeddings.



  """
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)
