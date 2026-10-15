"""Select the original normalization implementation without eager training imports."""

from torch import nn


def layer_norm_class(use_rms_norm: bool):
  if not use_rms_norm:
    return nn.LayerNorm
  try:
    from torchtune.modules import RMSNorm
  except ImportError as exc:
    raise ImportError(
      "This experimental RMSNorm configuration requires torchtune and a "
      "compatible torchao installation. The published SeisLM encoders use "
      "LayerNorm and do not require these packages."
    ) from exc
  return RMSNorm
