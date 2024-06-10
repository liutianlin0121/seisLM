import numpy as np

class RMSNorm:
  """
  Root mean square normalization.
  https://arxiv.org/pdf/1910.07467

  Attributes:
    rms_axis: The axis root mean square should be computed.
    eps: A small positive constant for stability purposes.
    key: The keys for reading from and writing to the state dict.
  """

  def __init__(self, rms_axis: int, eps=1e-10, key="X"):
    self.eps = eps
    self.rms_axis = rms_axis
    if isinstance(key, str):
      self.key = (key, key)
    else:
      self.key = key


  def __call__(self, state_dict):
    x, metadata = state_dict[self.key[0]]

    if isinstance(x, list):
      x = [self._rms_norm(y) for y in x]
    else:
      x = self._rms_norm(x)

    state_dict[self.key[1]] = (x, metadata)

  def _rms_norm(self, x):
    denom = np.sqrt((x * x).mean(axis=self.rms_axis, keepdims=True) + self.eps)
    x = x / denom
    return x


