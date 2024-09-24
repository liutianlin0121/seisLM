from typing import Any, Dict

import numpy as np
import seisbench.generate as sbg


class StdSafeNormalize(sbg.Normalize):  # type: ignore
  def _amp_norm(self, x: np.ndarray) -> np.ndarray:
    if self.amp_norm_axis is not None:
      if self.amp_norm_type == "peak":
        x = x / (
          np.max(np.abs(x), axis=self.amp_norm_axis, keepdims=True) + self.eps
        )
      elif self.amp_norm_type == "std":
        std = np.std(x, axis=self.amp_norm_axis, keepdims=True)
        std = np.where(std == 0, 1, std)
        x = x / (std + self.eps)
    return x


class FillMissingComponents:
  def __init__(
    self,
    key: str = "X",
  ):
    if isinstance(key, str):
      self.key = (key, key)
    else:
      self.key = key

  def __call__(self, state_dict: Dict) -> Any:
    x, metadata = state_dict[self.key[0]]

    if isinstance(x, list):
      x = [self._fill_missing_component(y) for y in x]
    else:
      x = self._fill_missing_component(x)

    state_dict[self.key[1]] = (x, metadata)

  def _fill_missing_component(self, x: np.ndarray) -> np.ndarray:
    std_devs = np.std(x, axis=1)

    # Find rows with zero standard deviation
    zero_std_rows = np.where(std_devs == 0)[0]
    non_zero_std_rows = np.where(std_devs != 0)[0]

    # If all rows or zero rows have zero standard deviation, return as is
    if len(zero_std_rows) == 0 or len(zero_std_rows) == x.shape[0]:
      return x

    # Otherwise, randomly replace zero-std rows with non-zero std rows
    for row_idx in zero_std_rows:
      replacement_row = np.random.choice(non_zero_std_rows)
      x[row_idx] = x[replacement_row]
    return x
