import numpy as np
import seisbench.generate as sbg


class StdSafeNormalize(sbg.Normalize):
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
