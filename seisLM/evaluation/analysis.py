""" Helper functions for analyzing the models. """
from typing import Any
import numpy as np

def compute_separation_fuzziness(
  X: np.ndarray,
  labels: np.ndarray,
  use_trace_division: bool = True,
  use_log_scale: bool = True
  ) -> Any:
  """Compute the separation fuzziness metric for the given data and labels.

  Args:
      X: [num_samples, feature_dim] numpy array of data
      labels:  [num_samples, ] numpy array of labels (strings)
      use_trace_division: bool, whether to use the trace division method

  Returns:
      The separation fuzziness metric
  """

  num_samples, feature_dim = X.shape
  unique_labels = np.unique(labels)

  # Overall mean
  X_mean = np.mean(X, axis=0)

  # Between-class scatter matrix (SSB) and within-class scatter matrix (SSW)
  SSB = np.zeros((feature_dim, feature_dim))
  SSW = np.zeros((feature_dim, feature_dim))

  for label in unique_labels:
    # Use direct comparison for string labels
    X_per_class = X[labels == label]

    class_mean = np.mean(X_per_class, axis=0)

    # SSB: between-class scatter
    n_class_samples = X_per_class.shape[0]
    mean_diff = (class_mean - X_mean).reshape(-1, 1)
    SSB += n_class_samples * (mean_diff @ mean_diff.T)

    # SSW: within-class scatter
    diff_per_class = X_per_class - class_mean
    SSW += diff_per_class.T @ diff_per_class

  SSB /= num_samples
  SSW /= num_samples


  # Compute the separation fuzziness metric
  if use_trace_division:
    D = np.trace(SSW) / np.trace(SSB)
    print(f'Trace of SSW: {np.trace(SSW): .2f}, Trace of SSB: {np.trace(SSB): .2f}')
  else:
    # Regularization term to prevent issues with SSB
    # regularization_term = 1e-8 * np.eye(feature_dim)
    # SSB += regularization_term
    D = np.trace(SSW @ np.linalg.pinv(SSB))

  if D <= 0:
    raise ValueError("D is non-positive, check the data and computation.")

  if use_log_scale:
    return np.log(D)
  else:
    return D
