"""Dataloaders for the foreshock-aftershock dataset."""

from typing import Any, Dict, Optional

import numpy as np
import torch

from seisLM.data_pipeline import foreshock_aftershock_dataset as dataset


def prepare_foreshock_aftershock_dataloaders(
  *,
  num_classes: int,
  batch_size: int,
  event_split_method: str,
  component_order: str,
  seed: int = 42,
  remove_class_overlapping_dates: bool = False,
  train_frac: float = 0.70,
  val_frac: float = 0.10,
  test_frac: float = 0.20,
  demean_axis: Optional[int] = -1,
  amp_norm_axis: Optional[int] = -1,
  amp_norm_type: str = "peak",
  num_workers: int = 8,
  dimension_order: str = "NCW",
  collator: Any = None,
) -> Dict[str, torch.utils.data.DataLoader]:
  """Create dataloaders for the foreshock-aftershock dataset."""

  datasets = dataset.create_foreshock_aftershock_datasets(
    num_classes=num_classes,
    event_split_method=event_split_method,
    component_order=component_order,
    dimension_order=dimension_order,
    seed=seed,
    remove_class_overlapping_dates=remove_class_overlapping_dates,
    train_frac=train_frac,
    val_frac=val_frac,
    test_frac=test_frac,
  )

  X_train, y_train = datasets["train"]["X"], datasets["train"]["y"]
  X_val, y_val = datasets["val"]["X"], datasets["val"]["y"]
  X_test, y_test = datasets["test"]["X"], datasets["test"]["y"]

  def normalize(X: np.ndarray) -> np.ndarray:
    if demean_axis is not None:
      X = X - np.mean(X, axis=demean_axis, keepdims=True)

    if amp_norm_axis is not None:
      if amp_norm_type == "std":
        X = X / (np.std(X, axis=amp_norm_axis, keepdims=True) + 1e-10)
      elif amp_norm_type == "peak":
        X = X / (np.max(np.abs(X), axis=amp_norm_axis, keepdims=True) + 1e-10)
      else:
        raise ValueError(f"Normalization type {amp_norm_type} not supported")
    return X

  X_train = normalize(X_train)
  X_val = normalize(X_val)
  X_test = normalize(X_test)

  X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
  X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
  X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

  loaders = {
    "train": torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_train, y_train),
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=num_workers,
      collate_fn=collator,
    ),
    "val": torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_val, y_val),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
      collate_fn=collator,
    ),
    "test": torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_test, y_test),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
      collate_fn=collator,
    ),
  }

  return loaders
