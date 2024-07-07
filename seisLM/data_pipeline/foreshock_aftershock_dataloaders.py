"""Dataloaders for the foreshock-aftershock dataset. """
import torch
import einops
from seisLM.data_pipeline import foreshock_aftershock_dataset as dataset

def prepare_foreshock_aftershock_dataloaders(
  num_classes: int,
  batch_size: int,
  event_split_method: str,
  seed: int = 42,
  train_frac: float = 0.70,
  val_frac: float = 0.10,
  test_frac: float = 0.20,
  standardize: bool = True,
  num_workers: int = 8,
  dimension_order: str = 'CW'
  ):
  ''' Create dataloaders for the foreshock-aftershock dataset.'''

  datasets = dataset.create_foreshock_aftershock_datasets(
    num_classes=num_classes,
    event_split_method=event_split_method,
    seed=seed,
    train_frac=train_frac,
    val_frac=val_frac,
    test_frac=test_frac,
  )

  X_train, y_train = datasets['train']['X'], datasets['train']['y']
  X_val, y_val = datasets['val']['X'], datasets['val']['y']
  X_test, y_test = datasets['test']['X'], datasets['test']['y']

  if standardize:
    mean, std = X_train.mean(), X_train.std()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

  if dimension_order == 'CW':
    X_train = einops.rearrange(X_train, 'n w c -> n c w')
    X_val = einops.rearrange(X_val, 'n w c -> n c w')
    X_test = einops.rearrange(X_test, 'n w c -> n c w')
  elif dimension_order == 'WC':
    pass
  else:
    raise ValueError(
      f'Invalid dimension_order {dimension_order}. Must be "CW" or "WC".'
    )

  X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
  X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
  X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)


  loaders = {
    'train': torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_train.float(), y_train),
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=num_workers,
    ),
    'val': torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_val.float(), y_val),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
    ),
    'test': torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_test.float(), y_test),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
    )
  }

  return loaders
