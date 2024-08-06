"""Pretrain dataloader"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import seisbench.generate as sbg
from seisbench.data import MultiWaveformDataset
from seisbench.generate.augmentation import Normalize
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader

from seisLM.data_pipeline.foreshock_aftershock_dataloaders import \
    prepare_foreshock_aftershock_dataloaders
from seisLM.data_pipeline.seisbench_dataloaders import (
    apply_training_fraction, get_dataset_by_name)


def prepare_pretrain_dataloaders(
  *,
  model: L.LightningModule,
  data_names: List[str],
  batch_size: int,
  num_workers: int,
  training_fraction: float = 1.0,
  sampling_rate: int = 100,
  component_order: str = "ZNE",
  dimension_order: str = "NCW",
  collator: Optional[Any] = None,
  cache: Optional[str] = None,
  prefetch_factor: int = 2,
  ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
  """
  Returns the training and validation data loaders
  """
  if isinstance(data_names, str):
    data_names = [data_names]

  multi_waveform_datasets = []
  dev_generators = {}
  for data_name in data_names:
    dataset = get_dataset_by_name(data_name)(
      sampling_rate=sampling_rate,
      component_order=component_order,
      dimension_order=dimension_order,
      cache=cache
    )
    if "split" not in dataset.metadata.columns:
      logging.warning("No split defined, adding auxiliary split.")
      split = np.array(["train"] * len(dataset))
      split[int(0.6 * len(dataset)) : int(0.7 * len(dataset))] = "dev"
      split[int(0.7 * len(dataset)) :] = "test"
      dataset._metadata["split"] = split  # pylint: disable=protected-access
    multi_waveform_datasets.append(dataset)
    dev_generator = sbg.GenericGenerator(dataset.dev())
    dev_generator.add_augmentations(model.get_val_augmentations())
    dev_generators[data_name] = dev_generator

  if len(multi_waveform_datasets) == 1:
    dataset = multi_waveform_datasets[0]
  else:
    # Concatenate multiple datasets
    dataset = MultiWaveformDataset(multi_waveform_datasets)

  train_data = dataset.train()
  apply_training_fraction(training_fraction, train_data)

  if cache:
    train_data.preload_waveforms(pbar=True)

  train_generator = sbg.GenericGenerator(train_data)
  train_generator.add_augmentations(model.get_train_augmentations())

  train_loader = DataLoader(
      train_generator,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      worker_init_fn=worker_seeding,
      drop_last=True,  # Avoid crashes from batch norm layers for batch size 1
      pin_memory=True,
      collate_fn=collator,
      prefetch_factor=prefetch_factor,
  )

  dev_loaders = {}
  for data_name, dev_generator in dev_generators.items():
    dev_loaders[data_name] = DataLoader(
      dev_generator,
      batch_size=batch_size,
      num_workers=num_workers,
      worker_init_fn=worker_seeding,
      pin_memory=True,
      collate_fn=collator,
      prefetch_factor=prefetch_factor,
  )


  norm = model.get_val_augmentations()[-1]
  assert isinstance(norm, Normalize)

  shock_loaders = prepare_foreshock_aftershock_dataloaders(
    num_classes=4, # doesn't matter
    batch_size=batch_size,
    component_order=component_order,
    event_split_method='temporal',
    demean_axis=norm.demean_axis,
    amp_norm_axis=norm.amp_norm_axis,
    amp_norm_type=norm.amp_norm_type,
    collator=collator,
  )

  dev_loaders['shock'] = shock_loaders['val']
  return train_loader, dev_loaders
