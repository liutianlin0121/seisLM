""" Dataloaders for SeisBench datasets. """
import logging
import numpy as np
import seisbench.generate as sbg
import seisbench.data as sbd
from torch.utils.data import DataLoader
from seisbench.util import worker_seeding
from seisbench.data import MultiWaveformDataset


def get_dataset_by_name(name):
  """
  Resolve dataset name to class from seisbench.data.

  :param name: Name of dataset as defined in seisbench.data.
  :return: Dataset class from seisbench.data
  """
  try:
    return sbd.__getattribute__(name)
  except AttributeError:
    raise ValueError(f"Unknown dataset '{name}'.")


def apply_training_fraction(training_fraction, train_data):
  """
  Reduces the size of train_data to train_fraction by inplace filtering.
  Filter blockwise for efficient memory savings.

  :param training_fraction: Training fraction between 0 and 1.
  :param train_data: Training dataset
  :return: None
  """

  if not 0.0 < training_fraction <= 1.0:
    raise ValueError("Training fraction needs to be between 0 and 1.")

  if training_fraction < 1:
    blocks = train_data["trace_name"].apply(lambda x: x.split("$")[0])
    unique_blocks = blocks.unique()
    np.random.shuffle(unique_blocks)
    target_blocks = unique_blocks[: int(training_fraction * len(unique_blocks))]
    target_blocks = set(target_blocks)
    mask = blocks.isin(target_blocks)
    train_data.filter(mask, inplace=True)


def prepare_seisbench_dataloaders(
  model, data_names, batch_size, num_workers,
  training_fraction=1.0,
  sampling_rate=100,
  component_order="ZNE", dimension_order="NCW",
  collator=None, cache=None, prefetch_factor=2,
  ):
  """
  Returns the training and validation data loaders
  :param config:
  :param model:
  :return:
  """
  if isinstance(data_names, str):
    data_names = [data_names]

  # train_generators = []
  # dev_generators = []

  multi_waveform_datasets = []
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

  if len(multi_waveform_datasets) == 1:
    dataset = multi_waveform_datasets[0]
  else:
    # Concatenate multiple datasets
    dataset = MultiWaveformDataset(multi_waveform_datasets)

  train_data, dev_data = dataset.train(), dataset.dev()
  apply_training_fraction(training_fraction, train_data)

  train_data.preload_waveforms(pbar=True)
  dev_data.preload_waveforms(pbar=True)

  train_generator = sbg.GenericGenerator(train_data)
  dev_generator = sbg.GenericGenerator(dev_data)

  train_generator.add_augmentations(model.get_train_augmentations())
  dev_generator.add_augmentations(model.get_val_augmentations())


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
  dev_loader = DataLoader(
      dev_generator,
      batch_size=batch_size,
      num_workers=num_workers,
      worker_init_fn=worker_seeding,
      pin_memory=True,
      collate_fn=collator,
      prefetch_factor=prefetch_factor,
  )

  return train_loader, dev_loader



data_aliases = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "stead": "STEAD",
    "neic": "NEIC",
    "instance": "InstanceCountsCombined",
    "iquique": "Iquique",
    "lendb": "LenDB",
    "scedc": "SCEDC",
}

