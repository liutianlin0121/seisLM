
import logging
import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
from torch.utils.data import DataLoader
from seisbench.util import worker_seeding

DEFAULT_NUM_WORKERS = 8

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


def prepare_seisbench_dataloaders(
  model, data_name, batch_size, num_workers, collator=None):
  """
  Returns the training and validation data loaders
  :param config:
  :param model:
  :return:
  """
  dataset = get_dataset_by_name(data_name)(
    sampling_rate=100, component_order="ZNE", dimension_order="NCW",
    cache="full"
  )

  if "split" not in dataset.metadata.columns:
    logging.warning("No split defined, adding auxiliary split.")
    split = np.array(["train"] * len(dataset))
    split[int(0.6 * len(dataset)) : int(0.7 * len(dataset))] = "dev"
    split[int(0.7 * len(dataset)) :] = "test"

    dataset._metadata["split"] = split

  train_data = dataset.train()
  dev_data = dataset.dev()

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
  )
  dev_loader = DataLoader(
      dev_generator,
      batch_size=batch_size,
      num_workers=num_workers,
      worker_init_fn=worker_seeding,
      pin_memory=True,
      collate_fn=collator,
  )

  return train_loader, dev_loader
