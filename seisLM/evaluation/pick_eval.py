"""  Module for evaluating phase-picking performance.

https://github.com/seisbench/pick-benchmark
"""

from pathlib import Path
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
import lightning as L
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisLM.model import supervised_models
from seisLM.utils import project_path

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


def _identify_instance_dataset_border(task_targets):
  """
  Calculates the dataset border between Signal and Noise for instance,
  assuming it is the only place where the bucket number does not increase
  """
  buckets = task_targets["trace_name"].apply(lambda x: int(x.split("$")[0][6:]))

  last_bucket = 0
  for i, bucket in enumerate(buckets):
    if bucket < last_bucket:
      return i
    last_bucket = bucket



def save_pick_predictions(
  checkpoint_path_or_data_name, model_name, targets, sets, batchsize=1024, num_workers=4,
  sampling_rate=None):
  targets = Path(targets)
  sets = sets.split(",")


  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  model_cls = supervised_models.__getattribute__(model_name + "Lit")
  if 'ckpt' in checkpoint_path_or_data_name:
    # In case of a checkpoint, load the model from the checkpoint
    model = model_cls.load_from_checkpoint(checkpoint_path_or_data_name)
  else:
    # In case of a data name, load the model from the pretrained model
    # TODO: This won't work if the model takes positional argument.
    model = model_cls()
    model.model = sbm.__getattribute__(model_name).from_pretrained(
      checkpoint_path_or_data_name
    )
    print(model.model.weights_docstring)

  dataset = get_dataset_by_name(data_aliases[targets.name])(
      sampling_rate=100, component_order="ZNE", dimension_order="NCW",
      cache="full"
  )

  if sampling_rate is not None:
    dataset.sampling_rate = sampling_rate
    pred_root = pred_root + "_resampled"
    weight_path_name = weight_path_name + f"_{sampling_rate}"

  for eval_set in sets:
    split = dataset.get_split(eval_set)
    if targets.name == "instance":
      logging.warning(
          "Overwriting noise trace_names to allow correct identification"
      )
      # Replace trace names for noise entries
      split._metadata["trace_name"].values[
          -len(split.datasets[-1]) :
      ] = split._metadata["trace_name"][-len(split.datasets[-1]) :].apply(
          lambda x: "noise_" + x
      )
      split._build_trace_name_to_idx_dict()

    logging.warning(f"Starting set {eval_set}")
    split.preload_waveforms(pbar=True)

    for task in ["1", "23"]:

      task_csv = targets / f"task{task}.csv"

      if not task_csv.is_file():
        continue

      logging.warning(f"Starting task {task}")

      task_targets = pd.read_csv(task_csv)
      task_targets = task_targets[task_targets["trace_split"] == eval_set]

      if task == "1" and targets.name == "instance":
        border = _identify_instance_dataset_border(task_targets)
        task_targets["trace_name"].values[border:] = task_targets["trace_name"][
            border:
        ].apply(lambda x: "noise_" + x)

      if sampling_rate is not None:
        for key in ["start_sample", "end_sample", "phase_onset"]:
          if key not in task_targets.columns:
              continue
          task_targets[key] = (
              task_targets[key]
              * sampling_rate
              / task_targets["sampling_rate"]
          )
        task_targets[sampling_rate] = sampling_rate

      generator = sbg.SteeredGenerator(split, task_targets)
      generator.add_augmentations(model.get_eval_augmentations())

      loader = DataLoader(
        generator, batch_size=batchsize, shuffle=False, num_workers=num_workers
      )
      trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,            # Disable the default logger
        enable_checkpointing=False  # Disable automatic checkpointing
      )


      predictions = trainer.predict(model, loader)

      # Merge batches
      merged_predictions = []
      for i, _ in enumerate(predictions[0]):
        merged_predictions.append(torch.cat([x[i] for x in predictions]))

      merged_predictions = [x.cpu().numpy() for x in merged_predictions]
      task_targets["score_detection"] = merged_predictions[0]
      task_targets["score_p_or_s"] = merged_predictions[1]
      task_targets["p_sample_pred"] = (
          merged_predictions[2] + task_targets["start_sample"]
      )
      task_targets["s_sample_pred"] = (
          merged_predictions[3] + task_targets["start_sample"]
      )


      pred_path = (
        Path(project_path.EVAL_SAVE_DIR)
        / f"{model_name}_{targets.name}"
        / f"{eval_set}_task{task}.csv"
      )
      pred_path.parent.mkdir(exist_ok=True, parents=True)
      # pred_path = f'./{eval_set}_task{task}.csv'
      task_targets.to_csv(pred_path, index=False)

