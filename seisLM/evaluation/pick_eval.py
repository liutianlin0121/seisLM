"""  Module for evaluating phase-picking performance.

https://github.com/seisbench/pick-benchmark
"""

from typing import Optional, Dict
from pathlib import Path
import logging
import numpy as np
from sklearn import metrics
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
import seisbench.data as sbd
import seisbench.generate as sbg
from seisLM.utils import project_path

def get_dataset_by_name(name: str):
  """
  Resolve dataset name to class from seisbench.data.

  :param name: Name of dataset as defined in seisbench.data.
  :return: Dataset class from seisbench.data
  """
  try:
    return sbd.__getattribute__(name)
  except AttributeError:
    raise ValueError(f"Unknown dataset '{name}'.")


def _identify_instance_dataset_border(task_targets: Dict) -> int:
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
  model: L.LightningModule,
  target_path: str,
  sets: str,
  save_tag: str,
  batch_size: int = 1024,
  num_workers: int = 4,
  sampling_rate: Optional[int] = None,
  ) -> None:

  targets = Path(target_path)
  sets = sets.split(",")
  model.eval()

  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  dataset = get_dataset_by_name(targets.name)(
      sampling_rate=100, component_order="ZNE", dimension_order="NCW",
      cache="full"
  )

  if sampling_rate is not None:
    dataset.sampling_rate = sampling_rate
    pred_root = pred_root + "_resampled"
    weight_path_name = weight_path_name + f"_{sampling_rate}"

  for eval_set in sets:
    split = dataset.get_split(eval_set)
    if targets.name == "InstanceCountsCombined":
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

      if task == "1" and targets.name == "InstanceCountsCombined":
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
        generator, batch_size=batch_size, shuffle=False, num_workers=num_workers
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
        / f"{save_tag}_{targets.name}"
        / f"{eval_set}_task{task}.csv"
      )
      pred_path.parent.mkdir(exist_ok=True, parents=True)
      # pred_path = f'./{eval_set}_task{task}.csv'
      print(f"Saving predictions to {pred_path}")
      task_targets.to_csv(pred_path, index=False)


def get_results_event_detection(pred_path):
  pred = pd.read_csv(pred_path)
  pred["trace_type_bin"] = pred["trace_type"] == "earthquake"
  pred["score_detection"] = pred["score_detection"].fillna(0)

  fpr, tpr, _ = metrics.roc_curve(
    pred["trace_type_bin"], pred["score_detection"])
  prec, recall, thr = metrics.precision_recall_curve(
    pred["trace_type_bin"], pred["score_detection"]
  )
  auc = metrics.roc_auc_score(
    pred["trace_type_bin"], pred["score_detection"]
  )


  f1 = 2 * prec * recall / (prec + recall)
  f1_threshold = thr[np.nanargmax(f1)]
  best_f1 = np.max(f1)

  return {
    'auc': auc,
    'fpr': fpr,
    'tpr': tpr,
    'prec': prec,
    'recall': recall,
    'f1': f1,
    'f1_threshold': f1_threshold,
    'best_f1': best_f1
  }

def get_results_phase_identification(pred_path):
  pred = pd.read_csv(pred_path)
  pred["phase_label_bin"] = pred["phase_label"] == "P"
  pred["score_p_or_s"] = pred["score_p_or_s"].fillna(0)
  fpr, tpr, _ = metrics.roc_curve(
    pred["phase_label_bin"], pred["score_p_or_s"]
  )
  prec, recall, thr = metrics.precision_recall_curve(
    pred["phase_label_bin"], pred["score_p_or_s"]
  )
  f1 = 2 * prec * recall / (prec + recall)
  f1_threshold = thr[np.nanargmax(f1)]
  best_f1 = np.nanmax(f1)

  return {
    'fpr': fpr,
    'tpr': tpr,
    'prec': prec,
    'recall': recall,
    'f1': f1,
    'f1_threshold': f1_threshold,
    'best_f1': best_f1
  }

def get_results_onset_determination(pred_path):
  pred = pd.read_csv(pred_path)
  results = {}
  for phase in ['P', 'S']:
    pred_phase = pred[pred["phase_label"] == phase]
    pred_col = f"{phase.lower()}_sample_pred"
    diff = (pred_phase[pred_col] - pred_phase["phase_onset"]
            ) / pred_phase["sampling_rate"]
    results[f'{phase}_onset_diff'] = diff
  return results
