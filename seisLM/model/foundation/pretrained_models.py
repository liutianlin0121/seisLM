"""Wav2Vec2 model."""
import logging
from collections import defaultdict
from typing import Dict, List, Any
import copy
import math
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
from lightning.pytorch.utilities import grad_norm
import ml_collections
import seisbench.generate as sbg
from seisLM.data_pipeline.augmentations import (
  StdSafeNormalize, FillMissingComponents)
from seisLM.model.foundation.multidim_wav2vec2 import MultiDimWav2Vec2ForPreTraining
from seisLM.utils.data_utils import phase_dict

class LitMultiDimWav2Vec2(L.LightningModule):
  """LightningModule for Wav2Vec2 model."""
  def __init__(
    self,
    config: ml_collections.ConfigDict,
    ) -> None:

    super().__init__()
    self.config = config
    self.model = MultiDimWav2Vec2ForPreTraining(config.model_config)
    self.save_hyperparameters()
    val_dataloader_names = copy.deepcopy(config.data_config.data_name)
    val_dataloader_names = val_dataloader_names + ['shock']
    self.val_dataloader_names = val_dataloader_names

    # Create a list to hold the outputs of `validation_step`
    self.validation_step_outputs = []

  def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    # inspect (unscaled) gradients here
    self.log_dict(grad_norm(self, norm_type=2))

  def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
    # pylint:disable=missing-function-docstring
    # pylint:disable=invalid-name

    mask_time_indices = batch["mask_time_indices"]
    num_losses = mask_time_indices.sum()
    percent_masked = mask_time_indices.float().mean()

    # forward
    outputs = self.model(**batch)
    loss = outputs.loss / num_losses

    temperature_max_min_gap = (
        self.config.training_config.max_gumbel_temperature -\
          self.config.training_config.min_gumbel_temperature
          )

    # Compute the total number of optimization steps that will be taken
    total_optimization_steps = self.config.training_config.max_train_steps // (
        self.trainer.num_devices * self.trainer.accumulate_grad_batches
    )

    # Calculate the ratio of completed steps based on the global step and the
    # total optimization steps
    ratio_completed_steps = self.trainer.global_step / total_optimization_steps

    # Calculate the temperature factor based on the completed ratio
    temperature_factor = (1 + math.cos(math.pi * ratio_completed_steps))/2

    # Compute the gumbel temperature based on the temperature factor
    gumbel_temperature = \
        self.config.training_config.min_gumbel_temperature + (
          temperature_factor * temperature_max_min_gap
        )

    self.model.set_gumbel_temperature(gumbel_temperature)

    self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True)

    train_logs = {
        "train/constrast_loss": outputs.contrastive_loss / num_losses,
        "train/div_loss": outputs.diversity_loss / num_losses,
        "train/%_mask_idx": percent_masked,
        "train/global_step": self.trainer.global_step,
        "train/batch_idx": batch_idx,
        "train/temperature_factor": temperature_factor,
    }

    self.log_dict(train_logs, sync_dist=True)
    self.log("train/gumbel_temperature", gumbel_temperature, prog_bar=True)
    self.log("train/ratio_completed_steps", ratio_completed_steps, prog_bar=True)
    self.log("train/ppl", outputs.codevector_perplexity, prog_bar=True)
    return loss


  def validation_step(
    self, batch: Dict, batch_idx: int, dataloader_idx: int=0) -> Dict:
    # pylint:disable=missing-function-docstring
    # pylint:disable=invalid-name
    data_name = self.val_dataloader_names[dataloader_idx]

    num_losses = batch["mask_time_indices"].sum().float()
    outputs = self.model(**batch)
    validation_outputs = {
      f"val/loss/{data_name}": outputs.loss / num_losses,
      f"val/contrastive_loss/{data_name}": outputs.contrastive_loss / num_losses,
      f"val/diversity_loss/{data_name}": outputs.diversity_loss / num_losses,
      f"val/ppl/{data_name}": outputs.codevector_perplexity,
      f"val/num_losses/{data_name}": num_losses,
    }
    # Average losses across all batches and all devices
    self.log_dict(
      validation_outputs, reduce_fx="mean",
      on_step=False, on_epoch=True, sync_dist=True,
      add_dataloader_idx=False
    )

    self.validation_step_outputs.append(validation_outputs)
    return validation_outputs

  def on_validation_epoch_end(self) -> None:

    # Step 1: Combine all dictionaries into a single dictionary
    sums = defaultdict(float)
    counts = defaultdict(int)

    # Step 2: Iterate over each dictionary and sum up the values
    for entry in self.validation_step_outputs:
      for key, value in entry.items():
        base_key = '/'.join(key.split('/')[:2])  # Extract 'val/loss/', etc.
        sums[base_key] += value.item()
        counts[base_key] += 1

    # Step 3: Calculate averages
    averages = {}
    for key in sums:
      avg_key = key.replace('val/', 'val/avg_')
      averages[avg_key] = sums[key] / counts[key]

    self.log_dict(
      averages, on_step=False, on_epoch=True, sync_dist=True,
    )

    self.validation_step_outputs.clear() # free up memory
    assert len(self.validation_step_outputs) == 0


  def configure_optimizers(self): # type: ignore
    optimizer = torch.optim.AdamW(
        params=self.model.parameters(),
        lr=self.config.training_config.learning_rate,
        weight_decay=self.config.training_config.weight_decay,
        betas=(
          self.config.training_config.adam_beta1,
          self.config.training_config.adam_beta2
        ),
        eps=self.config.training_config.adam_epsilon,
    )

    t_max = int(
      self.config.training_config.max_train_steps // self.trainer.num_devices
    )
    t_warmup = int((self.config.training_config.warmup_frac_step * (
      self.config.training_config.max_train_steps)) // self.trainer.num_devices
    )

    # Linear warmup and half-cycle cosine decay
    def lr_lambda(step: int) -> Any:
      if step < t_warmup:
        # Linear warm-up
        return step / t_warmup
      else:
        # Cosine annealing over remaining steps
        return 0.5 * (
          1 + np.cos((step - t_warmup) * math.pi / (t_max - t_warmup))
        )

    sched_config = {
        'scheduler': LambdaLR(optimizer, lr_lambda),
        'interval': "step",
        'frequency': 1,
    }
    return {"optimizer": optimizer, "lr_scheduler": sched_config}

  def get_train_augmentations(self) -> List:
    augmentation_list = [
        sbg.RandomWindow(
            low=None,
            high=None,
            windowlen=3001,
            strategy="pad",
        ),
        sbg.ChangeDtype(np.float32),
        FillMissingComponents(),
        StdSafeNormalize(
          demean_axis=tuple(self.config.data_config.demean_axis) if isinstance(
            self.config.data_config.demean_axis, list
          ) else self.config.data_config.demean_axis,
          amp_norm_axis=tuple(self.config.data_config.amp_norm_axis) if isinstance(
              self.config.data_config.amp_norm_axis, list
          ) else self.config.data_config.amp_norm_axis,
          amp_norm_type=self.config.data_config.amp_norm_type,
          eps=self.config.data_config.get('norm_eps', 1e-10),
        ),
    ]
    if self.config.data_config.get('sample_around_picks', True):
      # Select windows around picks to reduce the amount of noise traces in
      # training.
      augmentation_list.insert(
        0,
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=6000,
            selection="random",
            strategy="variable",
        ),
      )
      logging.warning("Sampling around picks during training.")

    print('train augmentation_list:', augmentation_list)
    return augmentation_list

  def get_val_augmentations(self) -> List:
    # return self.get_train_augmentations()
    return [
        # Select windows around picks to reduce the amount of noise traces
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=6000,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(
            low=None,
            high=None,
            windowlen=3001,
            strategy="pad",
        ),
        sbg.ChangeDtype(np.float32),
        FillMissingComponents(),
        StdSafeNormalize(
          demean_axis=tuple(self.config.data_config.demean_axis) if isinstance(
            self.config.data_config.demean_axis, list
          ) else self.config.data_config.demean_axis,
          amp_norm_axis=tuple(self.config.data_config.amp_norm_axis) if isinstance(
              self.config.data_config.amp_norm_axis, list
          ) else self.config.data_config.amp_norm_axis,
          amp_norm_type=self.config.data_config.amp_norm_type,
          eps=self.config.data_config.get('norm_eps', 1e-10),
        ),
    ]
