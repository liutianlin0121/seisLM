"""Wav2Vec2 model."""
from typing import Dict, List, Any
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
import ml_collections
import seisbench.generate as sbg
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

    ratio_completed_steps = self.trainer.global_step / (
        self.config.training_config.max_train_steps // self.trainer.num_devices
    )
    temperature_factor = (1 + math.cos(math.pi * ratio_completed_steps))/2

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
        "train/ppl": outputs.codevector_perplexity,
        "train/gumbel_temperature": gumbel_temperature,
        "train/global_step": self.trainer.global_step,
        "train/batch_idx": batch_idx,
        "train/temperature_factor": temperature_factor,
        "train/ratio_completed_steps": ratio_completed_steps,
    }

    self.log_dict(train_logs, sync_dist=True)
    return loss


  def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
    # pylint:disable=missing-function-docstring
    # pylint:disable=invalid-name
    outputs = self.model(**batch)
    validation_outputs = {
      "val/sum_loss": outputs.loss,
      "val/sum_contrastive_loss": outputs.contrastive_loss,
      "val/sum_diversity_loss": outputs.diversity_loss,
      "val/sum_num_losses": batch["mask_time_indices"].sum().float(),
    }

    # Sum losses across all batches and all devices
    self.log_dict(
      validation_outputs, reduce_fx="sum",
      on_step=False, on_epoch=True, sync_dist=True
    )
    return validation_outputs

  def on_validation_epoch_end(self) -> None:
    # pylint:disable=missing-function-docstring
    # pylint:disable=invalid-name

    metrics = self.trainer.logged_metrics
    sum_num_losses = metrics.get('val/sum_num_losses', 0)
    if sum_num_losses == 0:
      raise ValueError("total_num_losses=0. Something went wrong.")

    avg_metrics = {}
    metric_keys = list(metrics.keys())
    for key in metric_keys:
      if key.startswith('val/sum_'):
        value = metrics.pop(key)
        avg_key = key.replace('sum_', '')
        avg_metrics[avg_key] = value / sum_num_losses
    avg_metrics.pop('val/num_losses')
    self.log_dict(avg_metrics, prog_bar=True, sync_dist=True)

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
    return [
        # Select windows around picks to reduce the amount of noise traces in
        # training.
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
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    ]

  def get_val_augmentations(self) -> List:
    return self.get_train_augmentations()
