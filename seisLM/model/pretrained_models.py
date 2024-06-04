"""Wav2Vec2 model."""
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
import seisbench.generate as sbg
from seisLM.model.multidim_wav2vec2 import MultiDimWav2Vec2ForPreTraining
from seisLM.utils.data_utils import phase_dict

class LitMultiDimWav2Vec2(L.LightningModule):
  """LightningModule for Wav2Vec2 model."""
  def __init__(self, model_config, training_config):
    super().__init__()
    self.training_config = training_config
    self.model_config = model_config
    self.model = MultiDimWav2Vec2ForPreTraining(model_config)
    self.save_hyperparameters()

  def training_step(self, batch, batch_idx):
    # pylint:disable=missing-function-docstring
    # pylint:disable=invalid-name

    mask_time_indices = batch["mask_time_indices"]
    num_losses = mask_time_indices.sum()
    percent_masked = mask_time_indices.float().mean()

    # forward
    outputs = self.model(**batch)
    loss = outputs.loss / num_losses

    temperature_max_min_gap = (
        self.training_config.max_gumbel_temperature -\
          self.training_config.min_gumbel_temperature
          )

    ratio_completed_steps = self.trainer.global_step / (
        self.training_config.max_train_steps // self.trainer.num_devices
    )
    temperature_factor = (1 + math.cos(math.pi * ratio_completed_steps))/2

    gumbel_temperature = \
        self.training_config.min_gumbel_temperature + (
          temperature_factor * temperature_max_min_gap
        )

    self.model.set_gumbel_temperature(gumbel_temperature)

    self.log("train/loss", loss, sync_dist=True, prog_bar=True)

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


  def validation_step(self, batch, batch_idx):
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

  def on_validation_epoch_end(self):
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

  def configure_optimizers(self, interval='step'):
    optimizer = torch.optim.AdamW(
        params=self.model.parameters(),
        lr=self.training_config.learning_rate,
        weight_decay=self.training_config.weight_decay,
        betas=(
          self.training_config.adam_beta1, self.training_config.adam_beta2
        ),
        eps=self.training_config.adam_epsilon,
    )

    t_max = int(
      self.training_config.max_train_steps // self.trainer.num_devices
    )
    t_warmup = int((self.training_config.warmup_frac_step * (
      self.training_config.max_train_steps)) // self.trainer.num_devices
    )

    # Linear warmup and half-cycle cosine decay
    def lr_lambda(step):
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

  def get_train_augmentations(self):
    return [
        # In 2/3 of the cases, select windows around picks, to reduce amount
        # of noise traces in training. Uses strategy variable, as padding will
        # be handled by the random window. In 1/3 of the cases, just returns
        # the original trace, to keep diversity high.
        sbg.OneOf(
            [
                sbg.WindowAroundSample(
                    list(phase_dict.keys()),
                    samples_before=3000,
                    windowlen=6000,
                    selection="random",
                    strategy="variable",
                ),
                sbg.NullAugmentation(),
            ],
            probabilities=[2, 1],
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

  def get_val_augmentations(self):
    return self.get_train_augmentations()
