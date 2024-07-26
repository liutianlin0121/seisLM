""" Models for the foreshock-aftershock classification task. """
import math
from typing import Dict, Tuple, Union, Any, Sequence

import lightning as L
import ml_collections
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.utilities import grad_norm
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from seisLM.model.foundation import pretrained_models
from seisLM.model.task_specific.shared_task_specific import (
  BaseMultiDimWav2Vec2ForDownstreamTasks)

from einops.layers.torch import Reduce, Rearrange


class DoubleConvBlock(nn.Module):
  """Two conv layers with batchnorm and ReLU activation, like in a 1d U-Net."""
  def __init__(
    self,
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dropout_rate: float
    ):
    super(DoubleConvBlock, self).__init__()

    conv_shared_kwargs = {
      'kernel_size': kernel_size,
      'out_channels': out_channels,
      'padding': 'valid',
      'bias': False, # Because batchnorm follows the conv layer.
    }

    self.double_conv = nn.Sequential(
      nn.Conv1d(in_channels=in_channels, stride=1, **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.GELU(),
      nn.Dropout(dropout_rate),
      nn.Conv1d(in_channels=out_channels, stride=2, **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.GELU(),
      nn.Dropout(dropout_rate),
    )

  def forward(self, x: Tensor) -> Tensor:
    x = self.double_conv(x)
    return x


class Conv1DShockClassifier(nn.Module):
  """A simple 1D conv classifier for foreshock-aftershock classification."""
  def __init__(
    self,
    config: ml_collections.ConfigDict
  ):
    super(Conv1DShockClassifier, self).__init__()
    self.config = config

    layers = []
    in_channels = config.in_channels
    for i in range(config.num_layers):
      out_channels = config.initial_filters * (2 ** i)
      layers.append(
        DoubleConvBlock(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=config.kernel_size,
          dropout_rate=config.dropout_rate
        )
      )
      in_channels = out_channels

    self.conv_encoder = nn.Sequential(*layers)
    self.global_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(out_channels, config.num_classes)

  def forward(self, x: Tensor) -> Tensor:
    x = self.conv_encoder(x)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


class Wav2Vec2ForSequenceClassification(BaseMultiDimWav2Vec2ForDownstreamTasks):
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)

    self.conv_head = nn.Sequential(
      Rearrange('b l c -> b c l'),
      DoubleConvBlock(
        in_channels=config.hidden_size,
        out_channels=config.hidden_size,
        kernel_size=3,
        dropout_rate=0.2
      ),
      DoubleConvBlock(
        in_channels=config.hidden_size,
        out_channels=config.classifier_proj_size,
        kernel_size=3,
        dropout_rate=0.2
      ),
      Reduce('b c l -> b c', reduction='mean'),
      nn.Linear(config.classifier_proj_size, config.num_classes)
    )

  def forward(self, input_values: torch.Tensor,) -> Tensor:
    """The forward pass of the sequence classification model.

    Args:
      input_values: The input waveforms.

    Returns:
      logits: The classification logits.
    """
    hidden_states = self.get_wav2vec2_hidden_states(input_values)
    logits = self.conv_head(hidden_states)
    return logits



class ShockClassifierLit(L.LightningModule):
  """ A LightningModule for the Conv1DShockClassifier model. """
  def __init__(
    self,
    model_name: str,
    model_config: ml_collections.ConfigDict,
    training_config: ml_collections.ConfigDict
    ):
    super().__init__()
    self.save_hyperparameters()
    self.training_config = training_config

    if model_name == "Conv1DShockClassifier":
      self.model = Conv1DShockClassifier(model_config)
    elif model_name == "Wav2Vec2ForSequenceClassification":
      pretrained_model = pretrained_models.LitMultiDimWav2Vec2.load_from_checkpoint(
          model_config.pretrained_ckpt_path
      ).model

      ## TODO: temp fix for the config issue
      new_config = pretrained_model.config
      for key, value in model_config.items():
        setattr(new_config, key, value)

      # for key, value in new_config.to_dict().items():
      #   if 'dropout' in key:
      #     new_value = 0.1
      #     print(f'Orig. {key} value: {value}. New value: {new_value}')
      #     setattr(new_config, key, new_value)

      model_config = new_config
      # model_config = config_utils.ConfigTracker(model_config)
      self.model = Wav2Vec2ForSequenceClassification(model_config)

      self.model.wav2vec2.load_state_dict(
          pretrained_model.wav2vec2.state_dict()
      )
      self.model.freeze_feature_encoder()
      # self.model.freeze_base_model()
      del pretrained_model
    else:
      raise ValueError(f"Model {model_name} not recognized.")

    self.model_config = model_config
    self.train_acc = torchmetrics.Accuracy(
      task="multiclass", num_classes=model_config.num_classes
    )
    self.val_acc = torchmetrics.Accuracy(
      task="multiclass", num_classes=model_config.num_classes
    )
    self.test_acc = torchmetrics.Accuracy(
      task="multiclass", num_classes=model_config.num_classes
    )

  def forward(self, waveforms: Tensor) -> Tensor:
    logits = self.model(waveforms)
    return logits


  def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    # inspect (unscaled) gradients here
    self.log_dict(grad_norm(self, norm_type=2))

  def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
    waveforms, labels = batch
    logits = self(waveforms)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predicted_labels = torch.argmax(logits, 1)
    self.train_acc(predicted_labels, labels)

    self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True)
    self.log("train/acc", self.train_acc, sync_dist=True, prog_bar=True)

    return loss  # this is passed to the optimizer for training

  def validation_step(self, batch: Tuple, batch_idx: int) -> None:

    waveforms, labels = batch

    logits = self(waveforms)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predicted_labels = torch.argmax(logits, 1)
    self.val_acc(predicted_labels, labels)
    self.log("val/loss", loss, sync_dist=True, prog_bar=True)
    self.log("val/acc", self.val_acc, sync_dist=True, prog_bar=True)


  def test_step(self, batch: Tuple, batch_idx: int) -> None:
    waveforms, labels = batch
    logits = self(waveforms)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predicted_labels = torch.argmax(logits, 1)
    self.test_acc(predicted_labels, labels)
    self.log("test/loss", loss, sync_dist=True, prog_bar=True)
    self.log("test/acc", self.test_acc, sync_dist=True, prog_bar=True)

  def configure_optimizers(self): # type: ignore

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=self.model_config.learning_rate,
        weight_decay=self.model_config.weight_decay,
    )

    t_max = int(
      self.training_config.max_train_steps // self.trainer.num_devices
    )
    t_warmup = int((self.training_config.warmup_frac_step * (
      self.training_config.max_train_steps)) // self.trainer.num_devices
    )

    # Linear warmup and half-cycle cosine decay
    def lr_lambda(step: int): # type: ignore
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
