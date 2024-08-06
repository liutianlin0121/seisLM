""" Models for the foreshock-aftershock classification task. """
import copy
import math
from typing import Dict, Tuple, Union, Any, Sequence

import einops
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
from einops.layers.torch import Reduce, Rearrange
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

from seisLM.model.foundation import initialization

from seisLM.model.foundation import pretrained_models
from seisLM.model.task_specific.shared_task_specific import (
  BaseMultiDimWav2Vec2ForDownstreamTasks)



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


class MeanStdPooling(nn.Module):
  def __init__(self):
    super(MeanStdPooling, self).__init__()

  def forward(self, x):
    """
    Forward pass for mean and standard deviation pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channel_number, seq_length)
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channel_number * 2)
    """
    mean = x.mean(dim=2)  # Compute mean over seq_length
    std = x.std(dim=2)    # Compute std over seq_length
    pooled = torch.cat((mean, std), dim=1)  # Concatenate along channel dimension
    return pooled


# class AttentiveStatPool1D(nn.Module):
#     def __init__(self, embedding_size: int, dim_to_reduce: int = 2):
#         super().__init__()
#         self.pooling_layer = AttentiveStatisticsPooling(embedding_size)
#         self.dim_to_reduce = dim_to_reduce

#     def forward(self, tensor: Tensor):
#         if self.dim_to_reduce == 2:
#             pooled_embedding = self.pooling_layer(tensor)
#         elif self.dim_to_reduce == 1:
#             pooled_embedding = self.pooling_layer(tensor.transpose(1, 2))
#         else:
#             raise ValueError("can only pool dimension 1 or 2")

#         pooled_embedding = pooled_embedding.squeeze()

#         if len(pooled_embedding.shape) == 1:
#             pooled_embedding = pooled_embedding[None, :]

#         return pooled_embedding

class Wav2Vec2ForSequenceClassification(BaseMultiDimWav2Vec2ForDownstreamTasks):
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)


    self.head = nn.Sequential(
      nn.Dropout1d(0.4), # dropout entire timesteps
      Rearrange('b l c -> b c l'),
      nn.Dropout1d(0.4), # dropout entire channels
      MeanStdPooling(), # [b, c, l] -> [b, 2c]
      nn.Linear(
        2 * config.hidden_size, config.classifier_proj_size, bias=False
      ),
      nn.LayerNorm(config.classifier_proj_size),
      nn.GELU(),
      nn.Dropout(config.dropout_rate),
      nn.Linear(config.classifier_proj_size, config.num_classes)
    )

    # self.head = nn.Sequential(
    #   Rearrange('b l c -> b c l'),
    #   Reduce('b c l -> b c', reduction='mean'),
    #   # MeanStdPooling(),
    #   nn.Linear(config.hidden_size, config.classifier_proj_size, bias=False),
    #   torch.nn.BatchNorm1d(config.classifier_proj_size),
    #   nn.Tanh(),
    #   # nn.Dropout(config.dropout_rate),
    #   nn.Linear(config.classifier_proj_size, config.num_classes)
    # )

    # self.apply(
    #   lambda module: initialization.init_wav2vec2_weights(
    #     config=config, module=module)
    # )
    # conv_config = ml_collections.ConfigDict({
    #   'in_channels': 3, #config.hidden_size,
    #   'num_classes': config.num_classes,
    #   'num_layers': 3,
    #   'initial_filters': 32,
    #   'kernel_size': 3,
    #   'dropout_rate': config.dropout_rate,
    # })
    # self.skip_head = nn.Sequential(
    #   Conv1DShockClassifier(conv_config)
    # )


  def forward(self, input_values: torch.Tensor,) -> Tensor:
    """The forward pass of the sequence classification model.

    Args:
      input_values: The input waveforms.

    Returns:
      logits: The classification logits.
    """
    hidden_states = self.get_wav2vec2_hidden_states(input_values)
    logits = self.head(hidden_states)
    return logits



class BaseShockClassifierLit(L.LightningModule):
  """ A LightningModule for the Conv1DShockClassifier model. """
  def __init__(
    self,
    model_config: ml_collections.ConfigDict,
    training_config: ml_collections.ConfigDict
    ):
    super().__init__()
    self.save_hyperparameters()
    self.model_config = model_config
    self.training_config = training_config
    self.model = nn.Identity() # dummy model
    self.loss_fn = nn.CrossEntropyLoss(
      label_smoothing=training_config.get('label_smoothing', 0.0)
    )

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
    # loss = torch.nn.functional.cross_entropy(logits, labels)
    loss = self.loss_fn(logits, labels)
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


class Conv1DShockClassifierLit(BaseShockClassifierLit):
  """ A LightningModule for the Conv1DShockClassifier model. """
  def __init__(
    self,
    model_config: ml_collections.ConfigDict,
    training_config: ml_collections.ConfigDict
    ):
    super().__init__(model_config, training_config)
    self.save_hyperparameters()
    self.training_config = training_config

    self.model = Conv1DShockClassifier(model_config)

  def configure_optimizers(self): # type: ignore

    if self.training_config.optimizer == "adamw":
      optimizer = torch.optim.AdamW(
          filter(lambda p: p.requires_grad, self.parameters()),
          **self.training_config.optimizer_args
      )
    elif self.training_config.optimizer == "sgd":
      optimizer = torch.optim.SGD(
          filter(lambda p: p.requires_grad, self.parameters()),
          **self.training_config.optimizer_args
      )
    else:
      raise ValueError(
          f"Optimizer {self.training_config.optimizer} not recognized."
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



class Wav2vec2ShockClassifierLit(BaseShockClassifierLit):
  """ Wav2vec2 model for shock classification. """
  def __init__(
    self,
    model_config: ml_collections.ConfigDict,
    training_config: ml_collections.ConfigDict
    ):
    super().__init__(model_config, training_config)
    self.save_hyperparameters()
    self.training_config = training_config

    pretrained_model = pretrained_models.LitMultiDimWav2Vec2.load_from_checkpoint(
        model_config.pretrained_ckpt_path
    ).model

    ## TODO: temp fix for the config issue
    new_config = pretrained_model.config
    for key, value in model_config.items():
      setattr(new_config, key, value)

    # for key, value in new_config.to_dict().items():
    #   if ('dropout' in key) and ('attention' not in key) and ('quantizer' not in key):
    #     new_value = model_config.dropout_rate
    #     print(f'Orig. {key} value: {value}. New value: {new_value}')
    #     setattr(new_config, key, new_value)

    model_config = new_config
    self.model = Wav2Vec2ForSequenceClassification(model_config)
    self.pretrained_weights = copy.deepcopy(
      pretrained_model.wav2vec2.state_dict()
    )

    self.model.wav2vec2.load_state_dict(
        pretrained_model.wav2vec2.state_dict()
    )
    # self.model.freeze_feature_encoder()
    # self.model.freeze_base_model()
    del pretrained_model
    self.model_config = model_config


  def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
    waveforms, labels = batch
    logits = self(waveforms)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    predicted_labels = torch.argmax(logits, 1)
    self.train_acc(predicted_labels, labels)

    self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True)
    self.log("train/acc", self.train_acc, sync_dist=True, prog_bar=True)
    return loss  # this is passed to the optimizer for training

  def configure_optimizers(self): # type: ignore

    if self.training_config.optimizer == "adamw":
      optimizer = torch.optim.AdamW(
          filter(lambda p: p.requires_grad, self.parameters()),
          **self.training_config.optimizer_args
      )
    elif self.training_config.optimizer == "sgd":
      optimizer = torch.optim.SGD(
          filter(lambda p: p.requires_grad, self.parameters()),
          **self.training_config.optimizer_args
      )
    else:
      raise ValueError(
          f"Optimizer {self.training_config.optimizer} not recognized."
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
