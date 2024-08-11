"""
This file contains the specifications for models used for phase-picking tasks.

Münchmeyer, J., Woollam, J., Rietbrock, A., Tilmann, F., Lange,
D., Bornstein, T., et al. (2022).
Which picker fits my data? A quantitative evaluation of deep learning based
seismic pickers. Journal of Geophysical Research: Solid Earth, 127.
https://doi.org/10.1029/2021JB023499

Modified from:
https://github.com/seisbench/pick-benchmark/blob/main/benchmark/models.py
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict
import math

import ml_collections
import einops
import lightning as L
import numpy as np
import seisbench.generate as sbg
import seisbench.models as sbm
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from seisLM.model.foundation import initialization
from seisLM.model.foundation import pretrained_models
from seisLM.model.task_specific.shared_task_specific import (
  BaseMultiDimWav2Vec2ForDownstreamTasks)
from seisLM.utils.data_utils import phase_dict


def vector_cross_entropy(
  y_pred: Tensor, y_true: Tensor, eps: float = 1e-5
  ) -> Tensor:
  """
  Cross entropy loss

  :param y_true [batch_size, pick_dim, seq_length]:
    True label probabilities
  :param y_pred [batch_size, pick_dim, seq_length]:
    Predicted label probabilities
  :param eps: Epsilon to clip values for stability
  :return: Average loss across batch
  """
  h = y_true * torch.log(y_pred + eps)
  if y_pred.ndim == 3:
    # Mean along sample dimension and sum along pick dimension
    h = h.mean(-1).sum(-1)
  else:
    h = h.sum(-1)  # Sum along pick dimension
  h = h.mean()  # Mean over batch axis
  return -h


class SeisBenchModuleLit(L.LightningModule, ABC):
  """
  Abstract interface for SeisBench lightning modules.
  Adds generic function, e.g., get_augmentations
  """

  @abstractmethod
  def get_augmentations(self) -> Any:
    """
    Returns a list of augmentations that can be passed to the
    seisbench.generate.GenericGenerator

    :return: List of augmentations
    """

  def get_train_augmentations(self) -> Any:
    """
    Returns the set of training augmentations.
    """
    return self.get_augmentations()

  def get_val_augmentations(self) -> Any:
    """
    Returns the set of validation augmentations for validations during training.
    """
    return self.get_augmentations()

  @abstractmethod
  def get_eval_augmentations(self) -> Any:
    """
    Returns the set of evaluation augmentations for evaluation after training.
    These augmentations will be passed to a SteeredGenerator and should usually
    contain a steered window.
    """

  @abstractmethod
  def predict_step(
    self,
    batch: Any,
    batch_idx: Optional[int] = None,
    dataloader_idx: Optional[int] = None
    ) -> Tuple:
    """
    Predict step for the lightning module. Returns results for three tasks:

    - earthquake detection (score, higher means more likely detection)
    - P to S phase discrimination (score, high means P, low means S)
    - phase location in samples (two integers, first for P, second for S wave)

    All predictions should only take the window defined
    by batch["window_borders"] into account.

    :param batch:
    :return:
    """
    score_detection = None
    score_p_or_s = None
    p_sample = None
    s_sample = None
    return score_detection, score_p_or_s, p_sample, s_sample


class BasePhaseNetLikeLit(SeisBenchModuleLit):
  """
  LightningModule for PhaseNet-like models
  """
  def __init__(
    self,
    model_config: ml_collections.ConfigDict,
    training_config: ml_collections.ConfigDict,
  ):

    super().__init__()
    self.save_hyperparameters()
    self.model_config = model_config
    self.training_config = training_config
    self.loss = vector_cross_entropy
    self.model = nn.Identity() # dummy model

  def forward(self, x: Tensor) -> Any:
    return self.model(x)

  def shared_step(self, batch: Dict) -> Tensor:
    x = batch["X"]
    y_true = batch["y"]
    y_pred = self.model(x)
    return self.loss(y_pred, y_true)

  def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
    loss = self.shared_step(batch)
    self.log("train/loss", loss, sync_dist=True, prog_bar=True, on_step=True)
    return loss

  def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    loss = self.shared_step(batch)
    self.log("val/loss", loss, sync_dist=True, prog_bar=True)
    return loss

  def configure_optimizers(self): # type: ignore
    optimizer = torch.optim.Adam(
      self.parameters(),
      **self.training_config.optimizer_args
    )
    return optimizer

  def get_augmentations(self): # type: ignore
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
            low=self.model_config.sample_boundaries[0],
            high=self.model_config.sample_boundaries[1],
            windowlen=3001,
            strategy="pad",
        ),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict,
            sigma=self.model_config.sigma,
            dim=0
        ),
    ]

  def get_eval_augmentations(self): # type: ignore
    return [
        sbg.SteeredWindow(windowlen=3001, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    ]

  def predict_step(
    self,
    batch: Dict,
    batch_idx: Optional[int] = None,
    dataloader_idx: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    x = batch["X"]
    window_borders = batch["window_borders"]

    pred = self.model(x)

    score_detection = torch.zeros(pred.shape[0])
    score_p_or_s = torch.zeros(pred.shape[0])
    p_sample = torch.zeros(pred.shape[0], dtype=int) # type: ignore
    s_sample = torch.zeros(pred.shape[0], dtype=int) # type: ignore

    for i in range(pred.shape[0]):
      start_sample, end_sample = window_borders[i]
      local_pred = pred[i, :, start_sample:end_sample]

      score_detection[i] = torch.max(1 - local_pred[-1])  # 1 - noise
      score_p_or_s[i] = torch.max(local_pred[0]) / torch.max(
          local_pred[1]
      )  # most likely P by most likely S

      p_sample[i] = torch.argmax(local_pred[0])
      s_sample[i] = torch.argmax(local_pred[1])

    return score_detection, score_p_or_s, p_sample, s_sample


class PhaseNetLit(BasePhaseNetLikeLit):
  """
  LightningModule for PhaseNet
  """
  def __init__(
    self,
    model_config: ml_collections.ConfigDict,
    training_config: ml_collections.ConfigDict,
  ):

    super().__init__(model_config, training_config)
    self.save_hyperparameters()
    self.model = sbm.PhaseNet(**model_config.kwargs)


class MultiDimWav2Vec2ForFrameClassification(
  BaseMultiDimWav2Vec2ForDownstreamTasks):
  """ Wav2Vec2 model with a contrastive loss head."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.classifier = nn.Linear(
      config.hidden_size + config.input_dim,
      config.num_labels
    )
    self.num_labels = config.num_labels
    self.apply(
      lambda module: initialization.init_wav2vec2_weights(
        config=config, module=module)
    )

  def forward(
      self,
      input_values: Optional[Tensor],
  ) -> Tensor:
    """The forward pass of the frame classification model."""

    hidden_states = self.get_wav2vec2_hidden_states(input_values)
    input_seq_length = input_values.shape[-1]

    # If seq_length of hidden_states and labels are not the same, we need to
    # interpolate the hidden_states to match the labels.
    if (hidden_states.shape[1] != input_seq_length):
      # change to [batch_size, hidden_size, seq_len]
      hidden_states = einops.rearrange(hidden_states, 'b l d -> b d l')
      hidden_states = torch.nn.functional.interpolate(
        hidden_states, size=input_seq_length,
        mode='linear', align_corners=False
      )
      hidden_states = einops.rearrange(hidden_states, 'b d l -> b l d')

    # Concatenate the hidden_states with the input_values

    hidden_states = torch.cat(
      [hidden_states,
       einops.rearrange(input_values, 'b d l -> b l d')], dim=-1)

    # logits: [batch_size, seq_len, num_classes]
    logits = self.classifier(hidden_states)

    # logits: [batch_size, num_classes, seq_len]
    logits = einops.rearrange(logits, 'b l c -> b c l')

    # softmax over the classes
    return torch.nn.functional.softmax(logits, dim=1)


class MultiDimWav2Vec2ForFrameClassificationLit(BasePhaseNetLikeLit):
  """
  LightningModule for MultiDimWav2Vec2ForFrameClassification

  """

  def __init__(
    self,
    model_config: ml_collections.ConfigDict,
    training_config: ml_collections.ConfigDict
    ):
    super().__init__(model_config, training_config)
    self.save_hyperparameters()

    pretrained_model = (
        pretrained_models.LitMultiDimWav2Vec2.load_from_checkpoint(
          model_config.pretrained_ckpt_path
      ).model
    )

    new_config = pretrained_model.config
    for key, value in model_config.items():
      setattr(new_config, key, value)

    model_config = new_config
    self.model = MultiDimWav2Vec2ForFrameClassification(model_config)

    if (not model_config.apply_spec_augment) or (
      model_config.mask_time_prob == 0.0
    ):
      # in this case, we don't need the masked spec embed
      # so we can remove it from both models.
      if hasattr(pretrained_model.wav2vec2, "masked_spec_embed"):
        del pretrained_model.wav2vec2.masked_spec_embed

      if hasattr(self.model.wav2vec2, "masked_spec_embed"):
        del self.model.wav2vec2.masked_spec_embed

    self.model.wav2vec2.load_state_dict(
        pretrained_model.wav2vec2.state_dict()
    )

    if model_config.freeze_feature_encoder:
      self.model.freeze_feature_encoder()

    if model_config.freeze_base_model:
      self.model.freeze_base_model()

    if model_config.freeze_base_model and (
      not model_config.freeze_feature_encoder):
      raise ValueError(
        "It's unconventional to freeze the base model" \
        "without freezing the feature encoder.")

    del pretrained_model
    self.model_config = model_config

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
