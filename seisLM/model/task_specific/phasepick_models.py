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
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import logging

import einops
import lightning as L
import ml_collections
import numpy as np
import seisbench.generate as sbg
import seisbench.models as sbm
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from seisLM.data_pipeline.augmentations import DuplicateEvent
from seisLM.model.foundation import initialization, pretrained_models
from seisLM.model.task_specific.shared_task_specific import (
    BaseMultiDimWav2Vec2ForDownstreamTasks, DoubleConvBlock)
from seisLM.utils.data_utils import phase_dict
from seisLM.data_pipeline.augmentations import FillMissingComponents



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
    self.hidden_dropout = nn.Dropout(config.head_dropout_rate)
    self.double_conv = DoubleConvBlock(
      in_channels=config.hidden_size + config.input_dim,
      out_channels=config.hidden_size + config.input_dim,
      kernel_size=3,
      dropout_rate=config.head_dropout_rate,
      padding='same',
      strides=[1, 1]
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


    hidden_states = einops.rearrange(hidden_states, 'b l d -> b d l')
    hidden_states = self.double_conv(hidden_states)
    hidden_states = einops.rearrange(hidden_states, 'b d l -> b l d')

    hidden_states = self.hidden_dropout(hidden_states)

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
    training_config: ml_collections.ConfigDict,
    load_pretrained: bool = True,
    ):
    super().__init__(model_config, training_config)

    if load_pretrained:
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

      if model_config.get("initialize_from_pretrained_weights", True):
        self.model.wav2vec2.load_state_dict(
            pretrained_model.wav2vec2.state_dict()
        )
      else:
        logging.warning("Skipping loading weights from pretrained model." +\
          "Use randomly initialized weights instead.")

      del pretrained_model
      self.model_config = model_config
    else:
      self.model = MultiDimWav2Vec2ForFrameClassification(model_config)

      if (not model_config.apply_spec_augment) or (
        model_config.mask_time_prob == 0.0
      ):
        # Remove masked_spec_embed from the instantiated models.
        if hasattr(self.model.wav2vec2, "masked_spec_embed"):
          del self.model.wav2vec2.masked_spec_embed


    # We save the hyperparameter after the model is instantiated.
    # This is because the model_config could get updated after loading the
    # pretrained model.
    self.save_hyperparameters()

    if model_config.freeze_feature_encoder:
      self.model.freeze_feature_encoder()

    if model_config.freeze_base_model:
      self.model.freeze_base_model()

    if model_config.freeze_base_model and (
      not model_config.freeze_feature_encoder):
      raise ValueError(
        "It's unconventional to freeze the base model" \
        "without freezing the feature encoder.")


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
        FillMissingComponents(),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
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
        FillMissingComponents(),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
    ]


class EQTransformerLit(SeisBenchModuleLit):
  """
  LightningModule for EQTransformer

  :param lr: Learning rate, defaults to 1e-2
  :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
  :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
  :param loss_weights: Loss weights for detection, P and S phase.
  :param rotate_array: If true, rotate array along sample axis.
  :param detection_fixed_window: Passed as parameter fixed_window to detection
  :param kwargs: Kwargs are passed to the SeisBench.models.EQTransformer constructor.
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
    self.loss = torch.nn.BCELoss()
    self.model = sbm.EQTransformer(**model_config.kwargs)

  def forward(self, x: Tensor) -> Any:
    return self.model(x)

  def shared_step(self, batch: Dict) -> Tensor:
    x = batch["X"]
    p_true = batch["y"][:, 0]
    s_true = batch["y"][:, 1]
    det_true = batch["detections"][:, 0]
    det_pred, p_pred, s_pred = self.model(x)

    return (
        self.training_config.loss_weights[0] * self.loss(det_pred, det_true)
        + self.training_config.loss_weights[1] * self.loss(p_pred, p_true)
        + self.training_config.loss_weights[2] * self.loss(s_pred, s_true)
    )

  def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
    loss = self.shared_step(batch)
    self.log("train/loss", loss, sync_dist=True)
    return loss

  def validation_step(self, batch: Dict, batch_idx) -> Tensor:
    loss = self.shared_step(batch)
    self.log("val/loss", loss, sync_dist=True)
    return loss

  def configure_optimizers(self) -> Any:
    optimizer = torch.optim.Adam(
      self.parameters(),
      **self.training_config.optimizer_args
    )
    return optimizer

  def get_joint_augmentations(self) -> Tuple:
    p_phases = [key for key, val in phase_dict.items() if val == "P"]
    s_phases = [key for key, val in phase_dict.items() if val == "S"]

    if self.training_config.detection_fixed_window is not None:
        detection_labeller = sbg.DetectionLabeller(
            p_phases,
            fixed_window=self.training_config.detection_fixed_window,
            key=("X", "detections"),
        )
    else:
        detection_labeller = sbg.DetectionLabeller(
            p_phases, s_phases=s_phases, key=("X", "detections")
        )

    block1 = [
        # In 2/3 of the cases, select windows around picks, to reduce amount
        # of noise traces in training.
        # Uses strategy variable, as padding will be handled by
        # the random window.
        # In 1/3 of the cases, just returns the original trace,
        # to keep diversity high.
        sbg.OneOf(
            [
                sbg.WindowAroundSample(
                    list(phase_dict.keys()),
                    samples_before=6000,
                    windowlen=12000,
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
            windowlen=6000,
            strategy="pad",
        ),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict, sigma=self.model_config.sigma, dim=0
        ),
        detection_labeller,
        # Normalize to ensure correct augmentation behavior
        sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    ]

    block2 = [
        sbg.ChangeDtype(np.float32, "X"),
        sbg.ChangeDtype(np.float32, "y"),
        sbg.ChangeDtype(np.float32, "detections"),
    ]

    return block1, block2

  def get_train_augmentations(self) -> Any:
    if self.training_config.rotate_array:
      rotation_block = [
          sbg.OneOf(
              [
                  sbg.RandomArrayRotation(["X", "y", "detections"]),
                  sbg.NullAugmentation(),
              ],
              [0.99, 0.01],
          )
      ]
    else:
      rotation_block = []

    augmentation_block = [
        # Add secondary event
        sbg.OneOf(
            [DuplicateEvent(label_keys="y"), sbg.NullAugmentation()],
            probabilities=[0.3, 0.7],
        ),
        # Gaussian noise
        sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [0.5, 0.5]),
        # Array rotation
        *rotation_block,
        # Gaps
        sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [0.2, 0.8]),
        # Channel dropout
        sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [0.3, 0.7]),
        # Augmentations make second normalize necessary
        sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    ]

    block1, block2 = self.get_joint_augmentations()

    return block1 + augmentation_block + block2

  def get_val_augmentations(self) -> Any:
    block1, block2 = self.get_joint_augmentations()

    return block1 + block2

  def get_augmentations(self) -> Any:
    raise NotImplementedError("Use get_train/val_augmentations instead.")

  def get_eval_augmentations(self) -> Any:
    return [
        sbg.SteeredWindow(windowlen=6000, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    ]

  def predict_step(
    self,
    batch: Dict,
    batch_idx: int = None,
    dataloader_idx: int = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x = batch["X"]
    window_borders = batch["window_borders"]

    det_pred, p_pred, s_pred = self.model(x)

    score_detection = torch.zeros(det_pred.shape[0])
    score_p_or_s = torch.zeros(det_pred.shape[0])
    p_sample = torch.zeros(det_pred.shape[0], dtype=int)
    s_sample = torch.zeros(det_pred.shape[0], dtype=int)

    for i in range(det_pred.shape[0]):
      start_sample, end_sample = window_borders[i]
      local_det_pred = det_pred[i, start_sample:end_sample]
      local_p_pred = p_pred[i, start_sample:end_sample]
      local_s_pred = s_pred[i, start_sample:end_sample]

      score_detection[i] = torch.max(local_det_pred)
      max_local_s_pred = torch.max(local_s_pred)
      max_local_s_pred = torch.where(
        max_local_s_pred == 0, 1e-6, max_local_s_pred
      )
      score_p_or_s[i] = torch.max(local_p_pred) / max_local_s_pred

      p_sample[i] = torch.argmax(local_p_pred)
      s_sample[i] = torch.argmax(local_s_pred)

    return score_detection, score_p_or_s, p_sample, s_sample


class GPDLit(SeisBenchModuleLit):
  """
  LightningModule for GPD

  :param lr: Learning rate, defaults to 1e-3
  :param sigma: Standard deviation passed to the ProbabilisticPickLabeller. If not, uses determinisic labels,
                i.e., whether a pick is contained.
  :param highpass: If not None, cutoff frequency for highpass filter in Hz.
  :param kwargs: Kwargs are passed to the SeisBench.models.GPD constructor.
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


    self.model = sbm.GPD(**model_config.kwargs)
    if self.model_config.sigma is None:
      self.nllloss = torch.nn.NLLLoss()
      self.loss = self.nll_with_probabilities
    else:
      self.loss = vector_cross_entropy
    self.predict_stride = 5

  def nll_with_probabilities(
    self,
    y_pred: Tensor,
    y_true: Tensor
  ) -> Tensor:
    y_pred = torch.log(y_pred)
    return self.nllloss(y_pred, y_true)

  def forward(self, x: Tensor) -> Any:
    return self.model(x)

  def shared_step(self, batch: Dict) -> Tensor:
    x = batch["X"]
    y_true = batch["y"].squeeze()
    y_pred = self.model(x)
    return self.loss(y_pred, y_true)

  def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
    loss = self.shared_step(batch)
    self.log("train/loss", loss, sync_dist=True)
    return loss

  def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    loss = self.shared_step(batch)
    self.log("val/loss", loss, sync_dist=True)
    return loss

  def configure_optimizers(self) -> Any:
    optimizer = torch.optim.Adam(
      self.parameters(),
      **self.training_config.optimizer_args
    )
    return optimizer

  def get_augmentations(self) -> Any:
    filter = []
    if self.training_config.highpass is not None:
      filter = [sbg.Filter(1, self.training_config.highpass, "highpass")]

    if self.model_config.sigma is None:
      labeller = sbg.StandardLabeller(
          label_columns=phase_dict,
          on_overlap="fixed-relevance",
          low=100,
          high=-100,
      )
    else:
      labeller = sbg.ProbabilisticPointLabeller(
          label_columns=phase_dict, position=0.5, sigma=self.model_config.sigma
      )

    return (
        [
          # In 2/3 of the cases, select windows around picks,
          # to reduce amount of noise traces in training.
          # Uses strategy variable, as padding will be handled
          # by the random window.
          # In 1/3 of the cases, just returns the original trace,
          # to keep diversity high.
          sbg.OneOf(
              [
                  sbg.WindowAroundSample(
                      list(phase_dict.keys()),
                      samples_before=400,
                      windowlen=800,
                      selection="random",
                      strategy="variable",
                  ),
                  sbg.NullAugmentation(),
              ],
              probabilities=[2, 1],
          ),
          sbg.RandomWindow(
              windowlen=400,
              strategy="pad",
          ),
          sbg.Normalize(
            detrend_axis=-1,
            amp_norm_axis=-1,
            amp_norm_type="peak"
          ),
          labeller,
        ]
        + filter
        + [sbg.ChangeDtype(np.float32)]
    )

  def get_eval_augmentations(self):
    filter = []
    if self.training_config.highpass is not None:
        filter = [sbg.Filter(1, self.training_config.highpass, "highpass")]

    return [
        # Larger window length ensures a sliding
        # window covering full trace can be applied
        sbg.SteeredWindow(windowlen=3400, strategy="pad"),
        sbg.SlidingWindow(timestep=self.predict_stride, windowlen=400),
        sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        *filter,
        sbg.ChangeDtype(np.float32),
    ]

  def predict_step(
    self,
    batch: Dict,
    batch_idx: Optional[int] = None,
    dataloader_idx: Optional[int] = None
    ):
    x = batch["X"]
    window_borders = batch["window_borders"]

    shape_save = x.shape
    x = x.reshape(
        (-1,) + shape_save[2:]
    )  # Merge batch and sliding window dimensions
    pred = self.model(x)
    pred = pred.reshape(shape_save[:2] + (-1,))
    pred = torch.repeat_interleave(
        pred, self.predict_stride, dim=1
    )  # Counteract stride
    pred = F.pad(pred, (0, 0, 200, 200))
    pred = pred.permute(0, 2, 1)

    # Otherwise windows shorter 30 s will automatically produce detections
    pred[:, 2, :200] = 1
    pred[:, 2, -200:] = 1

    score_detection = torch.zeros(pred.shape[0])
    score_p_or_s = torch.zeros(pred.shape[0])
    p_sample = torch.zeros(pred.shape[0], dtype=int)
    s_sample = torch.zeros(pred.shape[0], dtype=int)

    for i in range(pred.shape[0]):
      start_sample, end_sample = window_borders[i]
      local_pred = pred[i, :, start_sample:end_sample]

      score_detection[i] = torch.max(1 - local_pred[-1])  # 1 - noise
      score_p_or_s[i] = torch.max(local_pred[0]) / torch.max(
          local_pred[1]
      )  # most likely P by most likely S

      # Adjust for prediction stride by choosing the sample in
      # the middle of each block
      p_sample[i] = torch.argmax(local_pred[0]) + self.predict_stride // 2
      s_sample[i] = torch.argmax(local_pred[1]) + self.predict_stride // 2

    return score_detection, score_p_or_s, p_sample, s_sample
