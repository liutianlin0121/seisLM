"""
This file contains the specifications for models used for phase-picking tasks.

Münchmeyer, J., Woollam, J., Rietbrock, A., Tilmann, F., Lange,
D., Bornstein, T., et al. (2022).
Which picker fits my data? A quantitative evaluation of deep learning based
seismic pickers. Journal of Geophysical Research: Solid Earth, 127.
https://doi.org/10.1029/2021JB023499

Taken from:
https://github.com/seisbench/pick-benchmark/blob/main/benchmark/models.py
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import einops
import lightning as L
import numpy as np
import seisbench.generate as sbg
import seisbench.models as sbm
import torch
import torch.nn as nn
import transformers.models.wav2vec2.modeling_wav2vec2 as hf_wav2vec2
from transformers.modeling_outputs import TokenClassifierOutput

from seisLM.model.foundation import pretrained_models
from seisLM.model.foundation.multidim_wav2vec2 import MultiDimWav2Vec2Model
from seisLM.utils.data_utils import phase_dict


def vector_cross_entropy(y_pred, y_true, eps=1e-5):
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
  def get_augmentations(self):
    """
    Returns a list of augmentations that can be passed to the
    seisbench.generate.GenericGenerator

    :return: List of augmentations
    """

  def get_train_augmentations(self):
    """
    Returns the set of training augmentations.
    """
    return self.get_augmentations()

  def get_val_augmentations(self):
    """
    Returns the set of validation augmentations for validations during training.
    """
    return self.get_augmentations()

  @abstractmethod
  def get_eval_augmentations(self):
    """
    Returns the set of evaluation augmentations for evaluation after training.
    These augmentations will be passed to a SteeredGenerator and should usually
    contain a steered window.
    """

  @abstractmethod
  def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
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


class PhaseNetLit(SeisBenchModuleLit):
  """
  LightningModule for PhaseNet

  :param lr: Learning rate, defaults to 1e-2
  :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
  :param sample_boundaries: Low and high boundaries for the
      RandomWindow selection.
  :param kwargs: Kwargs are passed to the SeisBench.models.PhaseNet constructor.
  """

  def __init__(self, lr=1e-2, sigma=20, sample_boundaries=(None, None),
               **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.lr = lr
    self.sigma = sigma
    self.sample_boundaries = sample_boundaries
    self.loss = vector_cross_entropy
    self.model = sbm.PhaseNet(**kwargs)

  def forward(self, x):
    return self.model(x)

  def shared_step(self, batch):
    x = batch["X"]
    y_true = batch["y"]
    y_pred = self.model(x)
    return self.loss(y_pred, y_true)

  def training_step(self, batch, batch_idx):
    loss = self.shared_step(batch)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    loss = self.shared_step(batch)
    self.log("val_loss", loss, sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

  def get_augmentations(self):
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
            low=self.sample_boundaries[0],
            high=self.sample_boundaries[1],
            windowlen=3001,
            strategy="pad",
        ),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict, sigma=self.sigma, dim=0
        ),
    ]

  def get_eval_augmentations(self):
    return [
        sbg.SteeredWindow(windowlen=3001, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    ]

  def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
    x = batch["X"]
    window_borders = batch["window_borders"]

    pred = self.model(x)

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

      p_sample[i] = torch.argmax(local_pred[0])
      s_sample[i] = torch.argmax(local_pred[1])

    return score_detection, score_p_or_s, p_sample, s_sample


_HIDDEN_STATES_START_POSITION = 2

class MultiDimWav2Vec2ForFrameClassification(
  hf_wav2vec2.Wav2Vec2ForAudioFrameClassification):
  """ Wav2Vec2 model with a contrastive loss head."""

  def __init__(self, config: hf_wav2vec2.Wav2Vec2Config):
    super().__init__(config)
    self.wav2vec2 = MultiDimWav2Vec2Model(config)
    self.classifier = nn.Linear(
      config.hidden_size + config.input_dim,
      config.num_labels
    )

  def forward(
      self,
      input_values: Optional[torch.Tensor],
  ) -> Union[Tuple, TokenClassifierOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, target_length, num_labels)`,
        *optional*):
        Onehot labels for computing the frame classification loss.
    """

    # input_values: [batch_size, num_channels, seq_len]
    input_seq_length = input_values.shape[-1]

    output_hidden_states = (
      True if self.config.use_weighted_layer_sum else False
    )

    outputs = self.wav2vec2(
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=output_hidden_states,
        return_dict=self.config.use_return_dict,
    )


    # The resulting hidden_states: [batch_size, seq_len, hidden_size]
    if self.config.use_weighted_layer_sum:
      hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
      hidden_states = torch.stack(hidden_states, dim=1)
      norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
      hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
    else:
      hidden_states = outputs[0]

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


class MultiDimWav2Vec2ForFrameClassificationLit(PhaseNetLit):
  """
  LightningModule for MultiDimWav2Vec2ForFrameClassification

  Attributes:
   model_name_or_path: pretrained model name or path, from which we load the
    checkpoint
   lr: Learning rate, defaults to 1e-2
   sigma: Standard deviation passed to the ProbabilisticPickLabeller
   sample_boundaries: Low and high boundaries for the RandomWindow selection.
   num_labels: Number of labels for the classification task.
  """

  def __init__(self, model_name_or_path, lr=1e-2, sigma=20,
              sample_boundaries=(None, None), num_labels=3, **kwargs):
    super().__init__(
        lr=lr, sigma=sigma, sample_boundaries=sample_boundaries, **kwargs
    )
    self.save_hyperparameters()
    self.model_name_or_path = model_name_or_path
    pretrained_model = pretrained_models.LitMultiDimWav2Vec2.load_from_checkpoint(
        model_name_or_path
    ).model

    model_config = pretrained_model.config
    model_config.num_labels = num_labels

    self.model = MultiDimWav2Vec2ForFrameClassification(model_config)

    self.model.wav2vec2.load_state_dict(
        pretrained_model.wav2vec2.state_dict()
    )
    self.model.freeze_feature_extractor()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=self.lr
    )
    return optimizer
