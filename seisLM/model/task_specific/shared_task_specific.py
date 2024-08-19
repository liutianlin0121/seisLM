""" Shared classes and functions for task-specific models. """
from typing import Optional
import abc
import einops
import torch
from torch import nn
from torch import Tensor
import ml_collections
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch import LightningModule
from seisLM.model.foundation.multidim_wav2vec2 import Wav2Vec2Model




class DoubleConvBlock(nn.Module):
  """Two conv layers with batchnorm and ReLU activation, like in a 1d U-Net."""
  def __init__(
    self,
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dropout_rate: float,
    padding: str = 'valid',
    strides: list[int] = [1, 2]
    ):
    super(DoubleConvBlock, self).__init__()

    conv_shared_kwargs = {
      'kernel_size': kernel_size,
      'out_channels': out_channels,
      'padding': padding,
      'bias': False, # Because batchnorm follows the conv layer.
    }

    self.double_conv = nn.Sequential(
      nn.Conv1d(in_channels=in_channels, stride=strides[0], **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.GELU(),
      nn.Dropout(dropout_rate),
      nn.Conv1d(in_channels=out_channels, stride=strides[1], **conv_shared_kwargs),
      nn.BatchNorm1d(out_channels),
      nn.GELU(),
      nn.Dropout(dropout_rate),
    )

  def forward(self, x: Tensor) -> Tensor:
    x = self.double_conv(x)
    return x


class BaseModelUnfreeze(BaseFinetuning):
  """ A finetuning class that unfreezes the base model at a specific epoch."""
  def __init__(self, unfreeze_at_epoch: int = 10):
    super().__init__()
    self._unfreeze_at_epoch = unfreeze_at_epoch

  def freeze_before_training(self, pl_module: LightningModule) -> None:
    # freeze any module you want
    # Here, we are freezing `feature_extractor`
    self.freeze(pl_module.model.wav2vec2)

  def finetune_function(
    self,
    pl_module: LightningModule,
    current_epoch: int,
    optimizer: torch.optim.Optimizer,
    ) -> None:
    # When `current_epoch` is 10, feature_extractor will start training.
    if current_epoch == self._unfreeze_at_epoch:
      self.unfreeze_and_add_param_group(
          modules=pl_module.model.wav2vec2,
          optimizer=optimizer,
          train_bn=True,
      )

class BaseMultiDimWav2Vec2ForDownstreamTasks(nn.Module, abc.ABC):
  """Wav2Vec2 model with a task-specific head."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.config = config
    self.wav2vec2 = Wav2Vec2Model(config)

     # num layers are the transformer layers and the input embedding layer
    num_layers = config.num_hidden_layers + 1
    if config.use_weighted_layer_sum:
      self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
      self.lns = nn.ModuleList(
        [nn.LayerNorm(config.hidden_size) for _ in range(num_layers)]
      )

  def freeze_feature_encoder(self) -> None:
    """Disable the gradient computation for the feature encoder."""
    self.wav2vec2.feature_extractor._freeze_parameters() # pylint: disable=protected-access

  def freeze_base_model(self) -> None:
    """Disable the gradient computation for the base model."""
    for param in self.wav2vec2.parameters():
      param.requires_grad = False

  def get_wav2vec2_hidden_states(self,
      input_values: Optional[torch.Tensor],
  ) -> torch.Tensor:
    """The forward pass of the sequence classification model.

    Args:
      input_values: The input waveforms.

    Returns:
      the hidden states of the Wav2Vec2 model.
    """
    input_seq_length = input_values.shape[-1]
    output_hidden_states = True if self.config.use_weighted_layer_sum else False
    outputs = self.wav2vec2(
        input_values,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=output_hidden_states,
    )

    if self.config.use_weighted_layer_sum:
      hidden_states = outputs.hidden_states
      hidden_states = tuple(
        ln(hidden_states[i]) for i, ln in enumerate(self.lns)
      )

      # [B, num_layers, L, config.hidden_size]
      hidden_states = torch.stack(hidden_states, dim=1)
      norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)

      # [B, L, config.hidden_size]
      hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
    else:
      # [B, L, config.hidden_size]
      hidden_states = outputs.last_hidden_state

    return hidden_states



  @abc.abstractmethod
  def forward(self, input_values: torch.Tensor,) -> torch.Tensor:
    pass
