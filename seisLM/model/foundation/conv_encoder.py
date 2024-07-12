"""Feature encoders.

Dimension key:

B: batch size
L: sequence length
D: feature dimension
"""
import einops
import ml_collections
from torch import Tensor, nn


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
  """Convolutional layer with no layer normalization"""
  def __init__(self, config: ml_collections.ConfigDict, layer_id: int = 0):
    super().__init__()

    self.in_conv_dim = (
        config.conv_dim[layer_id - 1] if layer_id > 0 else getattr(
          config, 'input_dim', 1
        )
    )

    self.out_conv_dim = config.conv_dim[layer_id]

    self.conv = nn.Conv1d(
        self.in_conv_dim,
        self.out_conv_dim,
        kernel_size=config.conv_kernel[layer_id],
        stride=config.conv_stride[layer_id],
        bias=config.conv_bias,
    )
    self.activation = nn.functional.gelu

  def forward(self, hidden_states: Tensor) -> Tensor:

    # hidden_states: [B, D, L]
    hidden_states = self.conv(hidden_states)
    hidden_states = self.activation(hidden_states)
    return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):
  """Convolutional layer with layer normalization"""
  def __init__(self, config: ml_collections.ConfigDict, layer_id: int = 0):
    super().__init__()

    self.in_conv_dim = (
        config.conv_dim[layer_id - 1] if layer_id > 0 else getattr(
          config, 'input_dim', 1
        )
    )

    self.out_conv_dim = config.conv_dim[layer_id]

    self.conv = nn.Conv1d(
        self.in_conv_dim,
        self.out_conv_dim,
        kernel_size=config.conv_kernel[layer_id],
        stride=config.conv_stride[layer_id],
        bias=config.conv_bias,
    )
    self.layer_norm = nn.LayerNorm(
      self.out_conv_dim, elementwise_affine=True
    )
    self.activation = nn.functional.gelu

  def forward(self, hidden_states: Tensor) -> Tensor:
    # hidden_states: [B, D, L]

    hidden_states = self.conv(hidden_states)

    hidden_states = hidden_states.transpose(-2, -1)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = hidden_states.transpose(-2, -1)

    hidden_states = self.activation(hidden_states)
    return hidden_states


class Wav2Vec2GroupNormConvLayer(nn.Module):
  """Convolutional layer with group normalization"""
  def __init__(self, config: ml_collections.ConfigDict, layer_id: int = 0):
    super().__init__()

    self.in_conv_dim = (
        config.conv_dim[layer_id - 1] if layer_id > 0 else getattr(
          config, 'input_dim', 1
        )
    )

    self.out_conv_dim = config.conv_dim[layer_id]

    self.conv = nn.Conv1d(
        self.in_conv_dim,
        self.out_conv_dim,
        kernel_size=config.conv_kernel[layer_id],
        stride=config.conv_stride[layer_id],
        bias=config.conv_bias,
    )
    self.activation = nn.functional.gelu

    self.layer_norm = nn.GroupNorm(
      num_groups=self.out_conv_dim,
      num_channels=self.out_conv_dim,
      affine=True
    )

  def forward(self, hidden_states: Tensor,) -> Tensor:
    # hidden_states: [B, D, L]

    hidden_states = self.conv(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.activation(hidden_states)
    return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
  """Construct the features from raw audio waveform"""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()

    if config.feat_extract_norm == "group":
      conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
          Wav2Vec2NoLayerNormConvLayer(
            config, layer_id=i + 1
          ) for i in range(config.num_feat_extract_layers - 1)
      ]
    elif config.feat_extract_norm == "layer":
      conv_layers = [
          Wav2Vec2LayerNormConvLayer( # type: ignore[misc]
            config, layer_id=i) for i in range(config.num_feat_extract_layers)
      ]
    else:
      raise ValueError(
          f"`config.feat_extract_norm` is {config.feat_extract_norm}," + \
          "but has to be one of ['group', 'layer']"
      )
    self.conv_layers = nn.ModuleList(conv_layers)
    self.gradient_checkpointing = False
    self._requires_grad = True

  def _freeze_parameters(self) -> None:
    for param in self.parameters():
      param.requires_grad = False
    self._requires_grad = False

  def forward(self, input_values: Tensor) -> Tensor:
    # hidden_states: [B, D, L] or [B, 1, L]

    if input_values.dim() == 2:
      hidden_states = einops.rearrange(input_values, 'B L -> B 1 L')
    else:
      assert input_values.dim() == 3
      hidden_states = input_values

    # make sure hidden_states require grad for gradient_checkpointing
    if self._requires_grad and self.training:
      hidden_states.requires_grad = True

    for conv_layer in self.conv_layers:
      hidden_states = conv_layer(hidden_states)

    return hidden_states
