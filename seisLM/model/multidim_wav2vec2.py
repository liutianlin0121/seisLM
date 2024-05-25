"""Wav2Vec2 model configuration."""
import transformers.models.wav2vec2.modeling_wav2vec2 as hf_wav2vec2
from transformers import activations
from torch import nn
import einops


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
  def __init__(self, config, layer_id=0):
    super().__init__()

    if layer_id > 0:
      self.in_conv_dim = config.conv_dim[layer_id - 1]
    else:
      if hasattr(config, 'input_dim'):
        self.in_conv_dim = config.input_dim
      else:
        self.in_conv_dim = 1

    self.out_conv_dim = config.conv_dim[layer_id]

    self.conv = nn.Conv1d(
        self.in_conv_dim,
        self.out_conv_dim,
        kernel_size=config.conv_kernel[layer_id],
        stride=config.conv_stride[layer_id],
        bias=config.conv_bias,
    )
    self.activation = activations.ACT2FN[config.feat_extract_activation]

  def forward(self, hidden_states):
    hidden_states = self.conv(hidden_states)
    hidden_states = self.activation(hidden_states)
    return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):
  def __init__(self, config, layer_id=0):
    super().__init__()
    if layer_id > 0:
      self.in_conv_dim = config.conv_dim[layer_id - 1]
    else:
      if hasattr(config, 'input_dim'):
        self.in_conv_dim = config.input_dim
      else:
        self.in_conv_dim = 1


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
    self.activation = activations.ACT2FN[config.feat_extract_activation]

  def forward(self, hidden_states):
    hidden_states = self.conv(hidden_states)

    hidden_states = hidden_states.transpose(-2, -1)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = hidden_states.transpose(-2, -1)

    hidden_states = self.activation(hidden_states)
    return hidden_states


class Wav2Vec2GroupNormConvLayer(nn.Module):
  def __init__(self, config, layer_id=0):
    super().__init__()
    if layer_id > 0:
      self.in_conv_dim = config.conv_dim[layer_id - 1]
    else:
      if hasattr(config, 'input_dim'):
        self.in_conv_dim = config.input_dim
      else:
        self.in_conv_dim = 1

    self.out_conv_dim = config.conv_dim[layer_id]

    self.conv = nn.Conv1d(
        self.in_conv_dim,
        self.out_conv_dim,
        kernel_size=config.conv_kernel[layer_id],
        stride=config.conv_stride[layer_id],
        bias=config.conv_bias,
    )
    self.activation = activations.ACT2FN[config.feat_extract_activation]

    self.layer_norm = nn.GroupNorm(
      num_groups=self.out_conv_dim,
      num_channels=self.out_conv_dim,
      affine=True
    )

  def forward(self, hidden_states):
    hidden_states = self.conv(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.activation(hidden_states)
    return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
  """Construct the features from raw audio waveform"""

  def __init__(self, config):
    super().__init__()

    if config.feat_extract_norm == "group":
      conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
          Wav2Vec2NoLayerNormConvLayer(
            config, layer_id=i + 1
          ) for i in range(config.num_feat_extract_layers - 1)
      ]
    elif config.feat_extract_norm == "layer":
      conv_layers = [
          Wav2Vec2LayerNormConvLayer(
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

  def _freeze_parameters(self):
    for param in self.parameters():
      param.requires_grad = False
    self._requires_grad = False

  def forward(self, input_values):
    if input_values.dim() == 2:
      # hidden_states = input_values[:, None]
      hidden_states = einops.rearrange(input_values, 'b t -> b 1 t')
    else:
      assert input_values.dim() == 3
      hidden_states = input_values

    # make sure hidden_states require grad for gradient_checkpointing
    if self._requires_grad and self.training:
      hidden_states.requires_grad = True

    for conv_layer in self.conv_layers:
      if self._requires_grad and self.gradient_checkpointing and self.training:
        hidden_states = self._gradient_checkpointing_func(
            conv_layer.__call__,
            hidden_states,
        )
      else:
        hidden_states = conv_layer(hidden_states)

    return hidden_states


class MultiDimWav2Vec2Model(hf_wav2vec2.Wav2Vec2Model):
  """ Wav2Vec2 model."""
  def __init__(self, config: hf_wav2vec2.Wav2Vec2Config):
    super().__init__(config)
    self.config = config
    self.feature_extractor = Wav2Vec2FeatureEncoder(config)


class MultiDimWav2Vec2ForPreTraining(hf_wav2vec2.Wav2Vec2ForPreTraining):
  """ Wav2Vec2 model with a contrastive loss head."""
  def __init__(self, config: hf_wav2vec2.Wav2Vec2Config):
    super().__init__(config)
    self.wav2vec2 = MultiDimWav2Vec2Model(config)
