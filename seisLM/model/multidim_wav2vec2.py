"""Wav2Vec2 model configuration."""
from typing import Optional, Tuple, Union
import transformers.models.wav2vec2.modeling_wav2vec2 as hf_wav2vec2
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import activations
import torch
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
