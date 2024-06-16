
"""Wav2Vec2 model configuration."""
from typing import Optional, Tuple, Union
import einops
import torch
import transformers.models.wav2vec2.modeling_wav2vec2 as hf_wav2vec2
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from seisLM.model.feature_encoder import Wav2Vec2FeatureEncoder
from seisLM.model.quantizer import Wav2Vec2GumbelVectorQuantizer


class MultiDimWav2Vec2Model(hf_wav2vec2.Wav2Vec2Model):
  """ Wav2Vec2 model."""
  def __init__(self, config: hf_wav2vec2.Wav2Vec2Config):
    super().__init__(config)
    self.config = config
    self.feature_extractor = Wav2Vec2FeatureEncoder(config)
    # Initialize weights and apply final processing
    super().post_init()

class MultiDimWav2Vec2ForPreTraining(hf_wav2vec2.Wav2Vec2ForPreTraining):
  """ Wav2Vec2 model with a contrastive loss head."""
  def __init__(self, config: hf_wav2vec2.Wav2Vec2Config):
    super().__init__(config)
    self.wav2vec2 = MultiDimWav2Vec2Model(config)
    self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

    # Initialize weights and apply final processing
    self.custom_post_init()


  def custom_post_init(self):
    # Manually initialize quantizer. This is necessary, since we replace the
    # `Wav2Vec2GumbelVectorQuantizer` in
    # `transformers.models.wav2vec2.modeling_wav2vec2` by our customized
    # one; `_init_weights`` method in `Wav2Vec2PreTrainedModel` is thus not
    # applied to our customized quantizer. We need to do the init this manually
    # to make the initialization behavior consistent.
    # self.quantizer.weight_proj.weight.data.normal_(mean=0.0, std=1)
    # self.quantizer.weight_proj.bias.data.zero_()
    # nn.init.uniform_(self.quantizer.codevectors)
    super().post_init()
    self.quantizer.weight_proj.weight.data.normal_(mean=0.0, std=1)
    self.quantizer.weight_proj.bias.data.zero_()
    nn.init.uniform_(self.quantizer.codevectors)


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

