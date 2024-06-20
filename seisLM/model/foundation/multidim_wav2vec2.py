
"""Wav2Vec2 model configuration."""
import einops
import torch
import transformers.models.wav2vec2.modeling_wav2vec2 as hf_wav2vec2
from torch import nn
from seisLM.model.foundation.feature_encoder import Wav2Vec2FeatureEncoder
from seisLM.model.foundation.quantizer import Wav2Vec2GumbelVectorQuantizer


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



