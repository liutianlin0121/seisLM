""" Initialization functions for the Wav2Vec2 model. """
import math
from torch import nn
import ml_collections
from seisLM.model.foundation import transformer_encoder
from seisLM.model.foundation import quantizer
from seisLM.model.foundation import multidim_wav2vec2

def init_wav2vec2_weights(
  *,
  config: ml_collections.ConfigDict,
  module: nn.Module
  ) -> None:
  """Initialize the weights"""
  # Wav2Vec2ForPreTraining last 2 linear layers need standard Linear init.
  if isinstance(module, multidim_wav2vec2.MultiDimWav2Vec2ForPreTraining):
    module.project_hid.reset_parameters()
    module.project_q.reset_parameters()

  # gumbel softmax requires special init
  elif isinstance(module, quantizer.Wav2Vec2GumbelVectorQuantizer):
    module.weight_proj.weight.data.normal_(mean=0.0, std=1)
    module.weight_proj.bias.data.zero_()
    nn.init.uniform_(module.codevectors)
  elif isinstance(module, transformer_encoder.Wav2Vec2PositionalConvEmbedding):
    nn.init.normal_(
        module.conv.weight,
        mean=0,
        std=2 * math.sqrt(1 / (
          module.conv.kernel_size[0] * module.conv.in_channels
        )),
    )
    module.conv.bias.data.zero_()

  elif isinstance(module, multidim_wav2vec2.Wav2Vec2FeatureProjection):
    k = math.sqrt(1 / module.projection.in_features)
    nn.init.uniform_(module.projection.weight, a=-k, b=k)
    nn.init.uniform_(module.projection.bias, a=-k, b=k)
  elif isinstance(module, nn.Linear):
    module.weight.data.normal_(mean=0.0, std=config.initializer_range)

    if module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)
  elif isinstance(module, nn.Conv1d):
    nn.init.kaiming_normal_(module.weight)

    if module.bias is not None:
      k = math.sqrt(module.groups / (
        module.in_channels * module.kernel_size[0])
      )
      nn.init.uniform_(module.bias, a=-k, b=k)


