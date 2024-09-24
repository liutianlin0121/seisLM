"""Model output registry

Dimension key:

B: batch size
L: sequence length
D: feature dimension
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor


@dataclass
class BaseModelOutput:
  """
  Base class for model's outputs, with potential hidden states and attentions.

  Args:
    last_hidden_state: hidden-states at the output of the last
      layer of the model. [B, L, D].

    hidden_states: Hidden-states of the model at the output of each layer plus
      the optional initial embedding outputs. Tuple of [B, L, D].

    attentions: Attentions weights after the attention softmax.
      Tuple of [B, H, L, L].
  """

  last_hidden_state: Optional[Tensor] = None
  hidden_states: Optional[Tuple[Tensor, ...]] = None
  attentions: Optional[Tuple[Tensor, ...]] = None


@dataclass
class Wav2Vec2ForPreTrainingOutput(BaseModelOutput):
  """
  Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states
  and attentions.

  Args:
      loss: Total loss as the sum of the contrastive loss and
          the diversity loss. [1,].

      projected_states: Hidden-states of the model projected
          to *config.proj_codevector_dim* that can be used to predict the masked
          projected quantized states. [B, L, proj_codevector_dim].

      codevector_perplexity: The perplexity of the codevectors

      projected_quantized_states: Quantized extracted feature vectors
          projected to config.proj_codevector_dim representing the positive
          target vectors for contrastive loss.
          [B, L, proj_codevector_dim].

      codevector_perplexity: The codevector perplexity as stated in the
          [official paper](https://arxiv.org/pdf/2006.11477.pdf). [1,].

      contrastive_loss: The contrastive loss (L_m) as stated in the
          [official paper](https://arxiv.org/pdf/2006.11477.pdf). [1,].

      diversity_loss: The diversity loss (L_d) as stated in the
          [official paper](https://arxiv.org/pdf/2006.11477.pdf). [1,].
  """

  loss: Optional[Tensor] = None
  projected_states: Optional[Tensor] = None
  projected_quantized_states: Optional[Tensor] = None
  codevector_perplexity: Optional[Tensor] = None
  contrastive_loss: Optional[Tensor] = None
  diversity_loss: Optional[Tensor] = None


@dataclass
class Wav2Vec2BaseModelOutput(BaseModelOutput):
  """
  Base class for models that have been trained with the Wav2Vec2 loss objective.

  Args:
      extract_features: Extracted feature vectors of the last convolutional
          layer of the model. [batch_size, L, config.conv_dim[-1]]:
  """

  extract_features: Optional[Tensor] = None
