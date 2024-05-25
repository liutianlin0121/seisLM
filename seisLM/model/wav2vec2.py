"""Wav2Vec2 model configuration."""

from typing import Optional, Tuple, Union
import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

import einops
from torch import nn


class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
  def __init__(self, config: Wav2Vec2Config):
    super().__init__(config)
    self.wav2vec2 = Wav2Vec2Model(config)
    self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

    self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

    self.project_hid = nn.Linear(
      in_features=config.hidden_size,
      out_features=config.proj_codevector_dim,
    )
    self.project_q = nn.Linear(
      in_features=config.codevector_dim,
      out_features=config.proj_codevector_dim,
    )

    # Initialize weights and apply final processing
    self.post_init()

  def set_gumbel_temperature(self, temperature: int):
    """ Set the Gumbel softmax temperature to a given value.
    Only necessary for training
    """
    self.quantizer.temperature = temperature

  def freeze_feature_encoder(self):
    """ Disable the gradient computation for the feature encoder."""
    self.wav2vec2.feature_extractor._freeze_parameters()

  @staticmethod
  def compute_contrastive_logits(
      target_features: torch.FloatTensor,
      negative_features: torch.FloatTensor,
      predicted_features: torch.FloatTensor,
      temperature: int = 0.1,
  ):
    """
    Compute logits for contrastive loss based using cosine similarity
    as the distance measure between `[positive_feature, negative_features]`
    and `[predicted_features]`. Additionally, temperature can be applied.

    Args:
      target_features [1, B, hidden_length, proj_codevector_dim]:
      negative_features: [num_negatives, B, hidden_length, proj_codevector_dim]:
      predicted_features: [B, hidden_length, proj_codevector_dim]
    """

    # [num_negatives+1, B, hidden_length, proj_codevector_dim]
    target_features = torch.cat([target_features, negative_features], dim=0)

    # [num_negatives+1, B, hidden_length]
    logits = torch.cosine_similarity(
      predicted_features.float(), target_features.float(), dim=-1
      ).type_as(target_features)

    # apply temperature
    logits = logits / temperature
    return logits


  def forward(
      self,
      input_values: Optional[torch.Tensor],
      attention_mask: Optional[torch.Tensor] = None,
      mask_time_indices: Optional[torch.BoolTensor] = None,
      sampled_negative_indices: Optional[torch.BoolTensor] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
  ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
    r"""
    input_values [B, input_length]:
      The input values (wavform).
    attention_mask [B, input_length]:
      The attention mask to apply on the input values.
    mask_time_indices [B, hidden_length]:
      Indices to mask extracted features for contrastive loss.
      When in training mode, model learns to predict
      masked extracted features in *config.proj_codevector_dim* space.
    sampled_negative_indices [B, hidden_length, num_negatives]:
      Indices indicating which quantized target vectors are used as
      negative sampled vectors in contrastive loss.
      Required input for pre-training.

    Returns:

    """
    if not return_dict:
      return_dict = self.config.use_return_dict

    if mask_time_indices is not None:
      mask_time_indices = mask_time_indices.to(torch.bool)

    outputs = self.wav2vec2(
        input_values,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        mask_time_indices=mask_time_indices,
        return_dict=return_dict,
    )

    # 1. project all transformed features (including masked) to final vq dim
    # [B, hidden_length, config.classifier_proj_size]
    transformer_features = self.project_hid(outputs.last_hidden_state)

    # 2. quantize all (unmasked) extracted features and project to final vq dim
    # [B, hidden_length, config.conv_dim[-1]]
    extract_features = self.dropout_features(outputs.extract_features)

    if attention_mask is not None:
      # compute reduced attention_mask correponding to feature vectors
      # [B, hidden_length]
      attention_mask = self._get_feature_vector_attention_mask(
          extract_features.shape[1], attention_mask, add_adapter=False
      )

    # [B, hidden_length, config.codevector_dim]
    quantized_features, codevector_perplexity = self.quantizer(
        extract_features, mask_time_indices=mask_time_indices
    )
    quantized_features = quantized_features.to(self.project_q.weight.dtype)

    # [B, hidden_length, config.proj_codevector_dim]
    quantized_features = self.project_q(quantized_features)

    loss = contrastive_loss = diversity_loss = None
    if sampled_negative_indices is not None:
      (batch_size, hidden_sequence_length, _) = quantized_features.shape

      # for training, we sample negatives
      # 3. sample K negatives (distractors) quantized states for contrastive
      # loss if attention_mask is passed, make sure that padded feature vectors
      # cannot be sampled sample negative quantized vectors

      # Flatten quantized_features to
      # [B * hidden_sequence_length, proj_codevector_dim]
      reshaped_quantized_features = einops.rearrange(
        quantized_features, 'b l d -> (b l) d')

      # Flatten sampled_negative_indices to [B * hidden_sequence_length * Nneg]
      flattened_indices = sampled_negative_indices.long().view(-1)

      # Select negative quantized features using the flattened indices
      # Resulting shape: [B * hidden_sequence_length * Nneg, proj_codevector_dim]
      negative_quantized_features = reshaped_quantized_features[
        flattened_indices]

      # Reshape to [Nneg, B, hidden_sequence_length, proj_codevector_dim]
      negative_quantized_features = einops.rearrange(
          negative_quantized_features,
          '(b l n) d -> n b l d',
          b=batch_size,
          l=hidden_sequence_length,
          n=self.config.num_negatives
      )

      # 4. compute logits, corresponding to
      # `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
      # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
      logits = self.compute_contrastive_logits(
          # [1, B, hidden_length, proj_codevector_dim]:
          target_features=quantized_features[None, :],
          # [num_negatives, B, hidden_length, proj_codevector_dim]:
          negative_features=negative_quantized_features,
          # [B, hidden_length, proj_codevector_dim]
          predicted_features=transformer_features,
          temperature=self.config.contrastive_logits_temperature,
      )

      # 5. if a negative vector is identical to the positive
      # (i.e. when codebook utilization is low),
      # its cosine similarity will be masked
      neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

      if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")

      # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
      # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
      logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
      target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

      contrastive_loss = nn.functional.cross_entropy(
        logits.float(), target, reduction="sum"
      )
      # 7. compute diversity loss: \mathbf{L}_d
      num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
      diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

      # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
      loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

    if not return_dict:
      if loss is not None:
          return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
      return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

    return Wav2Vec2ForPreTrainingOutput(
        loss=loss,
        projected_states=transformer_features,
        projected_quantized_states=quantized_features,
        codevector_perplexity=codevector_perplexity,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        contrastive_loss=contrastive_loss,
        diversity_loss=diversity_loss,
    )
