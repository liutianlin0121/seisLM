"""Wav2Vec2 model configuration."""
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch import nn
import ml_collections
from seisLM.model.foundation import modeling_outputs
from seisLM.model.foundation.conv_encoder import Wav2Vec2FeatureEncoder
from seisLM.model.foundation import transformer_encoder
from seisLM.model.foundation.quantizer import Wav2Vec2GumbelVectorQuantizer
import seisLM.model.foundation.mask_utils as mask_utils
from seisLM.model.foundation import initialization

class Wav2Vec2FeatureProjection(nn.Module):
  """Projects the extracted features to the model's hidden size."""
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.layer_norm = nn.LayerNorm(
        config.conv_dim[-1], eps=config.layer_norm_eps
    )
    self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
    self.dropout = nn.Dropout(config.feat_proj_dropout)

  def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
    '''
    Args:
      hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]

    Returns:
      hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
      norm_hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
    '''

    # non-projected hidden states are needed for quantization
    norm_hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.projection(norm_hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states, norm_hidden_states


# class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
class Wav2Vec2Model(nn.Module):
  def __init__(self, config: ml_collections.ConfigDict):
    # super().__init__(config)
    super().__init__()
    self.config = config
    self.feature_extractor = Wav2Vec2FeatureEncoder(config)
    self.feature_projection = Wav2Vec2FeatureProjection(config)

    # model only needs masking vector if mask prob is > 0.0
    if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
      self.masked_spec_embed = nn.Parameter(
        torch.Tensor(config.hidden_size).uniform_()
      )

    if config.do_stable_layer_norm:
      self.encoder = transformer_encoder.Wav2Vec2EncoderStableLayerNorm(config)
    else:
      self.encoder = transformer_encoder.Wav2Vec2Encoder(config)

    # Initialize weights and apply final processing
    self.apply(
      lambda module: initialization.init_wav2vec2_weights(
        config=config, module=module)
    )

  def freeze_feature_encoder(self) -> None:
    """
    Calling this function will disable the gradient computation for
    the feature encoder so that its parameter will
    not be updated during training.
    """
    self.feature_extractor._freeze_parameters()

  def _mask_hidden_states(
      self,
      hidden_states: Tensor,
      *,
      mask_time_indices: Optional[Tensor] = None,
      attention_mask: Optional[Tensor] = None,
  ) -> Tensor:
    """
    Masks extracted features along time axis and/or along feature axis
    according to [SpecAugment](https://arxiv.org/abs/1904.08779).
    """

    # `config.apply_spec_augment` can set masking to False
    if not getattr(self.config, "apply_spec_augment", True):
      return hidden_states

    # generate indices & apply SpecAugment along time axis
    batch_size, sequence_length, hidden_size = hidden_states.size()

    if mask_time_indices is not None:
      # apply SpecAugment along time axis with given mask_time_indices
      hidden_states[mask_time_indices] = self.masked_spec_embed.to(
        hidden_states.dtype
      )
    elif self.config.mask_time_prob > 0 and self.training:
      mask_time_indices = mask_utils.compute_mask_indices( # type: ignore
          (batch_size, sequence_length),
          mask_prob=self.config.mask_time_prob,
          mask_length=self.config.mask_time_length,
          attention_mask=attention_mask,
          min_masks=self.config.mask_time_min_masks,
      )
      mask_time_indices = torch.tensor(
        mask_time_indices, device=hidden_states.device, dtype=torch.bool)
      hidden_states[mask_time_indices] = self.masked_spec_embed.to(
        hidden_states.dtype)

    if self.config.mask_feature_prob > 0 and self.training:
      # generate indices & apply SpecAugment along feature axis
      mask_feature_indices = mask_utils.compute_mask_indices(
          (batch_size, hidden_size),
          mask_prob=self.config.mask_feature_prob,
          mask_length=self.config.mask_feature_length,
          min_masks=self.config.mask_feature_min_masks,
      )
      mask_feature_indices = torch.tensor(
        mask_feature_indices,
        device=hidden_states.device,
        dtype=torch.bool
      )
      mask_feature_indices = mask_feature_indices[:, None].expand(
        -1, sequence_length, -1
      )
      hidden_states[mask_feature_indices] = 0

    return hidden_states


  def forward(
      self,
      input_values: Optional[torch.Tensor],
      attention_mask: Optional[torch.Tensor] = None,
      mask_time_indices: Optional[torch.FloatTensor] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
  ) -> Union[Tuple, modeling_outputs.Wav2Vec2BaseModelOutput]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    extract_features = self.feature_extractor(input_values)
    extract_features = extract_features.transpose(1, 2)

    if attention_mask is not None:
        # compute reduced attention_mask corresponding to feature vectors
        attention_mask = mask_utils.get_feature_vector_attention_mask(
          config=self.config,
          feature_vector_length=extract_features.shape[1],
          attention_mask=attention_mask
            # extract_features.shape[1], attention_mask,
        )

    hidden_states, extract_features = self.feature_projection(extract_features)
    hidden_states = self._mask_hidden_states(
        hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
    )

    encoder_outputs = self.encoder(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    hidden_states = encoder_outputs.last_hidden_state


    return modeling_outputs.Wav2Vec2BaseModelOutput(
        last_hidden_state=hidden_states,
        extract_features=extract_features,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )



# class MultiDimWav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
class MultiDimWav2Vec2ForPreTraining(nn.Module):
  """ Wav2Vec2 model with a contrastive loss head."""
  def __init__(self, config: ml_collections.ConfigDict):
    # super().__init__(config)
    super().__init__()
    self.config = config
    self.wav2vec2 = Wav2Vec2Model(config)
    self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

    self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

    self.project_hid = nn.Linear(
      config.hidden_size, config.proj_codevector_dim
    )
    self.project_q = nn.Linear(
      config.codevector_dim, config.proj_codevector_dim
    )

    # Initialize weights and apply final processing
    self.apply(
      lambda module: initialization.init_wav2vec2_weights(
        config=config, module=module)
    )

  def set_gumbel_temperature(self, temperature: int) -> None:
    """Set the Gumbel softmax temperature to a given value."""
    self.quantizer.temperature = temperature

  def freeze_feature_encoder(self) -> None:
    """Disable the gradient computation for the feature encoder."""
    self.wav2vec2.feature_extractor._freeze_parameters() # pylint: disable=protected-access

  @staticmethod
  def compute_contrastive_logits(
      target_features: torch.Tensor,
      negative_features: torch.Tensor,
      predicted_features: torch.Tensor,
      temperature: float = 0.1,
  ) -> torch.Tensor:
    """Compute logits for contrastive loss."""
    target_features = torch.cat([target_features, negative_features], dim=0)

    logits = torch.cosine_similarity(
      predicted_features.float(), target_features.float(), dim=-1).type_as(
      target_features
    )

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
  ) -> modeling_outputs.Wav2Vec2ForPreTrainingOutput:
    """Forward pass for the Wav2Vec2ForPreTraining model."""

    if mask_time_indices is not None:
      mask_time_indices = mask_time_indices.to(torch.bool) # type: ignore

    outputs = self.wav2vec2(
        input_values,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        mask_time_indices=mask_time_indices,
    )

    # 1. project all transformed features (including masked) to final vq dim
    # transformer_features = self.project_hid(outputs[0])
    transformer_features = self.project_hid(outputs.last_hidden_state)

    # 2. quantize all (unmasked) extracted features and project to final vq dim
    # extract_features = self.dropout_features(outputs[1])
    extract_features = self.dropout_features(outputs.extract_features)

    # if attention_mask is not None:
    #   # compute reduced attention_mask correponding to feature vectors
    #   attention_mask = self._get_feature_vector_attention_mask(
    #       extract_features.shape[1], attention_mask,
    #   )

    quantized_features, codevector_perplexity = self.quantizer(
        extract_features, mask_time_indices=mask_time_indices
    )

    quantized_features = quantized_features.to(self.project_q.weight.dtype)
    quantized_features = self.project_q(quantized_features)

    loss = contrastive_loss = diversity_loss = None
    if sampled_negative_indices is not None:
      batch_size, sequence_length, hidden_size = quantized_features.shape

      # for training, we sample negatives
      # 3. sample K negatives (distractors) quantized states for
      # contrastive loss if attention_mask is passed, make sure that padded
      # feature vectors cannot be sampled
      # sample negative quantized vectors BTC => (BxT)C
      negative_quantized_features = quantized_features.view(-1, hidden_size)[
          sampled_negative_indices.long().view(-1)
      ]
      negative_quantized_features = negative_quantized_features.view(
          batch_size, sequence_length, -1, hidden_size
      ).permute(2, 0, 1, 3)

      # 4. compute logits, corresponding to
      # `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
      # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
      logits = self.compute_contrastive_logits(
          quantized_features[None, :],
          negative_quantized_features,
          transformer_features,
          self.config.contrastive_logits_temperature,
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
      target = ((1 - mask_time_indices.long()) * -100).transpose( # type: ignore
        0, 1).flatten()

      contrastive_loss = nn.functional.cross_entropy(
        logits.float(), target, reduction="sum"
      )
      # 7. compute diversity loss: \mathbf{L}_d
      num_codevectors = (
        self.config.num_codevectors_per_group * (
          self.config.num_codevector_groups)
      )
      diversity_loss = (
        (num_codevectors - codevector_perplexity) / num_codevectors
      ) * mask_time_indices.sum() # type: ignore

      # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
      loss = contrastive_loss + (
        self.config.diversity_loss_weight * diversity_loss)

    outputs =  modeling_outputs.Wav2Vec2ForPreTrainingOutput(
        loss=loss,
        projected_states=transformer_features,
        projected_quantized_states=quantized_features,
        codevector_perplexity=codevector_perplexity,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        contrastive_loss=contrastive_loss,
        diversity_loss=diversity_loss,
    )

    return outputs # type: ignore




