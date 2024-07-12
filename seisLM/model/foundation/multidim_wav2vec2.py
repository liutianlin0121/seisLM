"""Wav2Vec2 model configuration."""
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
import numpy as np
import ml_collections
from transformers import PreTrainedModel
from seisLM.model.foundation import modeling_outputs
from seisLM.model.foundation.conv_encoder import Wav2Vec2FeatureEncoder
from seisLM.model.foundation import transformer_encoder
from seisLM.model.foundation.quantizer import Wav2Vec2GumbelVectorQuantizer



class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states

class Wav2Vec2PreTrainedModel(PreTrainedModel):
  """
  An abstract class to handle weights initialization and
  a simple interface for downloading and loading pretrained
  models.
  """

  config = ml_collections.ConfigDict
  base_model_prefix = "wav2vec2"
  main_input_name = "input_values"
  supports_gradient_checkpointing = True
  _supports_flash_attn_2 = True
  _supports_sdpa = True

  def _init_weights(self, module: nn.Module) -> None:
    """Initialize the weights"""
    # Wav2Vec2ForPreTraining last 2 linear layers need standard Linear init.
    if isinstance(module, MultiDimWav2Vec2ForPreTraining):
      module.project_hid.reset_parameters()
      module.project_q.reset_parameters()
      module.project_hid._is_hf_initialized = True
      module.project_q._is_hf_initialized = True
    # gumbel softmax requires special init
    elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
      module.weight_proj.weight.data.normal_(mean=0.0, std=1)
      module.weight_proj.bias.data.zero_()
      nn.init.uniform_(module.codevectors)
    elif isinstance(module, transformer_encoder.Wav2Vec2PositionalConvEmbedding):
      nn.init.normal_(
          module.conv.weight,
          mean=0,
          std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
      )
      nn.init.constant_(module.conv.bias, 0)
    elif isinstance(module, Wav2Vec2FeatureProjection):
      k = math.sqrt(1 / module.projection.in_features)
      nn.init.uniform_(module.projection.weight, a=-k, b=k)
      nn.init.uniform_(module.projection.bias, a=-k, b=k)
    elif isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

      if module.bias is not None:
          module.bias.data.zero_()
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Conv1d):
      nn.init.kaiming_normal_(module.weight)

      if module.bias is not None:
        k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
        nn.init.uniform_(module.bias, a=-k, b=k)

  def _get_feat_extract_output_lengths(
      self, input_lengths: Union[torch.LongTensor, int],
  ) -> int:
    """
    Computes the output length of the convolutional layers
    """

    def _conv_out_length(input_length, kernel_size, stride):
      # 1D convolutional layer output length formula taken
      # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
      return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

    for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
      input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    return input_lengths

  def _get_feature_vector_attention_mask(
      self,
      feature_vector_length: int,
      attention_mask: torch.LongTensor,
  ) -> torch.Tensor:
    # Effectively attention_mask.sum(-1), but not inplace to be able to run
    # on inference mode.
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

    output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
    output_lengths = output_lengths.to(torch.long)

    batch_size = attention_mask.shape[0]

    attention_mask = torch.zeros(
        (batch_size, feature_vector_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device
    )
    # these two operations makes sure that all values
    # before the output lengths idxs are attended to
    attention_mask[(torch.arange(
      attention_mask.shape[0],
      device=attention_mask.device
    ), output_lengths - 1)] = 1
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return attention_mask


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
  def __init__(self, config: ml_collections.ConfigDict):
      super().__init__(config)
      self.config = config
      self.feature_extractor = Wav2Vec2FeatureEncoder(config)
      self.feature_projection = Wav2Vec2FeatureProjection(config)

      # model only needs masking vector if mask prob is > 0.0
      if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
          self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

      if config.do_stable_layer_norm:
          self.encoder = transformer_encoder.Wav2Vec2EncoderStableLayerNorm(config)
      else:
          self.encoder = transformer_encoder.Wav2Vec2Encoder(config)

      self.adapter = None

      # Initialize weights and apply final processing
      self.post_init()

  def freeze_feature_encoder(self):
      """
      Calling this function will disable the gradient computation for the feature encoder so that its parameter will
      not be updated during training.
      """
      self.feature_extractor._freeze_parameters()

  def _mask_hidden_states(
      self,
      hidden_states: torch.FloatTensor,
      mask_time_indices: Optional[torch.FloatTensor] = None,
      attention_mask: Optional[torch.LongTensor] = None,
  ):
      """
      Masks extracted features along time axis and/or along feature axis according to
      [SpecAugment](https://arxiv.org/abs/1904.08779).
      """

      # `config.apply_spec_augment` can set masking to False
      if not getattr(self.config, "apply_spec_augment", True):
          return hidden_states

      # generate indices & apply SpecAugment along time axis
      batch_size, sequence_length, hidden_size = hidden_states.size()

      if mask_time_indices is not None:
          # apply SpecAugment along time axis with given mask_time_indices
          hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
      elif self.config.mask_time_prob > 0 and self.training:
          mask_time_indices = _compute_mask_indices(
              (batch_size, sequence_length),
              mask_prob=self.config.mask_time_prob,
              mask_length=self.config.mask_time_length,
              attention_mask=attention_mask,
              min_masks=self.config.mask_time_min_masks,
          )
          mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
          hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

      if self.config.mask_feature_prob > 0 and self.training:
          # generate indices & apply SpecAugment along feature axis
          mask_feature_indices = _compute_mask_indices(
              (batch_size, hidden_size),
              mask_prob=self.config.mask_feature_prob,
              mask_length=self.config.mask_feature_length,
              min_masks=self.config.mask_feature_min_masks,
          )
          mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
          mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
          hidden_states[mask_feature_indices] = 0

      return hidden_states

  def forward(
      self,
      input_values: Optional[torch.Tensor],
      attention_mask: Optional[torch.Tensor] = None,
      mask_time_indices: Optional[torch.FloatTensor] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
  ) -> Union[Tuple, modeling_outputs.Wav2Vec2BaseModelOutput]:
      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
      output_hidden_states = (
          output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      )
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict

      extract_features = self.feature_extractor(input_values)
      extract_features = extract_features.transpose(1, 2)

      if attention_mask is not None:
          # compute reduced attention_mask corresponding to feature vectors
          attention_mask = self._get_feature_vector_attention_mask(
              extract_features.shape[1], attention_mask,
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
          return_dict=return_dict,
      )

    #   hidden_states = encoder_outputs[0]
      hidden_states = encoder_outputs.last_hidden_state

      if self.adapter is not None:
          hidden_states = self.adapter(hidden_states)

      if not return_dict:
          return (hidden_states, extract_features) + encoder_outputs[1:]

      return modeling_outputs.Wav2Vec2BaseModelOutput(
          last_hidden_state=hidden_states,
          extract_features=extract_features,
          hidden_states=encoder_outputs.hidden_states,
          attentions=encoder_outputs.attentions,
      )



class MultiDimWav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
  """ Wav2Vec2 model with a contrastive loss head."""
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    # self.wav2vec2 = MultiDimWav2Vec2Model(config)
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
    self.post_init()

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
        return_dict=True,
    )

    # 1. project all transformed features (including masked) to final vq dim
    # transformer_features = self.project_hid(outputs[0])
    transformer_features = self.project_hid(outputs.last_hidden_state)

    # 2. quantize all (unmasked) extracted features and project to final vq dim
    # extract_features = self.dropout_features(outputs[1])
    extract_features = self.dropout_features(outputs.extract_features)

    if attention_mask is not None:
      # compute reduced attention_mask correponding to feature vectors
      attention_mask = self._get_feature_vector_attention_mask(
          extract_features.shape[1], attention_mask,
      )

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
      target = ((1 - mask_time_indices.long()) * -100).transpose(
        0, 1).flatten() # type: ignore

      contrastive_loss = nn.functional.cross_entropy(
        logits.float(), target, reduction="sum"
      )
      # 7. compute diversity loss: \mathbf{L}_d
      num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
      diversity_loss = (
        (num_codevectors - codevector_perplexity) / num_codevectors
      ) * mask_time_indices.sum()

      # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
      loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

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

    return outputs




