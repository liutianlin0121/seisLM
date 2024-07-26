
"""Attention-based feature encoder of Wav2Vec2"""
from typing import Optional, Tuple, Union
import ml_collections
import einops
import torch
from torch import nn, Tensor
# import transformers.models.wav2vec2.modeling_wav2vec2 as hf_wav2vec2
from torchtune.modules import RMSNorm
from seisLM.model.foundation import modeling_outputs


class Wav2Vec2PositionalConvEmbedding(nn.Module):
  """Use a convolutional layer, which acts as relative positional embedding."""
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.conv = nn.Conv1d(
        config.hidden_size,
        config.hidden_size,
        kernel_size=config.num_conv_pos_embeddings,
        padding=config.num_conv_pos_embeddings // 2,
        groups=config.num_conv_pos_embedding_groups,
    )

    weight_norm = nn.utils.weight_norm
    if hasattr(nn.utils.parametrizations, "weight_norm"):
      weight_norm = nn.utils.parametrizations.weight_norm

    self.conv = weight_norm(self.conv, name="weight", dim=2)

    self.activation = nn.functional.gelu

    # With a kernel size k, padding k//2, and stride 1, the output of the
    # conv layer has a length of (input_length + 2 (k//2) - k + 1).
    # So if k is even, the output is 1 element longer than the input;
    # we remove the last element to ensure that the
    # position embeddings have the same size as the input sequence.
    # If k is odd, the output has the same length as the input, so we don't
    # need to remove any elements.
    self.remove_one_right = (
      True if config.num_conv_pos_embeddings % 2 == 0 else False
    )

  def forward(self, hidden_states: Tensor) -> Tensor:
    hidden_states = einops.rearrange(hidden_states, "b t c -> b c t")
    hidden_states = self.conv(hidden_states)
    if self.remove_one_right:
      hidden_states = hidden_states[:, :, :-1]

    hidden_states = self.activation(hidden_states)
    hidden_states = einops.rearrange(hidden_states, "b c t -> b t c")
    return hidden_states



class Wav2Vec2FeedForward(nn.Module):
  """Feedforward layer of Wav2Vec2"""
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.intermediate_dropout = nn.Dropout(config.activation_dropout)

    self.intermediate_dense = nn.Linear(
      config.hidden_size, config.intermediate_size
    )
    self.intermediate_act_fn = nn.functional.gelu

    self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.output_dropout = nn.Dropout(config.hidden_dropout)

  def forward(self, hidden_states: Tensor) -> Tensor:
    hidden_states = self.intermediate_dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    hidden_states = self.intermediate_dropout(hidden_states)

    hidden_states = self.output_dense(hidden_states)
    hidden_states = self.output_dropout(hidden_states)
    return hidden_states



class Wav2Vec2SdpaAttention(nn.Module):
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      dropout: float = 0.0,
      bias: bool = True,
  ):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.head_dim = embed_dim // num_heads

    if (self.head_dim * num_heads) != self.embed_dim:
      raise ValueError(
          f"embed_dim must be divisible by num_heads "
          f"(got `embed_dim`: {self.embed_dim}"
          f" and `num_heads`: {num_heads})."
      )
    self.scaling = self.head_dim**-0.5

    self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

  def _shape(self,
             tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
    return tensor.view(
      bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      output_attentions: bool = False,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
             Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    assert output_attentions is False, "output_attentions not supported"
    bsz, tgt_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    query_states = self._shape(query_states, tgt_len, bsz)

    # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2)
    # bugged when using non-contiguous inputs and a custom attn_mask,
    # but we are fine here as `_shape` do call `.contiguous()`.
    # Reference: https://github.com/pytorch/pytorch/issues/112577
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )

    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
      raise ValueError(
          f"`attn_output` should be of size"
          f" {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
          f" {attn_output.size()}"
      )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather
    # than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    return attn_output, None, None


class Wav2Vec2EncoderBase(nn.Module): # pylint: disable=abstract-method
  """ Base Wav2Vec2 encoder.

  Contains the following:
  1. An attention block
  2. An MLP block
  """
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    # self.attention = hf_wav2vec2.Wav2Vec2SdpaAttention(
    self.attention = Wav2Vec2SdpaAttention(
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout,
        # is_decoder=False,
    )
    LayerOrRMSNorm = RMSNorm if config.use_rms_norm else nn.LayerNorm

    self.dropout = nn.Dropout(config.hidden_dropout)
    self.layer_norm = LayerOrRMSNorm(
      config.hidden_size, eps=config.layer_norm_eps
    )
    self.feed_forward = Wav2Vec2FeedForward(config)
    self.final_layer_norm = LayerOrRMSNorm(
      config.hidden_size, eps=config.layer_norm_eps
    )



class Wav2Vec2EncoderLayer(Wav2Vec2EncoderBase):
  """ Wav2Vec2 encoder layer, with post-layer normalization.

  Contains the following:
  1. An attention block, which *ends* with a layer norm.
  2. An MLP block, which *ends* with a layer norm.
  """
  def forward(
    self,
    hidden_states: Tensor,
    *,
    attention_mask: Optional[Tensor],
    output_attentions: bool = False
  )-> Union[Tuple[Tensor], Tuple[Tensor, Tensor]]:

    attn_residual = hidden_states
    hidden_states, attn_weights, _ = self.attention(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions
    )
    hidden_states = self.dropout(hidden_states)
    hidden_states = attn_residual + hidden_states

    hidden_states = self.layer_norm(hidden_states)
    hidden_states = hidden_states + self.feed_forward(hidden_states)
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
      outputs += (attn_weights,) # type: ignore

    return outputs # type: ignore


class Wav2Vec2EncoderLayerStableLayerNorm(Wav2Vec2EncoderBase):
  """ Wav2Vec2 encoder layer, with post-layer normalization.

  Contains the following:
  1. An attention block, which *starts* with a layer norm.
  2. An MLP block, which *starts* with a layer norm.
  """
  def forward(
      self,
      hidden_states: torch.Tensor,
      *,
      attention_mask: Optional[torch.Tensor] = None,
      output_attentions: bool = False,
  )-> Union[Tuple[Tensor], Tuple[Tensor, Tensor]]:
    attn_residual = hidden_states
    hidden_states = self.layer_norm(hidden_states)
    hidden_states, attn_weights, _ = self.attention(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions
    )
    hidden_states = self.dropout(hidden_states)
    hidden_states = attn_residual + hidden_states
    hidden_states = hidden_states + self.feed_forward(
      self.final_layer_norm(hidden_states)
    )

    outputs = (hidden_states,)

    if output_attentions:
      outputs += (attn_weights,)

    return outputs


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.config = config
    self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)

    LayerOrRMSNorm = RMSNorm if config.use_rms_norm else nn.LayerNorm

    self.layer_norm = LayerOrRMSNorm(
      config.hidden_size, eps=config.layer_norm_eps
    )
    self.dropout = nn.Dropout(config.hidden_dropout)
    self.layers = nn.ModuleList(
        [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(
          config.num_hidden_layers
        )]
    )

  def forward(
      self,
      hidden_states: Tensor,
      *,
      attention_mask: Optional[Tensor] = None,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
  ) -> modeling_outputs.BaseModelOutput:
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    if attention_mask is not None:
      # make sure padded tokens are not attended to
      expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
        1, 1, hidden_states.shape[2])
      hidden_states[~expand_attention_mask] = 0
      # extend attention_mask
      attention_mask = 1.0 - attention_mask[:, None, None, :].to(
        dtype=hidden_states.dtype)
      attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min

      attention_mask = attention_mask.expand(
          attention_mask.shape[0], 1,
          attention_mask.shape[-1], attention_mask.shape[-1]
      )

    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.dropout(hidden_states)

    for layer in self.layers:
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
      dropout_probability = torch.rand([])

      skip_the_layer = True if self.training and (
        dropout_probability < self.config.layerdrop) else False

      if skip_the_layer:
        layer_outputs = (None, None)
      else:
        layer_outputs = layer(
            hidden_states, attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = layer_outputs[0]

      if output_attentions:
        all_self_attentions = all_self_attentions + (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    return modeling_outputs.BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )



class Wav2Vec2Encoder(nn.Module):
  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.config = config
    self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)

    LayerOrRMSNorm = RMSNorm if config.use_rms_norm else nn.LayerNorm

    self.layer_norm = LayerOrRMSNorm(
      config.hidden_size, eps=config.layer_norm_eps
    )
    self.dropout = nn.Dropout(config.hidden_dropout)

    self.layers = nn.ModuleList(
        [Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
    ) # type: ignore

  def forward(
      self,
      hidden_states: Tensor,
      *,
      attention_mask: Optional[Tensor] = None,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
  ) -> modeling_outputs.BaseModelOutput:

    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    if attention_mask is not None:
      # make sure padded tokens output 0
      expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
        1, 1, hidden_states.shape[2])
      hidden_states[~expand_attention_mask] = 0

      # extend attention_mask
      attention_mask = 1.0 - attention_mask[:, None, None, :].to(
        dtype=hidden_states.dtype)
      attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
      attention_mask = attention_mask.expand(
          attention_mask.shape[0], 1,
          attention_mask.shape[-1], attention_mask.shape[-1]
      )

    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)


    for layer in self.layers:
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
      dropout_probability = torch.rand([])

      skip_the_layer = True if self.training and (
        dropout_probability < self.config.layerdrop) else False

      if skip_the_layer:
        layer_outputs = (None, None)
      else:
        layer_outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = layer_outputs[0]

      if output_attentions:
        all_self_attentions = all_self_attentions + (layer_outputs[1],)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    return modeling_outputs.BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
