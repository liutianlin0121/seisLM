"Quantization module"
import math
import einops
from torch import nn
import torch
from seisLM.model.foundation.gumbel import gumbel_softmax

class Wav2Vec2GumbelVectorQuantizer(nn.Module):
  """
  Vector quantization using gumbel softmax.

  Dimension keys:
    B: batch size
    L: sequence length
    G: number of codevector groups
    V: number of codevectors per group
  """

  def __init__(self, config):
    super().__init__()

    self.num_groups = config.num_codevector_groups # = G
    self.num_vars = config.num_codevectors_per_group # = V

    if config.codevector_dim % self.num_groups != 0:
      raise ValueError(
          f"`config.codevector_dim {config.codevector_dim} must be divisible "
          f"by `config.num_codevector_groups` {self.num_groups}"
          "for concatenation"
      )

    # storage for codebook variables (codewords)
    # [1, G * V, codevector_dim // G]
    self.codevectors = nn.Parameter(
      torch.FloatTensor(
        1, self.num_groups * self.num_vars,
        config.codevector_dim // self.num_groups
      )
    )
    self.weight_proj = nn.Linear(
      in_features=config.conv_dim[-1],
      out_features=self.num_groups * self.num_vars
    )

    # can be decayed for training
    self.temperature = 2
    self.sinkhorn_quantization_iters = getattr(
      config, 'sinkhorn_quantization_iters', 0
    )

    self.scale_logits_in_quantization = getattr(
      config, 'scale_logits_in_quantization', False
    )

  @staticmethod
  def _compute_perplexity(probs, mask=None):
    ''' Compute perplexity of the code selection distribution.
    Args:
      probs: [num_sequences, G, V]. For each sequence idx s, group index g,
        probs[s, g, c] is the probabilites of selecting the c-th codevector
        in that group.
    '''

    # avg_probs: [G, V]
    # It is the averaged probabilites over all sequences,
    # denoted by the \bar{p}_{gv} in the Wav2Vec2 paper.
    if mask is not None:
      mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
      probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
      avg_probs = einops.reduce(probs, 's g v -> g v', 'sum') / mask.sum()
    else:
      avg_probs = einops.reduce(probs, 's g v -> g v', 'mean')

    perplexity = torch.exp(
      -einops.reduce(avg_probs * torch.log(avg_probs + 1e-7), 'g v -> g', 'sum')
    )

    perplexity = einops.reduce(perplexity, 'g ->', 'sum')
    return perplexity

  def forward(self, hidden_states, mask_time_indices=None):
    batch_size, sequence_length, feature_dim = hidden_states.shape


    # print('hidden_states.std:', hidden_states.std().item())
    # print('weight_proj.std:', self.weight_proj.weight.std().item())
    # project to codevector dim: [B, L, G * V]
    hidden_states = self.weight_proj(hidden_states)
    # print('hidden_states.std after weight_proj:', hidden_states.std().item())

    if self.scale_logits_in_quantization:
      hidden_states = hidden_states / math.sqrt(feature_dim)
      # print('hidden_states.std after scaling:',
      #       hidden_states.std().item())
      # print('denom:', math.sqrt(feature_dim))



    # hidden_states: [B * L * G, V]
    hidden_states = einops.rearrange(
      hidden_states, 'b l (g v) -> (b l g) v',
      b=batch_size,
      l=sequence_length,
      g=self.num_groups
    )

    if self.training:
      # sample code vector probs via gumbel in differentiateable way
      # codevector_probs: [B * L * G, V]

      # split out the group variable
      hidden_states = einops.rearrange(
        hidden_states,
        '(b l g) v -> (b l) g v',
        b=batch_size,
        l=sequence_length,
        g=self.num_groups
      )

      # codevector_probs: [B * L, G, V]
      codevector_probs = gumbel_softmax(
          hidden_states.float(), tau=self.temperature, hard=True,
          num_sinkhorn_iters=self.sinkhorn_quantization_iters,
          sinkhorn_dims=[2, 0],
      ).type_as(hidden_states)


      # compute perplexity
      # codevector_soft_dist: [B * L, G, V]
      codevector_soft_dist = torch.softmax(hidden_states.float(), dim=-1)
      # print('codevector_soft_dict', codevector_soft_dist)

      perplexity = self._compute_perplexity(
        codevector_soft_dist, mask_time_indices
      )
    else:

      codevector_idx = hidden_states.argmax(dim=-1, keepdim=True)

      # codevector_probs: [B * L * G, V]
      # Each row of codevector_probs is a one-hot vector.
      # The the non-zero index of the i-th row is codevector_idx[i].
      codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
          -1, codevector_idx, 1.0
      )

      codevector_probs = einops.rearrange(
        codevector_probs, '(b l g) v -> (b l) g v',
        b=batch_size,
        l=sequence_length,
        g=self.num_groups
      )

      perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
    # print('codevector_probs.shape [B * L, G, V]:', codevector_probs.shape)
    # print('codevector_probs sum over B*L', codevector_probs.sum(0).detach().cpu().numpy().astype(int))
    # print('codevector_probs over V', codevector_probs.sum(2).detach().cpu().numpy().astype(int))

    # codevector_probs: [B * L, G * V]
    # codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
    codevector_probs = einops.rearrange(
      codevector_probs, '(b l) g v -> (b l) (g v)',
      b=batch_size,
      l=sequence_length,
      g=self.num_groups
    )

    # use probs to retrieve codevectors
    # codevectors_per_group: [B * L, G * V, codevector_dim // G]
    codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors

    # [B * L, G, V, codevector_dim // G]
    codevectors = codevectors_per_group.view(
      batch_size * sequence_length, self.num_groups, self.num_vars, -1
    )
    # sum over code vectors within each group
    codevectors = einops.reduce(
      codevectors, '(b l) g v d -> b l (g d)', 'sum',
      b=batch_size,
      l=sequence_length,
      g=self.num_groups,
      v=self.num_vars,
    )
    return codevectors, perplexity

