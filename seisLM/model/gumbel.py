import warnings
import torch
from torch.overrides import has_torch_function_unary
from torch.overrides import handle_torch_function


Tensor = torch.Tensor



def log_sinkhorn(log_alpha, n_iter):
  """Performs incomplete Sinkhorn normalization to log_alpha.
  By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
  with positive entries can be turned into a doubly-stochastic matrix
  (i.e. its rows and columns add up to one) via the successive row and column
  normalization.

  [1] Sinkhorn, Richard and Knopp, Paul.
  Concerning nonnegative matrices and doubly stochastic
  matrices. Pacific Journal of Mathematics, 1967
  Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
      or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    n_iters: number of sinkhorn iterations (in practice, as little as 20
      iterations are needed to achieve decent convergence for N~100)
  Returns:
    A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
      converted to 3D tensors with batch_size equals to 1)
  """
  for _ in range(n_iter):
    log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
    log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
  return log_alpha.exp()



def gumbel_softmax(
  logits: Tensor, tau: float = 1, hard: bool = False,
  eps: float = 1e-10, dim: int = -1, num_sinkhorn_iters = 0) -> Tensor:
  r"""
  Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_)
  and optionally discretize.

  Args:
    logits: `[..., num_features]` unnormalized log probabilities
    tau: non-negative scalar temperature
    hard: if ``True``, the returned samples will be discretized as one-hot
          vectors, but will be differentiated as if it is the soft sample
          in autograd
    dim (int): A dimension along which softmax will be computed. Default: -1.

  Returns:
    Sampled tensor of same shape as `logits` from the Gumbel-Softmax
    distribution.
    If ``hard=True``, the returned samples will be one-hot, otherwise they will
    be probability distributions that sum to 1 across `dim`.

  .. note::
    This function is here for legacy reasons, may be removed from nn.Functional
    in the future.

  .. note::
    The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

    It achieves two things:
    - makes the output value exactly one-hot
    (since we add then subtract y_soft value)
    - makes the gradient equal to y_soft gradient
    (since we strip all other gradients)

  Examples::
      >>> logits = torch.randn(20, 32)
      >>> # Sample soft categorical using reparametrization trick:
      >>> F.gumbel_softmax(logits, tau=1, hard=False)
      >>> # Sample hard categorical using "Straight-through" trick:
      >>> F.gumbel_softmax(logits, tau=1, hard=True)

  .. _Link 1:
      https://arxiv.org/abs/1611.00712
  .. _Link 2:
      https://arxiv.org/abs/1611.01144
  """
  if has_torch_function_unary(logits):
      return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
  if eps != 1e-10:
      warnings.warn("`eps` parameter is deprecated and has no effect.")

  gumbels = (
      -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
  )  # ~Gumbel(0,1)
  gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

  if num_sinkhorn_iters > 0:
    y_soft = log_sinkhorn(gumbels, num_sinkhorn_iters)
  else:
    y_soft = gumbels.softmax(dim)

  if hard:
      # Straight through.
      index = y_soft.max(dim, keepdim=True)[1]
      y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
      ret = y_hard - y_soft.detach() + y_soft
  else:
      # Reparametrization trick.
      ret = y_soft
  return ret
