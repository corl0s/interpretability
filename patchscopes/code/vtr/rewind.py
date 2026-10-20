"""Rewind maps: label-free affine maps between layers of a visual token's state.

For a pair of layers (s, t), fit h_t ~ h_s @ W + b by ridge regression on the image-token states
of unlabeled images (same token, same image, the output of decoder blocks s and t). A state read
at layer s can then be translated into layer t's format before it is injected at layer t, so a
deep state can be injected early and still look like what that layer normally sees.

No class labels are used anywhere: the maps only learn how the model's own visual states change
between layers.

  PairStats   streams sufficient statistics (sums, Gram and cross matrices, float64)
  fit_maps    ridge solutions per pair, alpha chosen on validation tokens, R^2 on test tokens
  RewindMaps  loads the saved maps and applies them
"""

import pandas as pd
import torch


class PairStats:
  """Sufficient statistics for ridge regression of layer t on layer s, for many (s, t) pairs."""

  def __init__(self, pairs, d, device):
    self.pairs = sorted(set(pairs))
    self.sources = sorted({s for s, _ in self.pairs})
    self.layers = sorted({l for p in self.pairs for l in p})
    self.device, self.d, self.n = device, d, 0
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=device)
    self.sum = {l: z(d) for l in self.layers}
    self.gram = {s: z(d, d) for s in self.sources}
    self.cross = {p: z(d, d) for p in self.pairs}

  @torch.no_grad()
  def update(self, full):
    """full: [n_layers, n_tokens, d] states of one image."""
    x = {l: full[l].to(self.device, torch.float32) for l in self.layers}
    for l in self.layers:
      self.sum[l] += x[l].sum(0).double()
    for s in self.sources:
      self.gram[s] += (x[s].T @ x[s]).double()
    for s, t in self.pairs:
      self.cross[(s, t)] += (x[s].T @ x[t]).double()
    self.n += full.shape[1]


def r2(pred, y):
  """Pooled R^2 over all dimensions (1 = perfect, 0 = no better than the mean of y)."""
  y = y.double()
  return float(1 - ((y - pred.double()) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


@torch.no_grad()
def fit_maps(stats, val, test, alphas=(1e-4, 1e-3, 1e-2, 1e-1)):
  """val / test: {layer: [n_tokens, d] tensor} of held-out tokens.

  alpha is relative: ridge = alpha * trace(cov_s) / d. Returns ({(s, t): (W fp16, b fp32)}, df).
  """
  maps, rows = {}, []
  n, dev = stats.n, stats.device
  eye = torch.eye(stats.d, dtype=torch.float64, device=dev)
  for s, t in stats.pairs:
    mu_s, mu_t = stats.sum[s] / n, stats.sum[t] / n
    cov = stats.gram[s] / n - torch.outer(mu_s, mu_s)
    cross = stats.cross[(s, t)] / n - torch.outer(mu_s, mu_t)
    scale = torch.trace(cov) / stats.d
    best = None
    for alpha in alphas:
      W = torch.linalg.solve(cov + alpha * scale * eye, cross).to(torch.float16)
      b = (mu_t - mu_s @ W.double()).float()
      score = r2(apply_map(val[s], W, b), val[t])
      if best is None or score > best[0]:
        best = (score, alpha, W.cpu(), b.cpu())
    score, alpha, W, b = best
    maps[(s, t)] = (W, b)
    rows.append({"source": s, "target": t, "alpha": alpha, "r2_val": score,
                 "r2_test": r2(apply_map(test[s], W, b), test[t]),
                 "r2_test_identity": r2(test[s], test[t]),
                 "norm_ratio_test": float(test[t].float().norm(dim=-1).mean()
                                          / test[s].float().norm(dim=-1).mean()),
                 "n_train_tokens": n})
    print(f"  map {s:2d} -> {t:2d}: alpha {alpha:g}, R2 val {score:.3f}, "
          f"test {rows[-1]['r2_test']:.3f} (identity {rows[-1]['r2_test_identity']:.3f})",
          flush=True)
  return maps, pd.DataFrame(rows)


def apply_map(h, W, b):
  dev = W.device
  return h.to(dev, torch.float32) @ W.float() + b.to(dev)


class RewindMaps:

  def __init__(self, path):
    blob = torch.load(path, weights_only=False)
    self.maps = blob["maps"]
    self.meta = {k: v for k, v in blob.items() if k != "maps"}

  def has(self, s, t):
    return s == t or (s, t) in self.maps

  def apply(self, h, s, t):
    """h [n, d] read at layer s -> its estimate at layer t (identity when s == t)."""
    if s == t:
      return h
    W, b = self.maps[(s, t)]
    return apply_map(h, W, b).to(h.dtype)


def map_pairs(sources, targets):
  return [(s, t) for s in sources for t in targets if s != t]
