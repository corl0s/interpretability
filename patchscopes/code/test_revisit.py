"""CPU tests (no model) for the revisiting experiments:

  * run_object_pruning.knockout_mask hides the chosen keys from every other query, in the active
    rows only, for None / bool / additive masks, and leaves causality intact
  * run_object_pruning.knockout edits the mask each layer receives only past layer K
  * run_token_maps: Kang's sink ratio, the largest near-duplicate cluster, map metrics
  * rescore_region.region_pool max-pools per image
  * run_hallucination.find_mentions drops matches inside a longer match of another class
"""

import numpy as np
import torch
import torch.nn as nn

import rescore_region as R
import run_hallucination as H
import run_object_pruning as P
import run_token_maps as T


def test_knockout_mask():
  q = k = 5
  base = P.knockout_mask(None, [[], []], [False, False], q, k, torch.float32, "cpu")
  assert base.dtype == torch.bool and base.shape == (2, 1, 5, 5)
  assert torch.equal(base[0, 0], torch.ones(5, 5, dtype=torch.bool).tril())
  m = P.knockout_mask(None, [[1, 2], [1]], [True, False], q, k, torch.float32, "cpu")
  assert not m[0, 0, 3, 1] and not m[0, 0, 4, 2] and not m[0, 0, 2, 1]
  assert m[0, 0, 1, 1] and m[0, 0, 2, 2]          # a token still attends to itself
  assert m[0, 0, 3, 0] and not m[0, 0, 0, 1]      # other keys and causality unchanged
  assert torch.equal(m[1], base[1])               # inactive row untouched
  add = torch.zeros(1, 1, 5, 5).masked_fill(~base[:1].bool(), float("-inf"))
  f = P.knockout_mask(add, [[1]], [True], q, k, torch.float32, "cpu")
  assert f[0, 0, 3, 1] == torch.finfo(torch.float32).min and f[0, 0, 1, 1] == 0
  assert f[0, 0, 0, 1] == float("-inf")


class Spy(nn.Module):
  def __init__(self):
    super().__init__()
    self.seen = None

  def forward(self, hidden_states, attention_mask=None):
    self.seen = attention_mask
    return (hidden_states,)


def test_knockout_layers():
  layers = nn.ModuleList([Spy() for _ in range(4)])
  x = torch.zeros(1, 6, 2)
  with P.knockout(layers, [np.array([2])], [1]):
    for l in layers:
      l(x, attention_mask=None)
  assert layers[0].seen is None and layers[1].seen is None
  assert layers[2].seen is not None and not layers[2].seen[0, 0, 4, 2]
  assert layers[3].seen[0, 0, 2, 2]


def test_sinks_and_clusters():
  x = torch.ones(3, 8)
  x[1, 5] = 100.0
  r = T.sink_ratio(x, [5])
  assert r[1] > 2.5 and abs(r[0] - 1) < 1e-5
  y = torch.tensor([[1.0, 0], [0.99, 0.01], [1.0, 0.02], [0, 1.0]])
  assert sorted(T.largest_cluster(y, 0.9).tolist()) == [0, 1, 2]


def test_map_metrics():
  cov = np.array([1.0, 0.6, 0.2, 0.0])
  scores = np.array([[2.0, 1.0], [0.0, 1.0], [3.0, 0.0], [1.0, 0.0]])  # label 0
  m = T.map_metrics(scores, cov, 0)
  assert m["recall"] == 0.5                       # token 0 of the two on-object tokens
  assert np.isclose(m["precision"], 2 / 3)        # tokens 0, 2, 3 name it; 0 and 2 touch it
  assert np.isclose(m["iou"], 1 / 4)
  assert m["region_top1"] == 1.0                  # max-pool over tokens 0, 1: [2, 1]


def test_region_pool():
  s = np.array([[1.0, 0], [0, 2.0], [5.0, 0]])
  pooled, ids = R.region_pool(s, [7, 7, 9])
  assert list(ids) == [7, 9] and pooled.tolist() == [[1.0, 2.0], [5.0, 0.0]]


def test_mentions():
  forms = {"dog": ["dog"], "hot dog": ["hot dog"], "person": ["person", "man"]}
  found = H.find_mentions("A man eats a hot dog. Two men and a dog.", forms)
  assert set(found) == {"person", "hot dog", "dog"}
  assert found["dog"] == "A man eats a hot dog. Two men and a ".__len__()
  assert "dog" not in H.find_mentions("A hot dog on a plate.", forms)


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_"):
      fn()
      print(f"{name}: ok")
