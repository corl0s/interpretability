"""COCO-VTR: benchmarking visual-token readouts (availability, accessibility, causal role).

Milestone 1: LLaVA-1.5-7B, object-identity task, all readouts scored closed-set.

  vtr.data      -- select COCO val2017 objects, per-token coverage (model geometry)
  vtr.geometry  -- map a model's visual tokens to image regions; overlap bins; token sampling
  vtr.cache     -- capture hidden states of sampled tokens (+ pooled, random) at every layer
  vtr.readouts  -- logit lens, LatentLens, Patchscopes, probes -> scores over candidate classes
  vtr.evaluate  -- ranks, controls, image-clustered CIs, plots

Entry point: run_vtr.py (one directory up).
"""
