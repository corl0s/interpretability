r"""Read EVERY visual token of a set of benchmark images, at a few layers, with several readouts.

The benchmark cache (run_vtr.py) holds a sample of each image's tokens; the questions below need
all of them, so this script captures all image tokens and reads each one.

  A1 maps    per readout and layer: which tokens name the image's object, against the mask:
             recall (tokens with coverage >= 0.5 that name it), precision (tokens naming it that
             touch the object), IoU of the two sets, and region top-1 (class scores max-pooled
             over the whole object region, as Takishita et al. 2506.05439 score parts)
  A2/A5      special tokens, flagged per token:
               llm_sink   Kang et al. (ICLR 2025) rule: max over --sink_dims of |x_d| / RMS(x)
                          >= --sink_tau, on the output of block --sink_layer
                          (LLaVA-1.5-7B: dims {1415, 2533}, tau 20)
               vit_sink   CLIP last-layer patch norm > --vit_tau (EmbedLens Sec. 4.1: 75)
  A4 dead    EmbedLens (Fan et al., 2603.00510) "dead" tokens, two versions:
               dead_id       the projector output's nearest input embedding (whole vocabulary,
                             cosine) is token --dead_token_id (their released code's target 30296)
               dead_cluster  the largest group of the image's projector outputs within cosine
                             --dead_cos (0.9) of one anchor (the token with most such
                             neighbours). Our reading of their Appendix B.1, which we only saw
                             summarised: treat as an approximation.
             For every group: the share of its tokens whose top-1 is the image's object, per
             readout and layer. If image-slot reads the object from tokens the embedding lens
             calls dead, they are not empty.

Readouts: logit_lens, embedding_lens (closed set, as in run_vtr.py) and image_slot_pmi (all
image slots of a blank-image prompt, target per source layer from <vtr_dir>/injection_proto).
Scores are over the fair classes.

  python run_token_maps.py --coco_dir /path/to/coco --vtr_dir ./results/vtr_llava15 \
      --n_images 50 --layers 0,8,16,24
    -> <vtr_dir>/token_maps/{maps/<image_id>.npz, maps.csv, groups.csv, special.csv}

Resumable per image. --stage summarize only re-reads the saved maps. LLaVA-1.5 only (the ViT
hook and the sink dimensions are LLaVA's).
"""

import argparse
import glob
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_vtr  # noqa: E402
from train_oracle import load_benchmark  # noqa: E402
from visual_object_identification import ensure_image, get_decoder  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE  # noqa: E402
from vtr.injection import IMAGE_QUESTION, InjectionReadout  # noqa: E402
from vtr.models import chat_inputs  # noqa: E402
from vtr.readouts import EmbeddingLens, LogitLens, class_forms  # noqa: E402

READOUTS = ("logit_lens", "embedding_lens", "image_slot_pmi")
GROUPS = ("object", "background", "llm_sink", "vit_sink", "dead_id", "dead_cluster", "alive")


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "run", "summarize"])
  p.add_argument("--coco_dir", default=None)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default <vtr_dir>/token_maps")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--n_images", type=int, default=50)
  p.add_argument("--layers", default="0,8,16,24")
  p.add_argument("--readouts", default=",".join(READOUTS))
  p.add_argument("--slot_target", default="dev", help="as in run_vtr.py")
  p.add_argument("--slot_rows_per_batch", type=int, default=32)
  p.add_argument("--sink_dims", default="1415,2533")
  p.add_argument("--sink_layer", type=int, default=2)
  p.add_argument("--sink_tau", type=float, default=20.0)
  p.add_argument("--vit_tau", type=float, default=75.0)
  p.add_argument("--dead_token_id", type=int, default=30296)
  p.add_argument("--dead_cos", type=float, default=0.9)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


# ---------------------------------------------------------------------------------- flags


def sink_ratio(x, dims):
  """Kang et al.'s sink score: max over `dims` of |x_d| / RMS(x). x [n, d] -> [n]."""
  x = x.float()
  rms = x.pow(2).mean(-1).sqrt().clamp_min(1e-6)
  return x[:, dims].abs().max(-1).values / rms


def largest_cluster(x, cos):
  """Indices of the largest set of rows within cosine `cos` of one anchor row (the row with the
  most such neighbours). x [n, d]."""
  z = F.normalize(x.float(), dim=-1)
  near = (z @ z.T) >= cos
  anchor = int(near.sum(1).argmax())
  return torch.nonzero(near[anchor]).flatten().cpu().numpy()


# ---------------------------------------------------------------------------------- model


@torch.no_grad()
def capture(mt, image):
  """(LLM input at the image positions [n, d], block outputs [L, n, d], CLIP last-layer patch
  norms [n]) for one image."""
  x, start, n = chat_inputs(mt, [image], IMAGE_QUESTION)
  layers = get_decoder(mt).layers
  got = {}

  def pre(module, args, kwargs):
    hs = args[0] if args else kwargs["hidden_states"]
    got["in"] = hs[0, start:start + n].detach()

  def out_hook(k):
    def hook(module, inp, out):
      got[k] = (out[0] if isinstance(out, tuple) else out)[0, start:start + n].detach()
    return hook

  def vit_hook(module, inp, out):
    got["vit"] = (out[0] if isinstance(out, tuple) else out)[0, 1:].detach()  # drop CLS

  handles = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
  handles += [l.register_forward_hook(out_hook(k)) for k, l in enumerate(layers)]
  # The CLIP layers' path differs across transformers versions (vision_model.encoder.layers in
  # 4.x); find the last one by name.
  vit_layers = [m for n, m in mt.vision_tower.named_modules()
                if re.match(r".*encoder\.layers\.\d+$", n)]
  handles.append(vit_layers[-1].register_forward_hook(vit_hook))
  try:
    mt.model(**x, use_cache=False)
  finally:
    for h in handles:
      h.remove()
  states = torch.stack([got[k] for k in range(len(layers))])
  return got["in"], states, got["vit"].float().norm(dim=-1).cpu()


def build_readouts(args, mt, class_names, layers):
  forms = class_forms(class_names, synonyms=False)
  names = args.readouts.split(",")
  out = {}
  if "logit_lens" in names:
    out["logit_lens"] = LogitLens(mt, class_names, forms, args.device).scores
  if "embedding_lens" in names:
    out["embedding_lens"] = EmbeddingLens(mt, class_names, forms, args.device).scores
  if "image_slot_pmi" in names:
    targets = run_vtr.slot_targets(SimpleNamespace(slot_target=args.slot_target,
                                                   out_dir=args.vtr_dir), layers)
    print(f"image-slot target per source layer: {targets}")
    ro = InjectionReadout(mt, class_names, forms, "image", args.device,
                          rows_per_batch=args.slot_rows_per_batch)
    out["image_slot_pmi"] = lambda h, l: ro.scores(h, targets[l], [ro.slots("all")] * len(h))[1]
  return out


def stage_run(args, out_dir, data, kept):
  map_dir = os.path.join(out_dir, "maps")
  os.makedirs(map_dir, exist_ok=True)
  class_names = data["class_names"]
  keep = [class_names.index(c) for c in kept]
  layers = [int(x) for x in args.layers.split(",")]
  dims = [int(x) for x in args.sink_dims.split(",")]
  pool = [s for s in data["samples"] if s["class_name"] in kept]
  chosen = [pool[i] for i in np.random.RandomState(args.seed).permutation(len(pool))[:args.n_images]]

  mt = run_vtr.load_model(SimpleNamespace(model_name=args.model_name, device=args.device))
  if getattr(mt, "family", "llava") != "llava":
    raise SystemExit("run_token_maps.py is LLaVA-1.5 only")
  readouts = build_readouts(args, mt, class_names, layers)
  emb = F.normalize(mt.model.get_input_embeddings().weight.detach().float(), dim=-1)

  for k, s in enumerate(chosen):
    path = os.path.join(map_dir, f"{s['image_id']}.npz")
    if os.path.exists(path):
      continue
    t0 = time.time()
    image = Image.open(ensure_image(args.coco_dir, s["file_name"])).convert("RGB")
    llm_in, states, vit_norm = capture(mt, image)
    cov = np.asarray(s["coverage"], dtype=np.float32)
    if states.shape[1] != len(cov):
      raise ValueError(f"image {s['image_id']}: {states.shape[1]} tokens, {len(cov)} coverage")
    nearest = (F.normalize(llm_in.float(), dim=-1) @ emb.T).argmax(-1).cpu().numpy()
    dead_cluster = np.zeros(len(cov), bool)
    dead_cluster[largest_cluster(llm_in, args.dead_cos)] = True
    scores = np.full((len(readouts), len(layers), len(cov), len(keep)), np.nan, np.float16)
    for ri, fn in enumerate(readouts.values()):
      for li, l in enumerate(layers):
        scores[ri, li] = np.asarray(fn(states[l].float(), l))[:, keep]
    np.savez(path, scores=scores, readouts=np.asarray(list(readouts)), layers=np.asarray(layers),
             coverage=cov, label=kept.index(s["class_name"]),
             sink_ratio=sink_ratio(states[args.sink_layer], dims).cpu().numpy(),
             vit_norm=vit_norm.numpy(), dead_id=nearest == args.dead_token_id,
             dead_cluster=dead_cluster)
    print(f"  [{k + 1}/{len(chosen)}] {s['class_name']} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------- summary


def groups(d, args):
  cov = d["coverage"]
  llm = d["sink_ratio"] >= args.sink_tau
  vit = d["vit_norm"] > args.vit_tau
  return {"object": cov >= HIGH_COVERAGE, "background": cov == 0, "llm_sink": llm,
          "vit_sink": vit, "dead_id": d["dead_id"], "dead_cluster": d["dead_cluster"],
          "alive": ~(llm | vit | d["dead_cluster"])}


def map_metrics(scores, cov, label):
  """scores [n_tokens, C] at one layer -> recall, precision, IoU, region top-1."""
  names = scores.argmax(1) == label
  on = cov >= HIGH_COVERAGE
  inter = (names & on).sum()
  union = (names | on).sum()
  region = scores[on].max(0) if on.any() else np.full(scores.shape[1], np.nan)
  return {"recall": inter / max(on.sum(), 1),
          "precision": (names & (cov > 0)).sum() / names.sum() if names.any() else np.nan,
          "iou": inter / union if union else np.nan,
          "region_top1": float(E.ranks(region[None], np.array([label]))[0] == 1)}


def stage_summarize(args, out_dir):
  files = sorted(glob.glob(os.path.join(out_dir, "maps", "*.npz")))
  if not files:
    print("no maps yet")
    return
  maps, grp, special = [], [], []
  for path in files:
    d = np.load(path)
    image_id = int(os.path.basename(path)[:-4])
    g = groups(d, args)
    special.append({"image_id": image_id, **{f"n_{k}": int(v.sum()) for k, v in g.items()},
                    "n_tokens": len(d["coverage"])})
    for ri, r in enumerate(d["readouts"]):
      for li, l in enumerate(d["layers"]):
        s = d["scores"][ri, li].astype(np.float32)
        maps.append({"image_id": image_id, "readout": r, "layer": int(l),
                     **map_metrics(s, d["coverage"], int(d["label"]))})
        names = s.argmax(1) == int(d["label"])
        for name, m in g.items():
          if m.any():
            grp.append({"image_id": image_id, "readout": r, "layer": int(l), "group": name,
                        "names_object": float(names[m].mean()), "n": int(m.sum())})
  maps, grp, special = pd.DataFrame(maps), pd.DataFrame(grp), pd.DataFrame(special)
  maps.to_csv(os.path.join(out_dir, "maps.csv"), index=False)
  grp.to_csv(os.path.join(out_dir, "groups.csv"), index=False)
  special.to_csv(os.path.join(out_dir, "special.csv"), index=False)

  pd.set_option("display.width", 200)
  print(f"\n{len(files)} images. Map quality (mean over images):")
  print(maps.groupby(["readout", "layer"])[["recall", "precision", "iou", "region_top1"]]
        .mean().round(2).to_string())
  share = special.filter(like="n_").drop(columns="n_tokens").div(special["n_tokens"], axis=0)
  print("\nShare of an image's tokens in each group (mean over images):")
  print(share.mean().round(3).to_string())
  print("\nShare of each group's tokens whose top-1 is the image's object (mean over images "
        "that have the group):")
  print(grp.groupby(["group", "readout", "layer"])["names_object"].mean().unstack("layer")
        .reindex(GROUPS, level=0).round(2).to_string())
  print(f"\nAll outputs in {out_dir}")


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "token_maps")
  os.makedirs(out_dir, exist_ok=True)
  if args.stage in ("all", "run"):
    if not args.coco_dir:
      raise SystemExit("--coco_dir is needed to run")
    data, kept = load_benchmark(args.vtr_dir)
    stage_run(args, out_dir, data, kept)
  if args.stage in ("all", "summarize"):
    stage_summarize(args, out_dir)


if __name__ == "__main__":
  main()
