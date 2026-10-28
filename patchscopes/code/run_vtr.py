r"""COCO-VTR Milestone 1: every readout on LLaVA-1.5-7B, object-identity task, closed-set.

Stages (each checkpointed in --out_dir; re-runs skip finished stages unless --overwrite):
  data      select COCO val2017 images / objects, per-token coverage    -> samples.json
  cache     hidden states of sampled tokens + pooled + random, all layers -> states.pt
  readouts  scores over the 80 COCO classes for each readout and layer  -> scores/*.npz
  eval      accuracies per layer / condition, image-clustered CIs, plots -> summary.csv ...
            plus the same on the "fair" class subset (every training-free readout can
            score and tell the classes apart)                        -> *_fair.*, fair_classes.json

Readout names in the outputs:
  logit_lens, logit_lens_syn         class name only / with unambiguous synonyms
  tuned_lens                         logit lens after learned per-layer translators
                                     (train them first with train_tuned_lens.py)
  embedding_lens                     nearest input embedding (LatentLens's EmbeddingLens)
  latentlens, latentlens_syn         same two variants (needs a LatentLens bank for LLaVA)
  patchscopes_raw, patchscopes_pmi   continuation log-likelihood; PMI subtracts the prompt prior
  selfie_raw, selfie_pmi             the same with the SelfIE-style prompt (5 placeholders)
  *_t<L>_raw, *_t<L>_pmi             Patchscopes / SelfIE injected at fixed target layer L
                                     (--ps_target_layer L) instead of the source layer
  image_slot_raw, image_slot_pmi     image-slot injection (vtr/injection.py): the state in all
                                     576 slots of a blank-image prompt; first-token scoring.
                                     Target per source layer from --slot_target (LLaVA only)
  oracle_raw, oracle_pmi             the visual activation oracle (train_oracle.py): image-slot
                                     injection with its trained LoRA, settings from its lora.pt
  slot_nn_<bank>_img_k<K>            image-slot injection read by LatentLens: the second pass's
  slot_nn_<bank>_ans_k<K>            states at block target + K, averaged over the image slots
                                     (img) or at the answer position (ans), nearest neighbours in
                                     the bank (<bank> = --latentlens_name; offsets --nn_offsets)
  probe_linear, probe_mlp            supervised references (availability)

  --readouts picks which to compute: logit_lens tuned_lens embedding_lens latentlens
  patchscopes selfie image_slot oracle slot_nn probe (finished ones are skipped, so new ones can be
  added later). image_slot and oracle are not in the fair-subset definition (TRAINING_FREE),
  so adding them leaves the 63 fair classes and the other readouts' numbers unchanged.

Usage on SCC (one GPU):
  # smoke run
  python run_vtr.py --coco_dir /projectnb/mlresearch/vishnuav/coco --out_dir ./results/vtr_smoke \
      --n_images 20 --layers 0,2,8,16,24,31 --n_boot 200
  # full run
  python run_vtr.py --coco_dir /projectnb/mlresearch/vishnuav/coco
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vtr import evaluate as E  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples, select_same_images, summarize_samples  # noqa: E402
from vtr.geometry import GEOMETRIES  # noqa: E402
from vtr.readouts import (EmbeddingLens, LatentLens, LogitLens, Patchscopes,  # noqa: E402
                          TunedLens, class_forms, probe_scores)
from visual_object_identification import get_decoder  # noqa: E402

MAIN_READOUTS = ["logit_lens", "tuned_lens", "embedding_lens", "latentlens", "patchscopes_pmi",
                 "selfie_pmi", "image_slot_pmi", "oracle_pmi", "probe_linear"]
# Readouts whose class coverage defines the fair subset (the probe is a supervised reference;
# the tuned lens is trained on text only, never on images, so it counts as a readout here).
# The *_pmi variants are the main injection numbers, fixed before looking at results.
TRAINING_FREE = ["logit_lens", "tuned_lens", "embedding_lens", "latentlens", "patchscopes_pmi",
                 "selfie_pmi"]


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "data", "cache", "readouts", "eval"])
  p.add_argument("--coco_dir", default=None)
  p.add_argument("--out_dir", default="./results/vtr_llava15")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--geometry", default=None, choices=sorted(GEOMETRIES),
                 help="token geometry; default: inferred from --model_name")
  p.add_argument("--min_pixels", type=int, default=56 * 56, help="Qwen2.5-VL resize lower bound")
  p.add_argument("--max_pixels", type=int, default=12845056,
                 help="Qwen2.5-VL resize upper bound (lower it to test fewer, larger tokens)")
  p.add_argument("--same_images", default=None,
                 help="samples.json of another run: reuse its images and objects")
  # data
  p.add_argument("--n_images", type=int, default=1000)
  p.add_argument("--per_class", type=int, default=20, help="max images per class")
  p.add_argument("--min_area", type=float, default=4000)
  p.add_argument("--max_area", type=float, default=0, help="0 = no upper bound")
  p.add_argument("--min_tokens", type=int, default=4, help="tokens with coverage >= 0.5")
  # cache
  p.add_argument("--per_bin", type=int, default=2, help="object tokens per overlap bin")
  p.add_argument("--n_outside", type=int, default=2)
  # readouts
  p.add_argument("--readouts", nargs="+", default=["logit_lens", "latentlens", "patchscopes",
                                                   "probe"],
                 choices=["logit_lens", "tuned_lens", "embedding_lens", "latentlens",
                          "patchscopes", "selfie", "image_slot", "oracle", "slot_nn", "probe"])
  p.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
  p.add_argument("--latentlens_bank", default="./results/latentlens_index/bank")
  p.add_argument("--latentlens_name", default="latentlens",
                 help="output name of the LatentLens readout (and the slot_nn_<name>_* outputs), "
                      "so a run on another bank does not overwrite the default one")
  p.add_argument("--nn_offsets", default="0,2,4,8,16",
                 help="slot_nn: read the second pass at target + each offset")
  p.add_argument("--tuned_lens", default="./results/tuned_lens/llava15_tuned_lens.pt",
                 help="translators from train_tuned_lens.py")
  p.add_argument("--ps_prompt", default="identity", choices=["identity", "entity"])
  p.add_argument("--ps_synonyms", action="store_true",
                 help="also score synonyms in Patchscopes (about 2x slower)")
  p.add_argument("--ps_max_images", type=int, default=0,
                 help="0 = all images (applies to every injection readout)")
  p.add_argument("--ps_rows_per_batch", type=int, default=256)
  p.add_argument("--ps_target_layer", type=int, default=None,
                 help="inject Patchscopes / SelfIE states at this target layer instead of the "
                      "source layer (outputs are named e.g. patchscopes_t0_pmi)")
  p.add_argument("--slot_target", default="dev",
                 help="image-slot target layer: an int, or 'dev' = the target chosen on "
                      "development images per source layer (img_all in "
                      "<out_dir>/injection_proto/proto_best.csv; nearest source for other layers)")
  p.add_argument("--slot_rows_per_batch", type=int, default=16)
  p.add_argument("--oracle_dir", default="./results/vtr_llava15/oracle_crop7_add",
                 help="train_oracle.py output with lora.pt")
  p.add_argument("--probe_splits", type=int, default=5)
  # eval
  p.add_argument("--n_boot", type=int, default=2000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--overwrite", action="store_true")
  return p.parse_args(argv)


def load_model(args):
  from general_utils import ModelAndTokenizer
  dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
  if "qwen" in args.model_name.lower():
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
  print(f"Loading {args.model_name} ...")
  mt = ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)
  if getattr(mt, "family", "") == "qwen2.5-vl" and mt.processor is not None:
    mt.processor.set_pixel_bounds(args.min_pixels, args.max_pixels)
  return mt


def make_geometry(args):
  name = args.geometry or ("qwen2.5-vl" if "qwen" in args.model_name.lower() else "llava-1.5")
  if name == "qwen2.5-vl":
    return GEOMETRIES[name](args.min_pixels, args.max_pixels)
  return GEOMETRIES[name]()


def save_scores(path, layers, scores):
  np.savez(path, layers=np.asarray(layers), scores=np.asarray(scores, dtype=np.float16))


def load_scores(path):
  d = np.load(path)
  return list(d["layers"]), d["scores"].astype(np.float32)


def slot_targets(args, layers):
  """{source layer: image-slot target layer}."""
  if args.slot_target != "dev":
    return {l: int(args.slot_target) for l in layers}
  path = os.path.join(args.out_dir, "injection_proto", "proto_best.csv")
  if not os.path.exists(path):
    raise FileNotFoundError(f"{path} not found: run run_injection_proto.py first, or pass "
                            "--slot_target <layer>")
  best = pd.read_csv(path)
  best = best[best["variant"] == "img_all"]
  chosen = dict(zip(best["source"].astype(int), best["target_chosen_on_dev"].astype(int)))
  if not chosen:
    raise ValueError(f"no img_all rows in {path}")
  return {l: chosen[min(chosen, key=lambda s: (abs(s - l), s))] for l in layers}


def run_readouts(args, mt, states, meta, class_names, layers):
  score_dir = os.path.join(args.out_dir, "scores")
  os.makedirs(score_dir, exist_ok=True)

  def done(*names):
    return not args.overwrite and all(
        os.path.exists(os.path.join(score_dir, f"{n}.npz")) for n in names)

  def per_layer(fn):
    out = []
    for l in layers:
      t0 = time.time()
      out.append(fn(states[:, l].float(), l))
      print(f"    layer {l} ({time.time() - t0:.1f}s)")
    return out

  plain = class_forms(class_names, synonyms=False)
  syn = class_forms(class_names, synonyms=True)

  if "logit_lens" in args.readouts:
    for name, forms in (("logit_lens", plain), ("logit_lens_syn", syn)):
      if done(name):
        continue
      print(f"Readout: {name}")
      ro = LogitLens(mt, class_names, forms, args.device)
      save_scores(os.path.join(score_dir, f"{name}.npz"), layers, per_layer(ro.scores))

  if "latentlens" in args.readouts:
    ll = args.latentlens_name
    for name, forms in ((ll, plain), (f"{ll}_syn", syn)):
      if done(name):
        continue
      print(f"Readout: {name} (bank: {args.latentlens_bank})")
      ro = LatentLens(args.latentlens_bank, mt.tokenizer, class_names, forms, args.device,
                      cache_file=os.path.join(score_dir, f"{name}_entries.npz"))
      save_scores(os.path.join(score_dir, f"{name}.npz"), layers, per_layer(ro.scores))
      del ro

  if "tuned_lens" in args.readouts and not done("tuned_lens"):
    if not os.path.exists(args.tuned_lens):
      raise FileNotFoundError(f"{args.tuned_lens} not found: run train_tuned_lens.py first")
    print(f"Readout: tuned_lens ({args.tuned_lens})")
    ro = TunedLens(mt, class_names, plain, args.device, args.tuned_lens)
    save_scores(os.path.join(score_dir, "tuned_lens.npz"), layers, per_layer(ro.scores))
    del ro

  if "embedding_lens" in args.readouts and not done("embedding_lens"):
    print("Readout: embedding_lens")
    ro = EmbeddingLens(mt, class_names, plain, args.device)
    save_scores(os.path.join(score_dir, "embedding_lens.npz"), layers, per_layer(ro.scores))

  def injection(prefix, fn, suffixes=("raw", "pmi")):
    """fn(h [n, d], source layer) -> one [n, n_classes] array per suffix (default (raw, pmi));
    saves <prefix>_<suffix>.npz."""
    rows = np.arange(len(meta))
    if args.ps_max_images:
      keep = meta["image_id"].drop_duplicates().iloc[:args.ps_max_images]
      rows = np.flatnonzero(meta["image_id"].isin(keep).to_numpy())
    out = {s: np.full((len(layers), len(meta), len(class_names)), np.nan, dtype=np.float32)
           for s in suffixes}
    # Layer-level checkpoint, so a job stopped by its time limit resumes where it was.
    partial = os.path.join(score_dir, f"{prefix}_partial.npz")
    finished = []
    if os.path.exists(partial):
      ck = np.load(partial)
      if list(ck["layers"]) == list(layers) and all(s in ck for s in suffixes):
        out, finished = {s: ck[s] for s in suffixes}, list(ck["finished"])
        print(f"    resuming: layers {finished} already done")
    for i, l in enumerate(layers):
      if l in finished:
        continue
      t0 = time.time()
      for s, v in zip(suffixes, fn(states[rows, l].float(), l)):
        out[s][i, rows] = v
      finished.append(l)
      np.savez(partial, layers=np.asarray(layers), finished=np.asarray(finished), **out)
      print(f"    layer {l}: {len(rows)} rows ({time.time() - t0:.1f}s)", flush=True)
    for s in suffixes:
      save_scores(os.path.join(score_dir, f"{prefix}_{s}.npz"), layers, out[s])
    os.remove(partial)

  # Injection readouts: Patchscopes (identity / entity prompt) and the SelfIE-style prompt.
  injections = []
  if "patchscopes" in args.readouts:
    injections.append(("patchscopes", args.ps_prompt))
  if "selfie" in args.readouts:
    injections.append(("selfie", "selfie"))
  for prefix, prompt in injections:
    if args.ps_target_layer is not None:
      prefix = f"{prefix}_t{args.ps_target_layer}"
    if done(f"{prefix}_raw", f"{prefix}_pmi"):
      continue
    print(f"Readout: {prefix} ({prompt} prompt)")
    ro = Patchscopes(mt, class_names, syn if args.ps_synonyms else plain, prompt,
                     args.device, rows_per_batch=args.ps_rows_per_batch)
    injection(prefix, lambda h, l, ro=ro: ro.scores(
        h, l if args.ps_target_layer is None else args.ps_target_layer))

  # Image-slot injection, then the oracle (the same path with its LoRA on).
  if "image_slot" in args.readouts and not done("image_slot_raw", "image_slot_pmi"):
    from vtr.injection import InjectionReadout
    from vtr.oracle import set_lora
    set_lora(mt, False)  # plain model, even if a LoRA is attached
    targets = slot_targets(args, layers)
    print(f"Readout: image_slot (all 576 slots; target per source {targets})")
    ro = InjectionReadout(mt, class_names, plain, "image", args.device,
                          rows_per_batch=args.slot_rows_per_batch)
    injection("image_slot", lambda h, l, ro=ro: ro.scores(
        h, targets[l], [ro.slots("all")] * h.shape[0]))

  if "oracle" in args.readouts and not done("oracle_raw", "oracle_pmi"):
    from vtr.injection import InjectionReadout
    from vtr.oracle import add_lora, lora_parameters, set_lora
    blob = torch.load(os.path.join(args.oracle_dir, "lora.pt"), weights_only=False)
    cfg = blob["config"]
    t, norm_match = cfg["target_layer"], not cfg["no_norm_match"]
    add = cfg.get("inject", "replace") == "add"
    print(f"Readout: oracle ({args.oracle_dir}: target {t}, {'add' if add else 'replace'}, "
          f"norm-matched {norm_match})")
    if not lora_parameters(mt):
      add_lora(mt, cfg["rank"], cfg["lora_alpha"])
    for n, p in lora_parameters(mt).items():
      p.data.copy_(blob["lora"][n].to(p.device))
    mt.model.eval()
    set_lora(mt, True)  # also for the no-injection prior computed in the constructor
    try:
      ro = InjectionReadout(mt, class_names, plain, "image", args.device,
                            rows_per_batch=args.slot_rows_per_batch)
      ro.ref_norm = blob["ref_norm"]  # the norms the oracle was trained with
      injection("oracle", lambda h, l, ro=ro: ro.scores(
          h, t, [ro.slots("all")] * h.shape[0], norm_match, add=add))
    finally:
      set_lora(mt, False)

  # Image-slot injection read by nearest neighbours: inject at the image-slot target t, take the
  # second pass's states at blocks t + k, and score them with LatentLens (cosine to the text
  # bank). Two read-outs per offset: the answer position (ans; k > 0, since the answer position
  # only sees the injection through later attention) and the mean over the injected slots (img;
  # k = 0 is plain LatentLens on h, a sanity check).
  if "slot_nn" in args.readouts:
    from vtr.injection import InjectionReadout
    from vtr.oracle import set_lora
    offsets = [int(k) for k in args.nn_offsets.split(",")]
    suffixes = [f"img_k{k}" for k in offsets] + [f"ans_k{k}" for k in offsets if k > 0]
    prefix = f"slot_nn_{args.latentlens_name}"
    if not done(*(f"{prefix}_{s}" for s in suffixes)):
      set_lora(mt, False)
      targets = slot_targets(args, layers)
      last = len(get_decoder(mt).layers) - 1
      print(f"Readout: {prefix} (target per source {targets}; read at t + {offsets}, capped at "
            f"{last}; bank {args.latentlens_bank})")
      ro = InjectionReadout(mt, class_names, plain, "image", args.device,
                            rows_per_batch=args.slot_rows_per_batch)
      nn = LatentLens(args.latentlens_bank, mt.tokenizer, class_names, plain, args.device,
                      cache_file=os.path.join(score_dir, f"{args.latentlens_name}_entries.npz"))

      def slot_nn(h, l):
        t = targets[l]
        read = {k: min(t + k, last) for k in offsets}
        out = {s: [] for s in suffixes}
        for start in range(0, h.shape[0], ro.batch):
          hb = h[start:start + ro.batch]
          got = ro.read_states(hb, [ro.slots("all")] * hb.shape[0], t, set(read.values()))
          for k, r in read.items():
            ans, img = got[r]
            out[f"img_k{k}"].append(nn.scores(img))
            if k > 0:
              out[f"ans_k{k}"].append(nn.scores(ans))
        return [np.concatenate(out[s]) for s in suffixes]

      injection(prefix, slot_nn, suffixes)
      del nn

  if "probe" in args.readouts:
    for kind in ("linear", "mlp"):
      name = f"probe_{kind}"
      if done(name):
        continue
      print(f"Readout: {name}")
      out = per_layer(lambda h, l: probe_scores(h.numpy(), meta, class_names, kind,
                                                args.probe_splits, args.seed))
      save_scores(os.path.join(score_dir, f"{name}.npz"), layers, out)


def main(argv=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  geometry = make_geometry(args)
  stages = ["data", "cache", "readouts", "eval"] if args.stage == "all" else [args.stage]

  # ------------------------------------------------------------------ data
  samples_path = os.path.join(args.out_dir, "samples.json")
  if "data" in stages and (args.overwrite or not os.path.exists(samples_path)):
    from visual_object_identification import ensure_annotations
    print("Selecting COCO-VTR object samples ...")
    if args.same_images:
      with open(args.same_images, encoding="utf-8") as f:
        previous = json.load(f)["samples"]
      samples, class_names = select_same_images(previous, ensure_annotations(args.coco_dir),
                                                geometry, args.min_tokens)
      print(f"  reused {len(samples)} of {len(previous)} images from {args.same_images}")
    else:
      samples, class_names = select_object_samples(
          ensure_annotations(args.coco_dir), geometry, args.n_images, args.per_class,
          args.min_area, args.max_area or None, args.min_tokens, args.seed)
    with open(samples_path, "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "geometry": geometry.name, "samples": samples}, f)
    summary = summarize_samples(samples)
    with open(os.path.join(args.out_dir, "data_summary.json"), "w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
  with open(samples_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  samples, class_names = data["samples"], data["class_names"]

  # ------------------------------------------------------------------ cache
  mt = None
  cache_path = os.path.join(args.out_dir, "states.pt")
  if "cache" in stages and (args.overwrite or not os.path.exists(cache_path)):
    mt = load_model(args)
    print(f"Caching hidden states for {len(samples)} images ...")
    states, meta = cache_states(mt, mt.processor, samples, args.coco_dir, args.per_bin,
                                args.n_outside, args.seed, geometry)
    torch.save({"states": states, "meta": meta}, cache_path)
  if not os.path.exists(cache_path):
    return
  cache = torch.load(cache_path, weights_only=False)
  states, meta = cache["states"], cache["meta"]
  n_layers = states.shape[1]
  layers = list(range(n_layers)) if args.layers == "all" else \
      [int(x) for x in args.layers.split(",")]
  print(f"cache: {tuple(states.shape)}; rows by kind: {meta['kind'].value_counts().to_dict()}")

  # ------------------------------------------------------------------ readouts
  if "readouts" in stages:
    if mt is None and any(r != "probe" for r in args.readouts):
      mt = load_model(args)
    run_readouts(args, mt, states, meta, class_names, layers)

  # ------------------------------------------------------------------ eval
  if "eval" in stages:
    score_dir = os.path.join(args.out_dir, "scores")
    scores = {}
    for fname in sorted(os.listdir(score_dir)):
      if fname.endswith(".npz") and not fname.endswith("_entries.npz"):
        ls, s = load_scores(os.path.join(score_dir, fname))
        scores[fname[:-4]] = (ls, s)
    print(f"Evaluating {sorted(scores)} on all {len(class_names)} classes ...")
    summary = E.evaluate(scores, meta, class_names, args.n_boot, args.seed)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    head = E.headline(summary, len(class_names))
    head.to_csv(os.path.join(args.out_dir, "headline.csv"), index=False)
    print(head.to_string(index=False))
    main_present = [r for r in MAIN_READOUTS if r in scores]
    E.plot(summary, args.out_dir, main_present)

    # Fair comparison: only classes every training-free readout can score and tell apart,
    # ranked among those classes only (the probe is evaluated on the same subset).
    keep, dropped = E.fair_classes(scores, [r for r in TRAINING_FREE if r in scores],
                                   class_names)
    with open(os.path.join(args.out_dir, "fair_classes.json"), "w", encoding="utf-8") as f:
      json.dump({"kept": [class_names[i] for i in keep], "dropped": dropped}, f, indent=2)
    print(f"\nFair subset: {len(keep)} of {len(class_names)} classes "
          f"(dropped: {', '.join(sorted(dropped)) or 'none'})")
    if len(keep) >= 2:
      sub_scores, sub_meta, sub_names = E.subset(scores, meta, class_names, keep)
      fair = E.evaluate(sub_scores, sub_meta, sub_names, args.n_boot, args.seed)
      fair.to_csv(os.path.join(args.out_dir, "summary_fair.csv"), index=False)
      head_fair = E.headline(fair, len(sub_names))
      head_fair.to_csv(os.path.join(args.out_dir, "headline_fair.csv"), index=False)
      print(f"{sub_meta['image_id'].nunique()} images, {len(sub_meta)} rows")
      print(head_fair.to_string(index=False))
      E.plot(fair, args.out_dir, main_present, prefix="fair_")
    print(f"All outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
