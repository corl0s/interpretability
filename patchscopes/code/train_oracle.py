r"""Pilot: a self-supervised visual activation oracle for LLaVA-1.5 visual tokens (vtr/oracle.py).

Stages (each checkpointed in --out_dir; re-running resumes):
  data   COCO val2017 images NOT in the benchmark (samples.json). For each image: the states of
         --tokens_per_image random visual tokens at --layers, and LLaVA's own answer about the
         token's region, shown as --label_view (vtr.oracle.make_view): box3 = red box around the
         3x3 patch block (the pilot), crop7 = only the 7x7 block, enlarged. The answer is the
         training target; no annotation is read.                          -> data/shard_*.pt
  train  LoRA on the attention projections. Input: the neutral-image prompt of image-slot
         injection, the norm-matched state in all 576 image slots at --target_layer, the
         question "What is the main object in this image? Answer with one word."; loss on the
         answer tokens only. Answers naming one of --n_holdout benchmark classes are dropped
         (held-out classes test open-vocabulary generalisation).           -> lora.pt
  eval   the run_injection_proto rows (200 benchmark images), same first-token PMI scoring and
         dev/test split, for the oracle and for the same model with LoRA switched off.
                                                                         -> eval/proto_best.csv
         Also split by seen / held-out class and by easy / hard row: hard = the token's class is
         not the image's main object (the plain model's whole-image answer), so naming the main
         object does not score there. The linear probe is scored on the same split.

  python train_oracle.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
      --vtr_dir ./results/vtr_llava15                                    # all stages
  python train_oracle.py ... --stage eval                                # one stage
  python train_oracle.py ... --label_view crop7      # better labels, in <vtr_dir>/oracle_crop7
  python train_oracle.py ... --label_view crop7 --inject add \
      --data_dir <vtr_dir>/oracle_crop7/data --out_dir <vtr_dir>/oracle_crop7_add   # a variant
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_injection_proto  # noqa: E402
import run_vtr  # noqa: E402
from fit_rewind_maps import pick_images  # noqa: E402
from run_ps_sweep import select_rows  # noqa: E402
from visual_object_identification import ensure_image  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.injection import IMAGE_QUESTION, InjectionReadout  # noqa: E402
from vtr.models import capture_states  # noqa: E402
from vtr.oracle import (BAD_ANSWERS, add_lora, ask, class_scores, first_tokens,  # noqa: E402
                        lora_parameters, make_view, set_lora, slot_injection, square_crop)
from vtr.readouts import class_forms  # noqa: E402


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "data", "train", "eval"])
  p.add_argument("--coco_dir", required=True)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None,
                 help="default: <vtr_dir>/oracle_pilot for box3, else <vtr_dir>/oracle_<view>")
  p.add_argument("--data_dir", default=None,
                 help="training shards (default <out_dir>/data); point at another run's data to "
                      "train a variant on the same labels")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  # data
  p.add_argument("--n_images", type=int, default=2000)
  p.add_argument("--tokens_per_image", type=int, default=8)
  p.add_argument("--layers", default="0,4,8,12,16,20,24,28", help="source layers stored")
  p.add_argument("--label_view", default="box3", help="boxK | cropK | maskK (vtr.oracle.make_view)")
  p.add_argument("--shard_size", type=int, default=100)
  p.add_argument("--gen_batch", type=int, default=16)
  # train
  p.add_argument("--target_layer", type=int, default=4)
  p.add_argument("--no_norm_match", action="store_true")
  p.add_argument("--inject", default="replace", choices=["replace", "add"],
                 help="replace the slot states with the state, or add it to them")
  p.add_argument("--n_holdout", type=int, default=10, help="benchmark classes kept out of training")
  p.add_argument("--train_examples", type=int, default=20000)
  p.add_argument("--micro_batch", type=int, default=4)
  p.add_argument("--accum", type=int, default=2)
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--warmup", type=int, default=50)
  p.add_argument("--rank", type=int, default=16)
  p.add_argument("--lora_alpha", type=float, default=32)
  p.add_argument("--save_every", type=int, default=100, help="optimizer steps")
  p.add_argument("--no_grad_ckpt", action="store_true")
  # eval
  p.add_argument("--eval_images", type=int, default=200)
  p.add_argument("--rows_per_batch", type=int, default=16)
  p.add_argument("--n_boot", type=int, default=500)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


def load_model(args):
  from general_utils import ModelAndTokenizer
  dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
  print(f"Loading {args.model_name} ({dtype}) ...")
  return ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)


def load_benchmark(vtr_dir):
  with open(os.path.join(vtr_dir, "samples.json"), encoding="utf-8") as f:
    data = json.load(f)
  path = os.path.join(vtr_dir, "fair_classes.json")
  kept = data["class_names"]
  if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
      kept = json.load(f)["kept"]
  return data, kept


# ---------------------------------------------------------------------------------- data


def check_label_view(data_dir, view):
  """All shards of one directory share a label view (pilot shards predate the file: box3)."""
  path = os.path.join(data_dir, "label_view.json")
  if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
      have = json.load(f)
  elif any(f.endswith(".pt") for f in os.listdir(data_dir)):
    have = "box3"
  else:
    have = view
  if have != view:
    raise ValueError(f"{data_dir} holds {have} labels, not {view}: use another --out_dir")
  with open(path, "w", encoding="utf-8") as f:
    json.dump(view, f)


def stage_data(args, mt, out_dir, data_dir):
  layers = [int(x) for x in args.layers.split(",")]
  geometry = run_vtr.make_geometry(SimpleNamespace(geometry=None, model_name=args.model_name))
  images = pick_images(args.coco_dir, args.vtr_dir, args.n_images, args.seed)
  os.makedirs(data_dir, exist_ok=True)
  check_label_view(data_dir, args.label_view)
  with open(os.path.join(out_dir, "data_images.json"), "w", encoding="utf-8") as f:
    json.dump([im["id"] for im in images], f)
  t0 = time.time()
  for k in range(0, len(images), args.shard_size):
    path = os.path.join(data_dir, f"shard_{k // args.shard_size:04d}.pt")
    if os.path.exists(path):
      continue
    rng = np.random.RandomState(args.seed * 100003 + k)
    states, answers, image_ids, tokens, views = [], [], [], [], []
    for i, im in enumerate(images[k:k + args.shard_size]):
      image = square_crop(Image.open(ensure_image(args.coco_dir, im["file_name"])).convert("RGB"))
      full = capture_states(mt, mt.processor, image, geometry, verify=(k == 0 and i == 0))
      toks = rng.choice(full.shape[1], args.tokens_per_image, replace=False)
      states.append(full[layers][:, toks].transpose(0, 1).contiguous())  # [n, n_layers, d]
      views += [make_view(image, t, args.label_view) for t in toks]
      image_ids += [im["id"]] * len(toks)
      tokens += [int(t) for t in toks]
    for b in range(0, len(views), args.gen_batch):
      chunk = views[b:b + args.gen_batch]
      answers += ask(mt, [v[0] for v in chunk], chunk[0][1])[0]
    torch.save({"states": torch.cat(states), "answers": answers, "image_id": image_ids,
                "token": tokens, "layers": layers, "label_view": args.label_view},
               path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"  data: images {k + len(images[k:k + args.shard_size])}/{len(images)} "
          f"({time.time() - t0:.0f}s); e.g. {answers[:6]}", flush=True)


def load_data(data_dir):
  shards = [torch.load(os.path.join(data_dir, f), weights_only=False)
            for f in sorted(os.listdir(data_dir)) if f.endswith(".pt")]
  if not shards:
    raise FileNotFoundError(f"no shards in {data_dir}: run --stage data first")
  return {"states": torch.cat([s["states"] for s in shards]),
          "answers": [a for s in shards for a in s["answers"]],
          "image_id": [i for s in shards for i in s["image_id"]],
          "layers": shards[0]["layers"]}


# ---------------------------------------------------------------------------------- train


def holdout_classes(args, kept):
  rng = np.random.RandomState(args.seed)
  return sorted(rng.choice(kept, min(args.n_holdout, len(kept)), replace=False).tolist())


def names_class(answer, forms):
  cands = {answer, answer.rstrip("s"), answer + "s", answer.split()[-1] if answer else ""}
  return any(f.lower() in cands for f in forms)


def training_pool(args, data, held, all_classes):
  forms = class_forms(all_classes, synonyms=True)
  held_forms = [f for c in held for f in forms[c]]
  ok, n_bad, n_held = [], 0, 0
  for i, a in enumerate(data["answers"]):
    if a in BAD_ANSWERS:
      n_bad += 1
    elif names_class(a, held_forms):
      n_held += 1
    else:
      ok.append(i)
  counts = pd.Series([data["answers"][i] for i in ok]).value_counts()
  print(f"  {len(data['answers'])} tokens: {n_bad} unusable answers, {n_held} name a held-out "
        f"class, {len(ok)} kept; {len(counts)} distinct answers; top: "
        f"{counts.head(12).to_dict()}")
  return np.array(ok)


def answer_ids(tok, prompt, answer):
  base = tok(prompt)["input_ids"]
  full = tok(prompt + " " + answer)["input_ids"]
  if full[:len(base)] != base:
    return None
  return full[len(base):] + [tok.eos_token_id]


def stage_train(args, mt, out_dir, data_dir, class_names, kept):
  final = os.path.join(out_dir, "lora.pt")
  if os.path.exists(final):
    print(f"  {final} exists: training done")
    return
  data = load_data(data_dir)
  held = holdout_classes(args, kept)
  with open(os.path.join(out_dir, "holdout.json"), "w", encoding="utf-8") as f:
    json.dump(held, f)
  print(f"  held-out classes: {held}")
  pool = training_pool(args, data, held, class_names)
  n_layers = len(data["layers"])
  pairs = np.stack(np.meshgrid(pool, np.arange(n_layers), indexing="ij"), -1).reshape(-1, 2)
  pairs = pairs[np.random.RandomState(args.seed).permutation(len(pairs))][:args.train_examples]
  per_step = args.micro_batch * args.accum
  n_steps = len(pairs) // per_step
  print(f"  {len(pairs)} (token, layer) examples, {n_steps} optimizer steps")

  # Neutral-image prompt, slots and reference norms from the model without LoRA.
  ro = InjectionReadout(mt, class_names, class_forms(class_names, synonyms=False), "image",
                        args.device, IMAGE_QUESTION, rows_per_batch=1)
  ref_norm = ro.ref_norm.clone()
  base_ids = ro.inputs["input_ids"][0]
  pixel = ro.inputs["pixel_values"]
  P = len(base_ids)
  tok = mt.tokenizer
  prompt = f"USER: <image>\n{IMAGE_QUESTION} ASSISTANT:"
  conts = {}

  params = add_lora(mt, args.rank, args.lora_alpha)
  opt = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=0.0)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
  ckpt_path = os.path.join(out_dir, "train_ckpt.pt")
  start, log = 0, []
  if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, weights_only=False, map_location=args.device)
    for n, p in params.items():
      p.data.copy_(ck["lora"][n])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    start, log = ck["step"], ck["log"]
    print(f"  resuming at step {start}")

  if not args.no_grad_ckpt:
    mt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
  mt.model.train()
  pad = tok.pad_token_id if tok.pad_token_id is not None else 0

  def save(step):
    torch.save({"lora": {n: p.detach().cpu() for n, p in params.items()},
                "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "log": log},
               ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)

  t0, running = time.time(), []
  for step in range(start, n_steps):
    for m in range(args.accum):
      chunk = pairs[(step * args.accum + m) * args.micro_batch:][:args.micro_batch]
      ids, h = [], []
      for ti, li in chunk:
        a = data["answers"][ti]
        if a not in conts:
          conts[a] = answer_ids(tok, prompt, a)
        if conts[a] is not None:
          ids.append(conts[a])
          h.append(data["states"][ti, li])
      if not ids:
        continue
      L = max(len(x) for x in ids)
      B = len(ids)
      cont = torch.full((B, L), pad, dtype=torch.long)
      labels = torch.full((B, L), -100, dtype=torch.long)
      for b, x in enumerate(ids):
        cont[b, :len(x)] = torch.tensor(x)
        labels[b, :len(x)] = torch.tensor(x)
      input_ids = torch.cat([base_ids.expand(B, -1), cont], 1).to(args.device)
      attn = torch.cat([torch.ones(B, P, dtype=torch.long), (labels != -100).long()], 1)
      h = torch.stack(h).to(args.device).float()
      if not args.no_norm_match:
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6) * float(ref_norm[args.target_layer])
      # backward() inside the hook's scope: gradient checkpointing re-runs the injection layer
      with slot_injection(mt, args.target_layer, ro.pos, h, add=args.inject == "add"):
        out = mt.model(input_ids=input_ids, attention_mask=attn.to(args.device),
                       pixel_values=pixel.expand(B, -1, -1, -1).to(args.device),
                       use_cache=False, logits_to_keep=L + 1)
        logits = out.logits[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.to(args.device).reshape(-1), ignore_index=-100)
        (loss / args.accum).backward()
      running.append(loss.item())
    torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    if (step + 1) % 20 == 0 or step == n_steps - 1:
      log.append({"step": step + 1, "loss": float(np.mean(running))})
      print(f"  step {step + 1}/{n_steps}: loss {np.mean(running):.3f} "
            f"({time.time() - t0:.0f}s)", flush=True)
      running = []
    if (step + 1) % args.save_every == 0:
      save(step + 1)
  mt.model.eval()
  if not args.no_grad_ckpt:
    mt.model.gradient_checkpointing_disable()
  torch.save({"lora": {n: p.detach().cpu() for n, p in params.items()}, "ref_norm": ref_norm,
              "config": vars(args), "log": log}, final)
  pd.DataFrame(log).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
  if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
  print(f"  saved {final}")


# ---------------------------------------------------------------------------------- eval


def stage_eval(args, mt, out_dir, data, kept):
  blob = torch.load(os.path.join(out_dir, "lora.pt"), weights_only=False)
  cfg = blob["config"]
  t, norm_match = cfg["target_layer"], not cfg["no_norm_match"]
  add = cfg.get("inject", "replace") == "add"
  if not lora_parameters(mt):
    add_lora(mt, cfg["rank"], cfg["lora_alpha"])
  params = lora_parameters(mt)
  for n, p in params.items():
    p.data.copy_(blob["lora"][n].to(p.device))
  mt.model.eval()
  class_names = data["class_names"]
  sources = [int(x) for x in cfg["layers"].split(",")]
  cache = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)
  states, meta = cache["states"], cache["meta"]
  rows = select_rows(meta, args.eval_images, args.seed)
  meta_rows = meta.iloc[rows].reset_index(drop=True)
  eval_dir = os.path.join(out_dir, "eval")
  forms = class_forms(class_names, synonyms=False)
  variants = {"untrained": False, "oracle": True}
  for name, enabled in variants.items():
    cell_dir = os.path.join(eval_dir, "cells", name)
    os.makedirs(cell_dir, exist_ok=True)
    todo = [s for s in sources if not os.path.exists(os.path.join(cell_dir, f"s{s}_t{t}.npz"))]
    if not todo:
      continue
    set_lora(mt, enabled)
    ro = InjectionReadout(mt, class_names, forms, "image", args.device, IMAGE_QUESTION,
                          args.rows_per_batch)
    ro.ref_norm = blob["ref_norm"]  # the norms the oracle was trained with
    positions = [ro.slots("all")] * len(rows)
    for s in todo:
      t0 = time.time()
      raw, pmi = ro.scores(states[rows, s].float(), t, positions, norm_match, add=add)
      np.savez(os.path.join(cell_dir, f"s{s}_t{t}.npz"), raw=raw.astype(np.float16),
               pmi=pmi.astype(np.float16))
      print(f"  eval {name}: source {s:2d} -> target {t} ({time.time() - t0:.1f}s)", flush=True)
  set_lora(mt, True)

  keep = [class_names.index(c) for c in kept]
  summ = SimpleNamespace(variants=list(variants), out_dir=eval_dir, dev_fraction=0.3,
                         seed=args.seed, n_boot=args.n_boot)
  best = run_injection_proto.summarize(summ, meta_rows, class_names, sources, [t], keep)
  table = best.pivot(index="variant", columns="source", values="test_object")
  probe = run_injection_proto.probe_reference(args.vtr_dir, sources)
  if probe is not None:
    table.loc["probe (reference)"] = probe.values
  table["mean"] = table.mean(axis=1)
  pd.set_option("display.width", 200)
  print(f"\nTop-1, object tokens, test images, target layer {t} (fair {len(keep)} classes):")
  print(table.round(3).to_string())
  print("\nRandom-vector control (test images):")
  print(best.pivot(index="variant", columns="source", values="test_random").round(3).to_string())
  print("\n95% CI of the oracle (test images):")
  o = best[best.variant == "oracle"].set_index("source")
  print((o["test_ci_lo"].round(3).astype(str) + "-" + o["test_ci_hi"].round(3).astype(str))
        .to_frame("ci").T.to_string())
  set_lora(mt, False)  # the main object is the plain model's whole-image answer
  main = main_objects(args, mt, data, meta_rows["image_id"].unique(), kept, eval_dir)
  set_lora(mt, True)
  split_tables(args, rows, meta_rows, class_names, sources, t, keep, list(variants), eval_dir,
               os.path.join(out_dir, "holdout.json"), main)


def main_objects(args, mt, data, image_ids, kept, eval_dir, batch=16):
  """{image_id: fair class LLaVA names as the main object of the whole image} (closed top-1)."""
  path = os.path.join(eval_dir, "main_object.csv")
  if os.path.exists(path):
    df = pd.read_csv(path, keep_default_na=False)
  else:
    files = {s["image_id"]: s["file_name"] for s in data["samples"]}
    cls, ids = first_tokens(mt.tokenizer, kept, class_forms(kept, synonyms=False))
    ids_list, preds, answers = list(image_ids), [], []
    for b in range(0, len(ids_list), batch):
      views = [make_view(square_crop(Image.open(ensure_image(args.coco_dir, files[i]))
                                     .convert("RGB")), 0, "full") for i in ids_list[b:b + batch]]
      a, logp = ask(mt, [v[0] for v in views], views[0][1])
      answers += a
      preds += [kept[k] for k in class_scores(logp, cls, ids, len(kept)).argmax(1)]
    df = pd.DataFrame({"image_id": ids_list, "main_object": preds, "answer": answers})
    df.to_csv(path, index=False)
  return dict(zip(df["image_id"], df["main_object"]))


def split_tables(args, rows, meta_rows, class_names, sources, t, keep, variants, eval_dir,
                 holdout_path, main):
  """Top-1 on object rows of test images, split by seen / held-out class (was the class kept out
  of training?) and by easy / hard row (hard: the class is not the image's main object)."""
  with open(holdout_path, encoding="utf-8") as f:
    held = set(json.load(f))
  index = {c: i for i, c in enumerate(class_names)}
  labels = meta_rows["class_name"].map(index).to_numpy()
  ok = np.isin(labels, keep)
  remap = {c: i for i, c in enumerate(keep)}
  m = meta_rows[ok].reset_index(drop=True)
  lab = np.array([remap[x] for x in labels[ok]])
  images = m["image_id"].drop_duplicates().sample(frac=1.0, random_state=args.seed).to_numpy()
  dev = set(images[:int(round(0.3 * len(images)))])
  test_obj = (~m["image_id"].isin(dev) & (m["kind"] == "object")).to_numpy()
  is_held = m["class_name"].isin(held).to_numpy()
  is_hard = (m["class_name"] != m["image_id"].map(main)).to_numpy()

  top1 = {}
  for v in variants:
    for s in sources:
      cell = np.load(os.path.join(eval_dir, "cells", v, f"s{s}_t{t}.npz"))["pmi"]
      top1[(v, s)] = E.ranks(cell.astype(np.float32)[ok][:, keep], lab) == 1
  probe_path = os.path.join(args.vtr_dir, "scores", "probe_linear.npz")
  if os.path.exists(probe_path):
    layers, scores = run_vtr.load_scores(probe_path)
    for s in sources:
      if s in layers:
        top1[("probe", s)] = E.ranks(scores[layers.index(s)][rows][ok][:, keep], lab) == 1

  splits = {"heldout_split": [("seen", ~is_held), ("held-out", is_held)],
            "hard_split": [("easy", ~is_hard), ("hard", is_hard)]}
  for name, groups in splits.items():
    out = [{"variant": v, "group": g, "source": s, "n": int((test_obj & mask).sum()),
            "top1": hit[test_obj & mask].mean() if (test_obj & mask).any() else np.nan}
           for (v, s), hit in top1.items() for g, mask in groups
           if name == "hard_split" or v != "probe"]
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(eval_dir, f"{name}.csv"), index=False)
    table = df.pivot_table(index=["variant", "group"], columns="source", values="top1")
    table["mean"] = table.mean(axis=1)
    n = df.groupby("group")["n"].first().to_dict()
    if name == "heldout_split":
      print(f"\nSeen vs held-out classes (test object rows: {n}; held-out: {sorted(held)}):")
    else:
      print(f"\nEasy vs hard rows (test object rows: {n}; hard = the class is not the main "
            f"object LLaVA names for the whole image, so naming the main object scores 0):")
    print(table.round(3).to_string())


def main(argv=None, mt=None):
  args = parse_args(argv)
  default = "oracle_pilot" if args.label_view == "box3" else f"oracle_{args.label_view}"
  out_dir = args.out_dir or os.path.join(args.vtr_dir, default)
  os.makedirs(out_dir, exist_ok=True)
  data_dir = args.data_dir or os.path.join(out_dir, "data")
  data, kept = load_benchmark(args.vtr_dir)
  stages = ["data", "train", "eval"] if args.stage == "all" else [args.stage]
  if mt is None:
    mt = load_model(args)
  if "data" in stages:
    print("Stage data")
    stage_data(args, mt, out_dir, data_dir)
  if "train" in stages:
    print("Stage train")
    stage_train(args, mt, out_dir, data_dir, data["class_names"], kept)
  if "eval" in stages:
    print("Stage eval")
    stage_eval(args, mt, out_dir, data, kept)
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
