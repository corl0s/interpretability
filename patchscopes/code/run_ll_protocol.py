r"""LatentLens evaluation protocol (Krojer et al., ICML 2026, arXiv 2602.00462) on LLaVA-1.5,
for their three lenses and for our readouts, so our numbers sit next to theirs.

Their protocol, as implemented in github.com/McGill-NLP/latentlens (cloned to --ll_repo):
  * PixMo-Cap validation images (their split: train_test_split(test_size=2048, seed=96817))
  * the prompt "Describe this image in detail.", no system prompt
  * layers 0, 1, 2, 4, 8, 16, 24, 30, 31, where layer l is hidden_states[l] (the INPUT of block
    l; 0 = projector output). Note: COCO-VTR's layer k is the OUTPUT of block k, i.e. l = k + 1.
  * every readout gives top-5 words per visual token; a judge sees the image (centre-cropped to
    512 px for LLaVA) with a red 3x3-patch box and the crop, and decides whether any word is
    related (concrete / abstract / global). Interpretable = at least one related word.
  * per layer, 100 patches: one random 3x3 block per image (random.seed(42) once, then layer by
    layer), the token judged is the block's centre. So patches differ across layers but are the
    same for every readout at a given layer.

Stages (run in this order; see job_ll_*.sh):
  fetch       download the first --n_images reachable images of their validation split
  extract     LLaVA forward pass: cache states at their layers; write logit-lens and
              embedding-lens results in their JSON format
  export_lm   save LLaVA's own language model as a standalone LlamaForCausalLM, for their
              extract_embeddings.py (the LatentLens text bank)
  latentlens  search the text bank (their recipe: cosine, top-5 over all bank layers jointly)
  plan        replay their judge's patch sampling -> which (layer, image, patch) gets judged
  ours        our readouts on the planned patches only: Patchscopes, SelfIE, image-slot
              injection, visual oracle (top-5 next tokens, raw, like their logit lens)
  judge       their evaluate_interpretability.py, with an open-weight judge behind an
              OpenAI-compatible server (vLLM) instead of GPT-5
  rejudge     judge again only the patches whose reply did not parse (see make_judgment)
  summary     % interpretable per readout and layer (+ Wilson 95% CI, relation types)

Deviations from the paper (state them when reporting):
  * the judge is open-weight (default Qwen3-VL-30B-A3B-Instruct-FP8), not GPT-5; max_tokens is
    raised from 1000 so its reasoning is not cut off before the JSON, temperature 0
  * replies that are not valid JSON are repaired (missing / trailing commas, Python booleans),
    else asked again once with the output constrained to their JSON fields; their parser
    counts any such reply as not interpretable
  * their validation split was drawn after dropping dead image URLs, which cannot be replayed;
    we draw it from the full list with their seed and skip dead URLs, so the images are a
    different random sample from the same distribution
  * the oracle was trained on COCO-VTR source layers (outputs of blocks 0, 4, ..., 28)

  python run_ll_protocol.py --stage fetch --out_dir ./results/ll_llava15
"""

import argparse
import glob
import io
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAYERS = (0, 1, 2, 4, 8, 16, 24, 30, 31)
PROMPT = "Describe this image in detail."
N_VAL, VAL_SEED = 2048, 96817
JUDGE_SEED, BBOX, TOP_K = 42, 3, 5
OURS = ("patchscopes", "selfie", "image_slot", "oracle")
THEIRS = {"logitlens": "logitlens", "embeddinglens": "embeddinglens", "latentlens": "latentlens"}


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", required=True,
                 choices=["fetch", "extract", "export_lm", "latentlens", "plan", "ours", "judge",
                          "rejudge", "summary"])
  p.add_argument("--out_dir", default="./results/ll_llava15")
  p.add_argument("--ll_repo", default="../../../external/latentlens",
                 help="clone of github.com/McGill-NLP/latentlens")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--device", default="cuda")
  p.add_argument("--dtype", default="float16")
  p.add_argument("--n_images", type=int, default=100)
  p.add_argument("--num_patches", type=int, default=100, help="judged patches per layer")
  # export_lm / latentlens
  p.add_argument("--lm_dir", default=None, help="default: <out_dir>/llava15_lm")
  p.add_argument("--bank_dir", default=None, help="default: <out_dir>/bank")
  # ours (targets use COCO-VTR's convention: the output of decoder block t)
  p.add_argument("--methods", default=",".join(OURS))
  p.add_argument("--ps_prompt", default="identity", choices=["identity", "entity"])
  p.add_argument("--ps_target", type=int, default=0, help="best target in our Patchscopes sweep")
  p.add_argument("--selfie_target", type=int, default=12, help="best target in our SelfIE sweep")
  p.add_argument("--slot_target", type=int, default=8, help="best untrained image-slot target")
  p.add_argument("--oracle_dir", default="./results/vtr_llava15/oracle_crop7_add")
  p.add_argument("--rows_per_batch", type=int, default=16)
  # judge
  p.add_argument("--readout", default=None, help="judge stage: one readout name")
  p.add_argument("--judge_url", default="http://localhost:8000/v1")
  p.add_argument("--judge_model", default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
  p.add_argument("--judge_max_tokens", type=int, default=2048)
  p.add_argument("--judge_threads", type=int, default=4, help="rejudge: parallel requests")
  return p.parse_args(argv)


def paths(args):
  d = args.out_dir
  return {"images": os.path.join(d, "images"), "meta": os.path.join(d, "images", "images.json"),
          "states": os.path.join(d, "states.pt"), "plan": os.path.join(d, "plan.json"),
          "results": os.path.join(d, "readouts"), "judge": os.path.join(d, "judge"),
          "lm": args.lm_dir or os.path.join(d, "llava15_lm"),
          "bank": args.bank_dir or os.path.join(d, "bank")}


def load_meta(args):
  with open(paths(args)["meta"], encoding="utf-8") as f:
    return json.load(f)[:args.n_images]


def image_path(args, idx):
  return os.path.join(paths(args)["images"], f"{idx:05d}.jpg")


def readout_dir(args, name):
  return os.path.join(paths(args)["results"], name)


def ll_eval_module(args, name):
  """Import a module from their judge code (reproduce/scripts/evaluate)."""
  d = os.path.abspath(os.path.join(args.ll_repo, "reproduce", "scripts", "evaluate"))
  if not os.path.isdir(d):
    raise FileNotFoundError(f"LatentLens repo not found at {args.ll_repo} (--ll_repo)")
  if d not in sys.path:
    sys.path.insert(0, d)
  return __import__(name)


# ------------------------------------------------------------------------------- outputs


def patch_entry(idx, side, key, values):
  r, c = divmod(idx, side)
  return {"patch_idx": idx, "patch_row": r, "patch_col": c, key: values}


def image_entry(meta, patches, **extra):
  return {"image_idx": meta["image_idx"], "ground_truth_caption": meta.get("caption", ""),
          **extra, "chunks": [{"chunk_name": "Full Image", "patches": patches}]}


def write_json(path, data):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f)  # ASCII-escaped: their loader opens files with the platform encoding


def decode(tok, i):
  return tok.decode([int(i)])


def top_predictions(tok, logits, k=TOP_K):
  v, i = logits.float().topk(k, dim=-1)
  return [[{"token": decode(tok, t), "token_id": int(t), "logit": float(x)}
           for x, t in zip(vr.tolist(), ir.tolist())] for vr, ir in zip(v, i)]


# ---------------------------------------------------------------------------------- fetch


def stage_fetch(args):
  import datasets
  import requests
  ds = datasets.load_dataset("allenai/pixmo-cap", split="train")
  val = ds.train_test_split(test_size=N_VAL, seed=VAL_SEED)["test"]
  os.makedirs(paths(args)["images"], exist_ok=True)
  meta, failed = [], 0
  for row in val:
    if len(meta) >= args.n_images:
      break
    try:
      r = requests.get(row["image_url"], timeout=20)
      r.raise_for_status()
      image = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:  # noqa: BLE001 -- dead or unreadable URL: skip, as their download does
      failed += 1
      continue
    idx = len(meta)
    image.save(image_path(args, idx), quality=95)
    meta.append({"image_idx": idx, "image_url": row["image_url"], "caption": row["caption"]})
  write_json(paths(args)["meta"], meta)
  print(f"saved {len(meta)} images ({failed} URLs failed) to {paths(args)['images']}")


# -------------------------------------------------------------------------------- extract


def load_model(args):
  from general_utils import ModelAndTokenizer
  dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
  return ModelAndTokenizer(args.model_name, torch_dtype=dtype[args.dtype], device=args.device)


def image_positions(mt, input_ids):
  pos = (input_ids == mt.model.config.image_token_index).nonzero(as_tuple=True)[0]
  side = int(round(len(pos) ** 0.5))
  if side * side != len(pos):
    raise ValueError(f"expected a square image grid, found {len(pos)} image tokens")
  return pos, side


@torch.no_grad()
def stage_extract(args, mt):
  from visual_object_identification import get_decoder
  meta = load_meta(args)
  tok, model, dec = mt.tokenizer, mt.model, get_decoder(mt)
  head = model.get_output_embeddings()
  emb = F.normalize(model.get_input_embeddings().weight.float(), dim=-1)
  prompt = f"USER: <image>\n{PROMPT} ASSISTANT:"
  states, logit, embed = [], defaultdict(list), defaultdict(list)
  for m in meta:
    image = Image.open(image_path(args, m["image_idx"])).convert("RGB")
    inputs = mt.processor(images=image, text=prompt, return_tensors="pt").to(mt.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    pos, side = image_positions(mt, inputs["input_ids"][0])
    hs = torch.stack([out.hidden_states[l][0, pos] for l in LAYERS])  # [len(LAYERS), T, d]
    states.append(hs.to(torch.float16).cpu())
    for k, l in enumerate(LAYERS):
      preds = top_predictions(tok, head(dec.norm(hs[k])))
      logit[l].append(image_entry(m, [patch_entry(i, side, "top_predictions", p)
                                      for i, p in enumerate(preds)]))
      v, ids = (F.normalize(hs[k].float(), dim=-1) @ emb.T).topk(TOP_K, dim=-1)
      nn = [[{"token": decode(tok, t), "similarity": float(x)} for x, t in zip(vr, ir)]
            for vr, ir in zip(v.tolist(), ids.tolist())]
      embed[l].append(image_entry(m, [patch_entry(i, side, "nearest_neighbors", p)
                                      for i, p in enumerate(nn)]))
  torch.save({"layers": LAYERS, "states": torch.stack(states), "side": side,
              "image_idx": [m["image_idx"] for m in meta]}, paths(args)["states"])
  for l in LAYERS:
    write_json(os.path.join(readout_dir(args, "logitlens"), f"logit_lens_layer{l}_topk5.json"),
               {"model": args.model_name, "layer_idx": l, "top_k": TOP_K, "results": logit[l]})
    write_json(os.path.join(readout_dir(args, "embeddinglens"),
                            f"nearest_neighbors_layer{l}_topk5.json"),
               {"model": args.model_name, "layer_idx": l,
                "splits": {"validation": {"images": embed[l]}}})
  print(f"cached {len(meta)} images x {len(LAYERS)} layers x {side * side} tokens; "
        f"wrote logit-lens and embedding-lens results")


# ------------------------------------------------------------------------------ export_lm


def stage_export_lm(args, mt):
  """LLaVA fine-tunes its LLM, so the text bank must come from LLaVA's own weights (as they did
  for Qwen2-VL), not from Vicuna."""
  from transformers import LlamaForCausalLM
  from visual_object_identification import get_decoder
  lm = LlamaForCausalLM._from_config(mt.model.config.text_config, torch_dtype=mt.model.dtype)
  # Base weights only: drop oracle LoRA adapters if the model carries them.
  state = {k.replace(".base.", "."): v for k, v in get_decoder(mt).state_dict().items()
           if ".lora_" not in k}
  lm.model.load_state_dict(state)
  lm.lm_head.load_state_dict(mt.model.get_output_embeddings().state_dict())
  out = paths(args)["lm"]
  lm.save_pretrained(out)
  mt.tokenizer.save_pretrained(out)
  print(f"saved LLaVA's language model to {out}")
  return lm


# ----------------------------------------------------------------------------- latentlens


def bank_layers(bank_dir):
  found = {}
  for f in glob.glob(os.path.join(bank_dir, "**", "layer_*", "embeddings_cache.pt"),
                     recursive=True):
    found[int(os.path.basename(os.path.dirname(f)).split("_")[1])] = f
  if not found:
    raise FileNotFoundError(f"no layer_*/embeddings_cache.pt under {bank_dir}")
  return dict(sorted(found.items()))


@torch.no_grad()
def stage_latentlens(args):
  blob = torch.load(paths(args)["states"], weights_only=False)
  S, side = blob["states"], blob["side"]
  meta = load_meta(args)
  bank = bank_layers(paths(args)["bank"])
  dev = args.device
  N, T = S.shape[0], S.shape[2]
  vals = {l: [] for l in LAYERS}
  idxs = {l: [] for l in LAYERS}
  metas = {}
  for c, f in bank.items():
    cache = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    raw = cache["embeddings"]
    E = torch.empty(raw.shape, dtype=torch.float16, device=dev)
    for s in range(0, raw.shape[0], 65536):  # normalise in chunks: the cache is float32, ~10 GB
      E[s:s + 65536] = F.normalize(raw[s:s + 65536].to(dev).float(), dim=-1).half()
    metas[c] = cache["metadata"]
    del cache, raw
    for k, l in enumerate(LAYERS):
      v_l, i_l = [], []
      for n in range(N):
        sim = F.normalize(S[n, k].to(dev).float(), dim=-1).half() @ E.T  # [T, M]
        v, i = sim.float().topk(TOP_K, dim=-1)
        v_l.append(v.cpu())
        i_l.append(i.cpu())
      vals[l].append(torch.stack(v_l))  # [N, T, K]
      idxs[l].append(torch.stack(i_l))
    del E
    torch.cuda.empty_cache()
    print(f"  searched bank layer {c} ({len(metas[c])} entries)", flush=True)
  ctx = list(bank)
  for l in LAYERS:
    v = torch.stack(vals[l], 2).flatten(2)  # [N, T, C*K]
    i = torch.stack(idxs[l], 2).flatten(2)
    gv, gp = v.topk(TOP_K, dim=-1)
    results = []
    for n in range(N):
      patches = []
      for t in range(T):
        nearest = []
        for sim, p in zip(gv[n, t].tolist(), gp[n, t].tolist()):
          c = ctx[p // TOP_K]
          e = metas[c][int(i[n, t, p])]
          nearest.append({"token_str": e["token_str"], "token_id": e["token_id"],
                          "caption": e["caption"], "position": e["position"],
                          "similarity": sim, "contextual_layer": c})
        patches.append(patch_entry(t, side, "nearest_contextual_neighbors", nearest))
      results.append(image_entry(meta[n], patches, llm_layer_used=l))
    write_json(os.path.join(readout_dir(args, "latentlens"),
                            f"contextual_neighbors_visual{l}_allLayers.json"),
               {"visual_layer": l, "contextual_layers_used": ctx, "top_k": TOP_K,
                "results": results})
  print(f"wrote LatentLens results for layers {list(LAYERS)}")


# ----------------------------------------------------------------------------------- plan


def stage_plan(args):
  """Replay their judge loop's sampling, without the judge."""
  utils = ll_eval_module(args, "utils")
  side = torch.load(paths(args)["states"], weights_only=False)["side"]
  meta = load_meta(args)
  random.seed(JUDGE_SEED)
  np.random.seed(JUDGE_SEED)
  plan = []
  for l in LAYERS:
    n = 0
    for m in meta:
      if n >= args.num_patches:
        break
      _, mask = utils.process_image_with_mask(image_path(args, m["image_idx"]),
                                              model_name="llava")
      for r, c in utils.sample_valid_patch_positions(mask, bbox_size=BBOX, num_samples=1,
                                                     grid_size=side):
        if n >= args.num_patches:
          break
        plan.append({"layer": l, "image_idx": m["image_idx"], "row": r + BBOX // 2,
                     "col": c + BBOX // 2})
        n += 1
  write_json(paths(args)["plan"], plan)
  print(f"planned {len(plan)} judged patches over {len(LAYERS)} layers")


# ----------------------------------------------------------------------------------- ours


def our_readers(args, mt, methods):
  """{method: fn(states [n, d] float) -> next-token log-probs [n, V]}."""
  from vtr.injection import IMAGE_QUESTION, InjectionReadout
  from vtr.readouts import Patchscopes
  dummy, forms = ["object"], {"object": ["object"]}
  readers = {}
  for m in methods:
    if m in ("patchscopes", "selfie"):
      key = args.ps_prompt if m == "patchscopes" else "selfie"
      target = args.ps_target if m == "patchscopes" else args.selfie_target
      ro = Patchscopes(mt, dummy, forms, key, args.device)
      readers[m] = lambda h, ro=ro, t=target: ro.next_logprobs(h, t)
    elif m == "image_slot":
      ro = InjectionReadout(mt, dummy, forms, "image", args.device)
      readers[m] = lambda h, ro=ro: ro.next_logprobs(
          h, [ro.slots("all")] * h.shape[0], args.slot_target).cpu()
    elif m != "oracle":
      raise ValueError(m)
  if "oracle" in methods:
    from vtr.oracle import add_lora, lora_parameters, set_lora
    blob = torch.load(os.path.join(args.oracle_dir, "lora.pt"), weights_only=False)
    cfg = blob["config"]
    if not lora_parameters(mt):
      add_lora(mt, cfg["rank"], cfg["lora_alpha"])
    for n, p in lora_parameters(mt).items():
      p.data.copy_(blob["lora"][n].to(p.device))
    set_lora(mt, False)  # on only while the oracle reads
    ro = InjectionReadout(mt, dummy, forms, "image", args.device, IMAGE_QUESTION)
    ro.ref_norm = blob["ref_norm"]

    def oracle(h, ro=ro, cfg=cfg):
      set_lora(mt, True)
      try:
        return ro.next_logprobs(h, [ro.slots("all")] * h.shape[0], cfg["target_layer"],
                                not cfg["no_norm_match"],
                                add=cfg.get("inject", "replace") == "add").cpu()
      finally:
        set_lora(mt, False)
    readers["oracle"] = oracle
  return readers


def stage_ours(args, mt):
  blob = torch.load(paths(args)["states"], weights_only=False)
  S, side = blob["states"], blob["side"]
  row_of = {idx: n for n, idx in enumerate(blob["image_idx"])}
  meta = load_meta(args)
  with open(paths(args)["plan"], encoding="utf-8") as f:
    plan = json.load(f)
  methods = [m for m in args.methods.split(",") if m]
  readers = our_readers(args, mt, methods)
  for m, read in readers.items():
    for k, l in enumerate(LAYERS):
      todo = [p for p in plan if p["layer"] == l]
      preds = {}
      for start in range(0, len(todo), args.rows_per_batch):
        chunk = todo[start:start + args.rows_per_batch]
        h = torch.stack([S[row_of[p["image_idx"]], k, p["row"] * side + p["col"]]
                         for p in chunk]).float()
        for p, top in zip(chunk, top_predictions(mt.tokenizer, read(h))):
          preds[(p["image_idx"], p["row"] * side + p["col"])] = top
      results = [image_entry(x, [patch_entry(t, side, "top_predictions",
                                             preds.get((x["image_idx"], t), []))
                                 for t in range(side * side)]) for x in meta]
      write_json(os.path.join(readout_dir(args, m), f"{m}_layer{l}_topk5.json"),
                 {"method": m, "layer_idx": l, "top_k": TOP_K, "results": results})
    print(f"  {m}: wrote {len(LAYERS)} layers", flush=True)


# ---------------------------------------------------------------------------------- judge


UNPARSED = "Could not parse response"
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "interpretable": {"type": "boolean"},
                   **{k: {"type": "array", "items": {"type": "string"}}
                      for k in ("concrete_words", "abstract_words", "global_words")}},
    "required": ["reasoning", "interpretable", "concrete_words", "abstract_words",
                 "global_words"]}


def parse_judgment(text):
  """Their parser (outermost {...} -> json.loads), then repairs for slips an open judge makes
  copying their output template, which has no commas after two of its fields: missing commas
  between fields, trailing commas, Python booleans. None if it still fails."""
  i, j = text.find("{"), text.rfind("}") + 1
  if i == -1 or j <= i:
    return None
  s = text[i:j]
  try:
    return json.loads(s)
  except json.JSONDecodeError:
    pass
  s = re.sub(r'(["\]}]|true|false|\d)\s*\n(\s*")', r"\1,\n\2", s)
  s = re.sub(r",(\s*[}\]])", r"\1", s)
  s = re.sub(r":\s*True\b", ": true", re.sub(r":\s*False\b", ": false", s))
  try:
    return json.loads(s)
  except json.JSONDecodeError:
    return None


def make_judgment(args, ev, client):
  """Their OpenRouter branch (chat.completions; image, prompt, crop) with more tokens, T=0 and a
  forgiving parser. A reply that still does not parse is asked again once with the output
  constrained to their JSON fields (vLLM structured output); if that fails too, it is kept as
  their unparsed fallback (not interpretable) with the full reply in "raw"."""

  def judgment(_, image_with_bbox, cropped_image, candidate_words, model="", api_provider=""):
    content = [{"type": "image_url", "image_url": {"url": ev.encode_image(image_with_bbox)}},
               {"type": "text",
                "text": ev.IMAGE_PROMPT_WITH_CROP.format(candidate_words=str(candidate_words))}]
    if cropped_image is not None:
      content.append({"type": "image_url",
                      "image_url": {"url": ev.encode_image(cropped_image)}})
    texts = []
    for constrained in (False, True):
      extra = {"response_format": {"type": "json_schema", "json_schema": {
          "name": "judgment", "schema": JUDGMENT_SCHEMA}}} if constrained else {}
      resp = client.chat.completions.create(
          model=args.judge_model, max_tokens=args.judge_max_tokens, temperature=0.0,
          messages=[{"role": "user", "content": content}], **extra)
      texts.append(resp.choices[0].message.content or "")
      out = parse_judgment(texts[-1])
      if out is not None:
        out["parse"] = "constrained" if constrained else "free"
        return out
    return {"interpretable": False, "concrete_words": [], "abstract_words": [],
            "global_words": [], "reasoning": f"{UNPARSED}: {texts[0][:200]}", "raw": texts}

  return judgment


def stage_judge(args):
  """Their judge loop unchanged, with the OpenAI client pointed at a local server."""
  from openai import OpenAI
  ev = ll_eval_module(args, "evaluate_interpretability")
  client = OpenAI(base_url=args.judge_url, api_key="EMPTY")
  ev.OpenAI = lambda **kw: client
  ev.get_llm_judgment = make_judgment(args, ev, client)
  names = [args.readout] if args.readout else list(THEIRS) + list(OURS)
  for name in names:
    ns = argparse.Namespace(
        api_key="EMPTY", api_provider="openrouter", api_model=args.judge_model,
        results_dir=readout_dir(args, name), images_dir=paths(args)["images"],
        output_dir=os.path.join(paths(args)["judge"], name), layers=list(LAYERS),
        num_patches=args.num_patches, num_samples_per_image=1, top_k=TOP_K,
        model_name="llava", seed=JUDGE_SEED)
    ev.evaluate_model(ns)


def stage_rejudge(args):
  """Judge again only the patches whose reply did not parse, with the same inputs their loop
  built (image with the red 3x3 box, and the crop) and make_judgment's repairs. The original
  results are kept as evaluation_results.orig.json."""
  from concurrent.futures import ThreadPoolExecutor
  from openai import OpenAI
  ev = ll_eval_module(args, "evaluate_interpretability")
  client = OpenAI(base_url=args.judge_url, api_key="EMPTY")
  judgment = make_judgment(args, ev, client)
  bbox_size = 3
  names = [args.readout] if args.readout else list(THEIRS) + list(OURS)
  for name in names:
    path = os.path.join(paths(args)["judge"], name, "evaluation_results.json")
    with open(path, encoding="utf-8") as f:
      res = json.load(f)
    backup = path.replace(".json", ".orig.json")
    if not os.path.exists(backup):
      write_json(backup, res)
    todo = [(layer, p) for layer in res for p in layer["patches"]
            if UNPARSED in str(p["judgment"].get("reasoning", ""))]
    print(f"{name}: re-judging {len(todo)} unparsed patches", flush=True)

    def redo(item):
      layer, p = item
      img, _ = ev.process_image_with_mask(ev.find_image_path(paths(args)["images"],
                                                             p["image_idx"]), model_name="llava")
      patch_size = 512.0 / layer["grid_size"]
      row, col = p["patch_row"] - bbox_size // 2, p["patch_col"] - bbox_size // 2
      boxed = ev.draw_bbox_on_image(img, ev.calculate_square_bbox_from_patch(
          row, col, patch_size=patch_size, size=bbox_size))
      crop = img.crop((max(0, int(col * patch_size)), max(0, int(row * patch_size)),
                       min(int((col + bbox_size) * patch_size), img.size[0]),
                       min(int((row + bbox_size) * patch_size), img.size[1])))
      j = judgment(None, boxed, crop, p["candidate_words"])
      p["judgment"] = j
      p["interpretable"] = any(len(j.get(k, [])) > 0
                               for k in ("concrete_words", "abstract_words", "global_words"))
      return j.get("parse", "failed")

    with ThreadPoolExecutor(args.judge_threads) as pool:
      outcome = list(pool.map(redo, todo))
    for layer in res:
      layer["interpretable_count"] = sum(p["interpretable"] for p in layer["patches"])
      layer["total_count"] = len(layer["patches"])
      layer["interpretable_fraction"] = (layer["interpretable_count"] / layer["total_count"]
                                         if layer["total_count"] else 0.0)
    write_json(path, res)
    print(f"{name}: " + ", ".join(f"{k} {outcome.count(k)}"
                                  for k in ("free", "constrained", "failed")), flush=True)


# -------------------------------------------------------------------------------- summary


def wilson(k, n, z=1.96):
  if n == 0:
    return float("nan"), float("nan")
  p = k / n
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
  mid = (p + z * z / (2 * n)) / (1 + z * z / n)
  return mid - half, mid + half


def stage_summary(args):
  import pandas as pd
  with open(paths(args)["plan"], encoding="utf-8") as f:
    plan = {(p["layer"], p["image_idx"], p["row"], p["col"]) for p in json.load(f)}
  rows = []
  for f in sorted(glob.glob(os.path.join(paths(args)["judge"], "*", "evaluation_results.json"))):
    name = os.path.basename(os.path.dirname(f))
    with open(f, encoding="utf-8") as fh:
      res = json.load(fh)
    for layer in res:
      ps = layer["patches"]
      off = sum((layer["layer"], p["image_idx"], p["patch_row"], p["patch_col"]) not in plan
                for p in ps)
      k, n = sum(p["interpretable"] for p in ps), len(ps)
      lo, hi = wilson(k, n)
      kinds = {t: np.mean([bool(p["judgment"].get(f"{t}_words")) for p in ps]) if ps else np.nan
               for t in ("concrete", "abstract", "global")}
      rows.append({"readout": name, "layer": layer["layer"], "depth": layer["layer"] / LAYERS[-1],
                   "n": n, "interpretable": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi,
                   **kinds, "unparsed": sum("Could not parse" in str(p["judgment"].get(
                       "reasoning", "")) for p in ps), "off_plan": off})
  if not rows:
    raise FileNotFoundError(f"no judge results under {paths(args)['judge']}")
  df = pd.DataFrame(rows)
  df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
  wide = df.pivot(index="readout", columns="layer", values="interpretable") * 100
  wide["mean"] = wide.mean(axis=1)
  pd.set_option("display.width", 200)
  print("% interpretable (judge), per layer; 100 patches per cell, 95% CI about +-10 points")
  print(wide.round(1).to_string())
  if df["off_plan"].any():
    print("\nWARNING: judged patches differ from plan.json (readouts not comparable):")
    print(df[df["off_plan"] > 0][["readout", "layer", "off_plan"]].to_string(index=False))
  if df["unparsed"].any():
    print(f"\nunparsed judge replies: {int(df['unparsed'].sum())} (counted as not interpretable)")
  print(f"\nSaved {os.path.join(args.out_dir, 'summary.csv')}")
  return df


def main(argv=None, mt=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  if args.stage in ("extract", "export_lm", "ours") and mt is None:
    mt = load_model(args)
  return {"fetch": lambda: stage_fetch(args), "extract": lambda: stage_extract(args, mt),
          "export_lm": lambda: stage_export_lm(args, mt), "latentlens": lambda: stage_latentlens(args),
          "plan": lambda: stage_plan(args), "ours": lambda: stage_ours(args, mt),
          "judge": lambda: stage_judge(args), "rejudge": lambda: stage_rejudge(args),
          "summary": lambda: stage_summary(args)}[args.stage]()


if __name__ == "__main__":
  main()
