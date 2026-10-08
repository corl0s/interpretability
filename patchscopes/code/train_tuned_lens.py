r"""Train a tuned lens (Belrose et al., 2023) for LLaVA-1.5's language model, on text.

For every decoder layer l a translator h -> h + A_l h + b_l (initialised at zero, i.e. the logit
lens) is trained so that lm_head(norm(translated h_l)) matches the model's own final next-token
distribution (KL divergence), on text sentences run through LLaVA's LLM. Layer l is the output of
decoder block l before the final norm, the same convention as everywhere in this project.

Layers are trained independently: each layer's loss is backpropagated on its own, so peak memory
is one layer's logits, not all 32.

The translators are saved to --out (a dict with A [L, d, d], b [L, d] in fp16, plus the KL on
held-out sentences for the logit lens and for the tuned lens, per layer). The tuned lens should
have a lower KL than the logit lens at every layer except the last, where they are equal.

  python train_tuned_lens.py                       # defaults: LatentLens concepts.txt corpus
  python train_tuned_lens.py --steps 50 --out ./results/tuned_lens/smoke.pt   # quick check
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from latentlens_object_identification import load_corpus  # noqa: E402
from visual_object_identification import get_decoder  # noqa: E402
from vtr.readouts import rms_norm  # noqa: E402


def capture_layers(mt, input_ids, attention_mask):
  """Outputs of every decoder block (pre final-norm) and the final log-probs."""
  decoder = get_decoder(mt)
  captured = {}
  handles = [layer.register_forward_hook(
      lambda m, i, o, k=k: captured.__setitem__(k, (o[0] if isinstance(o, tuple) else o)))
      for k, layer in enumerate(decoder.layers)]
  try:
    with torch.no_grad():
      logits = mt.model(input_ids=input_ids, attention_mask=attention_mask).logits
  finally:
    for h in handles:
      h.remove()
  return [captured[k] for k in range(len(decoder.layers))], logits


def batches(tokenizer, texts, batch_size, max_len, device, rng):
  tokenizer.padding_side = "right"
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  while True:
    batch = rng.sample(texts, min(batch_size, len(texts)))
    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len)
    yield enc["input_ids"].to(device), enc["attention_mask"].to(device)


def layer_kl(A, b, h, target_logp, norm_w, eps, head_w):
  """Mean KL(final || lens) over positions, for one layer's translator."""
  x = h.float()
  x = x + x @ A.T + b
  logp = (rms_norm(x, norm_w, eps) @ head_w.T).log_softmax(-1)
  return (target_logp.exp() * (target_logp - logp)).sum(-1).mean()


def select_positions(attention_mask, n, gen):
  """Up to n random (row, position) pairs among real tokens, skipping BOS."""
  mask = attention_mask.clone()
  mask[:, 0] = 0
  idx = mask.nonzero()
  if len(idx) > n:
    idx = idx[torch.randperm(len(idx), generator=gen)[:n]]
  idx = idx.to(attention_mask.device)
  return idx[:, 0], idx[:, 1]


def evaluate(mt, A, b, texts, args, norm_w, eps, head_w):
  """Held-out KL per layer: logit lens (A=b=0) vs tuned lens."""
  n_layers = A.shape[0]
  kl_lens, kl_tuned, count = torch.zeros(n_layers), torch.zeros(n_layers), 0
  gen = torch.Generator().manual_seed(args.seed + 1)
  for start in range(0, len(texts), args.batch_size):
    enc = mt.tokenizer(texts[start:start + args.batch_size], return_tensors="pt", padding=True,
                       truncation=True, max_length=args.max_len)
    ids, attn = enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)
    hidden, logits = capture_layers(mt, ids, attn)
    rows, cols = select_positions(attn, args.tokens_per_step, gen)
    target = logits[rows, cols].float().log_softmax(-1)
    with torch.no_grad():
      for l in range(n_layers):
        h = hidden[l][rows, cols]
        zero_a, zero_b = torch.zeros_like(A[l]), torch.zeros_like(b[l])
        kl_lens[l] += layer_kl(zero_a, zero_b, h, target, norm_w, eps, head_w).item()
        kl_tuned[l] += layer_kl(A[l], b[l], h, target, norm_w, eps, head_w).item()
    count += 1
  return (kl_lens / max(count, 1)).tolist(), (kl_tuned / max(count, 1)).tolist()


def train(mt, texts, args):
  decoder = get_decoder(mt)
  n_layers = len(decoder.layers)
  d = decoder.norm.weight.shape[0]
  norm_w = decoder.norm.weight.detach().float().to(args.device)
  eps = decoder.norm.variance_epsilon
  head_w = mt.model.lm_head.weight.detach().float().to(args.device)

  rng = random.Random(args.seed)
  texts = list(texts)
  rng.shuffle(texts)
  n_eval = max(1, int(len(texts) * args.eval_fraction))
  eval_texts, train_texts = texts[:n_eval], texts[n_eval:]

  A = torch.zeros(n_layers, d, d, device=args.device, requires_grad=True)
  b = torch.zeros(n_layers, d, device=args.device, requires_grad=True)
  opt = torch.optim.Adam([A, b], lr=args.lr)
  sched = torch.optim.lr_scheduler.LambdaLR(
      opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.0, 1 - s / args.steps))
  gen = torch.Generator().manual_seed(args.seed)
  stream = batches(mt.tokenizer, train_texts, args.batch_size, args.max_len, args.device, rng)

  t0 = time.time()
  for step in range(args.steps):
    ids, attn = next(stream)
    hidden, logits = capture_layers(mt, ids, attn)
    rows, cols = select_positions(attn, args.tokens_per_step, gen)
    target = logits[rows, cols].float().log_softmax(-1)
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for l in range(n_layers):  # independent per layer: backprop one layer at a time
      loss = layer_kl(A[l], b[l], hidden[l][rows, cols].detach(), target, norm_w, eps, head_w)
      loss.backward()
      total += loss.item()
    opt.step()
    sched.step()
    if step % args.log_every == 0 or step == args.steps - 1:
      print(f"  step {step}/{args.steps}  mean KL over layers {total / n_layers:.3f}  "
            f"({time.time() - t0:.0f}s)")

  kl_lens, kl_tuned = evaluate(mt, A.detach(), b.detach(), eval_texts, args, norm_w, eps,
                               head_w)
  return A.detach(), b.detach(), kl_lens, kl_tuned


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--corpus", default=None,
                 help="one sentence per line; default: LatentLens concepts.txt")
  p.add_argument("--corpus_dir", default="./results/latentlens_index",
                 help="where concepts.txt is (downloaded if missing)")
  p.add_argument("--corpus_limit", type=int, default=0)
  p.add_argument("--out", default="./results/tuned_lens/llava15_tuned_lens.pt")
  p.add_argument("--steps", type=int, default=2000)
  p.add_argument("--batch_size", type=int, default=16)
  p.add_argument("--max_len", type=int, default=64)
  p.add_argument("--tokens_per_step", type=int, default=512)
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--warmup", type=int, default=100)
  p.add_argument("--eval_fraction", type=float, default=0.02)
  p.add_argument("--log_every", type=int, default=100)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


def main(argv=None, mt=None):
  args = parse_args(argv)
  if args.device.startswith("cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True
  if mt is None:
    from general_utils import ModelAndTokenizer
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    mt = ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)
  texts = load_corpus(args.corpus, args.corpus_dir, args.corpus_limit, args.seed)
  print(f"Training tuned lens on {len(texts)} sentences, {args.steps} steps")
  A, b, kl_lens, kl_tuned = train(mt, texts, args)

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  torch.save({"A": A.half().cpu(), "b": b.half().cpu(), "kl_logit_lens": kl_lens,
              "kl_tuned_lens": kl_tuned, "args": vars(args)}, args.out)
  report = [{"layer": l, "kl_logit_lens": round(a, 4), "kl_tuned_lens": round(t, 4)}
            for l, (a, t) in enumerate(zip(kl_lens, kl_tuned))]
  with open(os.path.splitext(args.out)[0] + "_kl.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
  print("held-out KL (logit lens -> tuned lens):")
  for r in report:
    print(f"  layer {r['layer']:2d}: {r['kl_logit_lens']:.3f} -> {r['kl_tuned_lens']:.3f}")
  print(f"saved {args.out}")
  return args.out


if __name__ == "__main__":
  main()
