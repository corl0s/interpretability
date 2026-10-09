r"""Get a LatentLens contextual bank for a model: download a pre-built one, or build it.

Pre-built banks from the LatentLens authors (Hugging Face, same layer_N/embeddings_cache.pt
layout our LatentLens readout loads), e.g.
  McGill-NLP/contextual_embeddings-qwen2.5-vl-7b   layers 1,2,4,8,16,24,26,27 (HF index), d=3584

  # download (needs disk space: tens of GB)
  python get_latentlens_bank.py --download McGill-NLP/contextual_embeddings-qwen2.5-vl-7b \
      --out ./results/latentlens_index_qwen25vl/bank

  # or build from the model's own LLM on the LatentLens corpus (~13 GPU-hours on the full corpus)
  python get_latentlens_bank.py --build --model_name Qwen/Qwen2.5-VL-7B-Instruct \
      --out ./results/latentlens_index_qwen25vl/bank
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def auto_bank_layers(n_layers):
  """LatentLens' default grid in HF hidden_states indices, kept below the post-norm last entry."""
  base = [l for l in (1, 2, 4, 8, 16, 24) if l < n_layers - 1]
  return sorted(set(base + [n_layers - 2, n_layers - 1]))


def main(argv=None, mt=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--out", required=True)
  p.add_argument("--download", default=None, help="Hugging Face repo id of a pre-built bank")
  p.add_argument("--build", action="store_true")
  p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
  p.add_argument("--corpus", default=None, help="default: LatentLens concepts.txt")
  p.add_argument("--corpus_dir", default="./results/latentlens_index")
  p.add_argument("--corpus_limit", type=int, default=0)
  p.add_argument("--bank_layers", default=None, help="HF indices; default: LatentLens grid")
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = p.parse_args(argv)

  if args.download:
    from huggingface_hub import snapshot_download
    path = snapshot_download(args.download, local_dir=args.out)
    print(f"downloaded {args.download} to {path}")
    return path

  if not args.build:
    p.error("give --download REPO or --build")
  import latentlens_object_identification as L
  if mt is None:
    from general_utils import ModelAndTokenizer
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    mt = ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)
  decoder = mt.model.model.language_model
  n_layers = decoder.config.num_hidden_layers
  layers = ([int(x) for x in args.bank_layers.split(",")] if args.bank_layers
            else auto_bank_layers(n_layers))
  texts = L.load_corpus(args.corpus, args.corpus_dir, args.corpus_limit, 0)
  print(f"building bank at HF layers {layers} from {len(texts)} sentences")
  bank = L.build_bank(decoder, mt.tokenizer, texts, layers, batch_size=args.batch_size)
  bank.save(args.out)
  print(f"saved {bank} to {args.out}")
  return args.out


if __name__ == "__main__":
  main()
