"""CPU tests for the visual activation oracle pilot (vtr/oracle.py, train_oracle.py).

  * helpers: square crop size, box position, answer cleaning, LoRA starts as the identity and
    switching it off restores the base model
  * slot injection is differentiable: gradients reach LoRA layers after the injection layer
  * train_oracle end to end on a tiny random LLaVA: data (benchmark images excluded), training
    (LoRA weights move, held-out answers dropped), eval tables
  * run_vtr's image_slot / oracle readouts on every row reproduce the oracle's own eval cell
  * run_vtr's slot_nn (image-slot + LatentLens): offset 0 equals LatentLens on h; nested bank

  python test_oracle.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_slot_nn  # noqa: E402
import run_vtr  # noqa: E402
import train_oracle  # noqa: E402
from run_ps_sweep import select_rows  # noqa: E402
from test_rewind import add_unlabeled_images  # noqa: E402
from test_vtr import test_latentlens  # noqa: E402
from test_visual_object_identification import build_tiny_vlm, make_fake_coco  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples  # noqa: E402
from vtr.geometry import Llava15Geometry  # noqa: E402
from vtr.readouts import LatentLens, class_forms  # noqa: E402
from vtr.oracle import (LoRALinear, add_lora, clean_answer, draw_box,  # noqa: E402
                        lora_parameters, set_lora, slot_injection, square_crop)
from visual_object_identification import get_decoder  # noqa: E402


def test_helpers():
  img = square_crop(Image.new("RGB", (640, 480), (0, 0, 0)))
  assert img.size == (336, 336)
  boxed = np.array(draw_box(img, 24 * 5 + 7))  # row 5, col 7 -> block rows 4-6, cols 6-8
  red = np.argwhere((boxed[..., 0] == 255) & (boxed[..., 1] == 0))
  assert red[:, 0].min() == 4 * 14 and red[:, 0].max() == 7 * 14 - 1
  assert red[:, 1].min() == 6 * 14 and red[:, 1].max() == 9 * 14 - 1
  assert clean_answer(" Dog.\nmore") == "dog"
  assert clean_answer("A traffic light!") == "traffic light"
  base = torch.nn.Linear(8, 4)
  lora = LoRALinear(base, rank=2, alpha=4)
  x = torch.randn(3, 8)
  assert torch.allclose(lora(x), base(x))  # B starts at zero
  lora.lora_B.data.normal_()
  assert not torch.allclose(lora(x), base(x))
  lora.enabled = False
  assert torch.allclose(lora(x), base(x))
  print("helpers OK (crop, box, answers, LoRA identity / switch)")


def test_injection_gradients(mt):
  params = add_lora(mt, rank=2, alpha=4)
  for p in params.values():
    p.data.normal_(0, 0.01)
  layer = 1
  ids = mt.tokenizer(["USER: hello there ASSISTANT:"] * 2, return_tensors="pt")["input_ids"]
  vecs = torch.randn(2, 64)
  seen = {}
  probe = get_decoder(mt).layers[layer + 1].register_forward_pre_hook(
      lambda m, a, kw: seen.__setitem__("h", (a[0] if a else kw["hidden_states"]).detach()),
      with_kwargs=True)
  with slot_injection(mt, layer, [2, 3], vecs):
    out = mt.model(input_ids=ids)
  probe.remove()
  assert torch.allclose(seen["h"][:, 2], vecs) and torch.allclose(seen["h"][:, 3], vecs)
  out.logits[:, -1].sum().backward()
  late = [p.grad for n, p in params.items() if f"layers.{layer + 1}." in n and "lora_A" in n]
  assert late and all(g is not None and g.abs().sum() > 0 for g in late)
  for m in mt.model.modules():  # undo, so the end-to-end test starts from the plain model
    for name, child in list(m.named_children()):
      if isinstance(child, LoRALinear):
        setattr(m, name, child.base)
  assert not lora_parameters(mt)
  print("slot injection OK (states land, gradients reach LoRA after the injection layer)")


def main():
  test_helpers()
  with tempfile.TemporaryDirectory() as tmp:
    coco_dir = os.path.join(tmp, "coco")
    ann_path = make_fake_coco(coco_dir)
    geometry = Llava15Geometry()
    samples, class_names = select_object_samples(ann_path, geometry, n_images=4, per_class=3,
                                                 min_area=20000, max_area=None, min_tokens=4,
                                                 seed=0)
    add_unlabeled_images(coco_dir, 4)
    mt, processor = build_tiny_vlm()
    test_injection_gradients(mt)
    states, meta = cache_states(mt, processor, samples, coco_dir, per_bin=2, n_outside=2,
                                seed=0)
    vtr_dir = os.path.join(tmp, "vtr")
    os.makedirs(vtr_dir)
    with open(os.path.join(vtr_dir, "samples.json"), "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "samples": samples}, f)
    torch.save({"states": states, "meta": meta}, os.path.join(vtr_dir, "states.pt"))
    with open(os.path.join(vtr_dir, "fair_classes.json"), "w", encoding="utf-8") as f:
      json.dump({"kept": class_names, "dropped": {}}, f)
    os.makedirs(os.path.join(vtr_dir, "scores"))  # stand-in probe scores for the split tables
    np.savez(os.path.join(vtr_dir, "scores", "probe_linear.npz"), layers=np.arange(3),
             scores=np.random.RandomState(0).randn(3, len(meta), len(class_names))
             .astype(np.float16))

    out_dir = os.path.join(tmp, "oracle")
    common = ["--coco_dir", coco_dir, "--vtr_dir", vtr_dir, "--out_dir", out_dir,
              "--device", "cpu", "--layers", "0,2", "--n_images", "4", "--tokens_per_image", "3",
              "--shard_size", "2", "--gen_batch", "2"]
    train_oracle.main(common + ["--stage", "data"], mt=mt)
    with open(os.path.join(out_dir, "data_images.json"), encoding="utf-8") as f:
      used = set(json.load(f))
    assert not used & {s["image_id"] for s in samples}, "oracle data used a benchmark image"
    shards = sorted(f for f in os.listdir(os.path.join(out_dir, "data")) if f.endswith(".pt"))
    assert len(shards) == 2, shards
    # Random weights generate noise; give the shards readable answers for the training test.
    words = ["dog", "cat", "tree", "box", "sky", "dogs"]
    for k, f in enumerate(shards):
      path = os.path.join(out_dir, "data", f)
      blob = torch.load(path, weights_only=False)
      assert blob["states"].shape == (6, 2, 64), blob["states"].shape
      assert len(blob["answers"]) == 6
      blob["answers"] = words
      torch.save(blob, path)

    train = ["--rank", "2", "--micro_batch", "2", "--accum", "2", "--train_examples", "12",
             "--target_layer", "1", "--n_holdout", "1", "--lr", "1e-2", "--warmup", "1",
             "--save_every", "1"]
    train_oracle.main(common + ["--stage", "train"] + train, mt=mt)
    with open(os.path.join(out_dir, "holdout.json"), encoding="utf-8") as f:
      held = json.load(f)
    blob = torch.load(os.path.join(out_dir, "lora.pt"), weights_only=False)
    assert any(v.abs().sum() > 0 for n, v in blob["lora"].items() if "lora_B" in n)
    assert not os.path.exists(os.path.join(out_dir, "train_ckpt.pt"))
    log = pd.read_csv(os.path.join(out_dir, "train_log.csv"))
    assert len(log) >= 1 and np.isfinite(log["loss"]).all()
    print(f"training OK (held out {held}, {len(log)} log rows)")

    train_oracle.main(common + ["--stage", "eval", "--n_boot", "20", "--eval_images", "4"],
                      mt=mt)
    ev = os.path.join(out_dir, "eval")
    best = pd.read_csv(os.path.join(ev, "proto_best.csv"))
    assert set(best["variant"]) == {"untrained", "oracle"}
    u = np.load(os.path.join(ev, "cells", "untrained", "s2_t1.npz"))["pmi"]
    o = np.load(os.path.join(ev, "cells", "oracle", "s2_t1.npz"))["pmi"]
    assert u.shape == o.shape and not np.allclose(u, o)
    assert os.path.exists(os.path.join(ev, "heldout_split.csv"))
    hard = pd.read_csv(os.path.join(ev, "hard_split.csv"))
    assert set(hard["group"]) <= {"easy", "hard"} and "probe" in set(hard["variant"])
    main_obj = pd.read_csv(os.path.join(ev, "main_object.csv"))
    assert set(main_obj["main_object"]) <= set(class_names)
    print("eval OK (oracle differs from the untrained model; seen/held-out and easy/hard tables)")

    # run_vtr's image_slot and oracle readouts on every row (full-benchmark evaluation).
    proto = os.path.join(vtr_dir, "injection_proto")
    os.makedirs(proto)
    pd.DataFrame({"variant": ["img_all", "img_all", "img_own"], "source": [0, 2, 0],
                  "target_chosen_on_dev": [1, 0, 2]}).to_csv(
        os.path.join(proto, "proto_best.csv"), index=False)
    args = run_vtr.parse_args(["--out_dir", vtr_dir, "--device", "cpu", "--oracle_dir", out_dir,
                               "--readouts", "image_slot", "oracle", "--slot_rows_per_batch", "4"])
    assert run_vtr.slot_targets(args, [0, 1, 2, 3]) == {0: 1, 1: 1, 2: 0, 3: 0}
    run_vtr.run_readouts(args, mt, states, meta, class_names, [0, 2])
    sc = {n: run_vtr.load_scores(os.path.join(vtr_dir, "scores", f"{n}.npz"))
          for n in ("image_slot_pmi", "oracle_pmi", "oracle_raw")}
    for n, (ls, s) in sc.items():
      assert ls == [0, 2] and s.shape == (2, len(meta), len(class_names)), (n, s.shape)
      assert np.isfinite(s).all(), n
    assert not np.allclose(sc["image_slot_pmi"][1], sc["oracle_pmi"][1])
    oracle_s2 = sc["oracle_pmi"][1][1]  # source layer 2
    assert not any(m.enabled for m in mt.model.modules() if hasattr(m, "enabled"))
    # The oracle readout reproduces train_oracle's own eval cell (source 2, target 1).
    cell = np.load(os.path.join(ev, "cells", "oracle", "s2_t1.npz"))["pmi"].astype(np.float32)
    rows = select_rows(meta, 4, 0)
    assert np.allclose(oracle_s2[rows], cell, atol=2e-2), np.abs(oracle_s2[rows] - cell).max()
    print("run_vtr image_slot + oracle readouts OK (all rows, LoRA off afterwards)")

    # slot_nn: image-slot injection read by LatentLens, on a bank nested one level down (the
    # layout extract_embeddings.py writes).
    flat = test_latentlens(mt, class_names, tmp)
    bank_dir = os.path.join(tmp, "nested_bank")
    shutil.copytree(flat, os.path.join(bank_dir, "some-model"))
    args = run_vtr.parse_args(["--out_dir", vtr_dir, "--device", "cpu", "--readouts", "slot_nn",
                               "--slot_rows_per_batch", "4", "--latentlens_bank", bank_dir,
                               "--latentlens_name", "llvg", "--nn_offsets", "0,1,5"])
    run_vtr.run_readouts(args, mt, states, meta, class_names, [0, 2])
    names = ["img_k0", "img_k1", "img_k5", "ans_k1", "ans_k5"]
    nn = {n: run_vtr.load_scores(os.path.join(vtr_dir, "scores", f"slot_nn_llvg_{n}.npz"))[1]
          for n in names}
    assert not os.path.exists(os.path.join(vtr_dir, "scores", "slot_nn_llvg_ans_k0.npz"))
    for n, s in nn.items():
      assert s.shape == (2, len(meta), len(class_names)) and not np.isnan(s).any(), n
    # Offset 0 reads the injected slots right after injection: plain LatentLens on h.
    plain = LatentLens(bank_dir, mt.tokenizer, class_names,
                       class_forms(class_names, synonyms=False), "cpu")
    for i, l in enumerate([0, 2]):
      ref = plain.scores(states[:, l].float())
      assert np.allclose(nn["img_k0"][i], ref, atol=2e-3), np.abs(nn["img_k0"][i] - ref).max()
    assert not np.allclose(nn["img_k1"], nn["img_k0"])
    assert not np.allclose(nn["ans_k1"], nn["img_k1"])
    compare_slot_nn.main(["--out_dir", vtr_dir, "--bank_name", "llvg", "--n_images", "2",
                          "--n_boot", "20"])
    cmp = pd.read_csv(os.path.join(vtr_dir, "slot_nn", "compare_llvg.csv"))
    assert {"slot_nn_llvg_img_k0", "image_slot_pmi", "oracle_pmi"} <= set(cmp["readout"])
    print("run_vtr slot_nn OK (offset 0 = LatentLens on h; nested bank found; comparison table)")

    try:  # the pilot's box3 shards cannot be mixed with another label view
      train_oracle.main(common + ["--stage", "data", "--label_view", "crop7"], mt=mt)
      raise AssertionError("mixed label views were accepted")
    except ValueError as e:
      assert "box3" in str(e)
    crop_dir = os.path.join(tmp, "oracle_crop")
    argv = [a if a != out_dir else crop_dir for a in common]
    train_oracle.main(argv + ["--stage", "data", "--label_view", "crop7"], mt=mt)
    blob = torch.load(os.path.join(crop_dir, "data", "shard_0000.pt"), weights_only=False)
    assert blob["label_view"] == "crop7" and len(blob["answers"]) == 6
    print("label views OK (crop7 data in its own directory, mixing refused)")

    # A variant trained on another run's shards: additive injection at another target layer.
    add_dir = os.path.join(tmp, "oracle_add")
    argv = [a if a != out_dir else add_dir for a in common]
    at0 = list(train)
    at0[at0.index("--target_layer") + 1] = "0"
    train_oracle.main(argv + at0 + ["--data_dir", os.path.join(out_dir, "data"), "--inject", "add",
                                    "--n_boot", "20", "--eval_images", "4"], mt=mt)
    assert not os.path.exists(os.path.join(add_dir, "data"))
    cfg = torch.load(os.path.join(add_dir, "lora.pt"), weights_only=False)["config"]
    assert cfg["inject"] == "add" and cfg["target_layer"] == 0
    a = np.load(os.path.join(add_dir, "eval", "cells", "untrained", "s2_t0.npz"))["pmi"]
    r = np.load(os.path.join(ev, "cells", "untrained", "s2_t1.npz"))["pmi"]
    assert a.shape == r.shape and not np.allclose(a, r)
    print("variant OK (shared data, additive injection, other target layer)")
  print("\nALL ORACLE TESTS PASSED")


if __name__ == "__main__":
  main()
