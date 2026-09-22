# Runbook: visual object identification (Patchscopes vs. logit lens vs. probe)

Script: `visual_object_identification.py` · Test: `test_visual_object_identification.py`

This is Step 1 of the evaluation plan in `Research/PROJECT_CONTEXT.md`: produce the first
numbers comparable to published work, so the project has something beyond the single dog
example.

## What it measures

At every LLM backbone layer, can the object that a visual patch token covers be read out of
that token's hidden state? Three readouts, **on identical cached states**:

| Readout | How it is scored |
|---|---|
| Patchscopes (this work) | patch the state into a text-only target prompt, generate 20 tokens, check whether the COCO class name appears (the original paper's attribute-extraction rule) |
| Logit lens (Neo et al.) | final norm + `lm_head` on the state; check top-1 and top-5 tokens against the class name's first sub-token |
| Linear probe | supervised logistic regression on the same states, grouped 5-fold by image |

Controls in the same run: patches **outside** the object, **norm-matched random** vectors,
and the target prompt with **no injection**. These set the chance level: a Patchscopes number
only means something if it is well above all three.

## Setup

Follows Neo et al. (arXiv 2410.07149 §4.1): COCO val2017, one object per image with area
20k–30k px², patches ≥50% covered by the object mask. Differences from their protocol, since
the paper does not fully specify it: the target object must be the only instance of its class
in the image, and the token-matching rule is the one defined in the script. **Their headline
number to beat: 23.7% of object-patch positions decode to the correct class token, best at
layer ~25.7 of 33.** `headline.json` reports the same quantity two ways (best layer overall,
and per-image best layer averaged) because their exact reading is ambiguous.

Target prompts are copied unchanged from the original Patchscopes notebooks, so no prompt
tuning has to be defended:
- `identity`: `"cat -> cat\n1135 -> 1135\nhello -> hello\n?"`
- `entity`: `"Syria: Country in the Middle East, ... Samsung: ..., x"`

Patching is same-layer (`layer_source == layer_target`), matching the original notebooks.

## Running it on SCC

```bash
conda activate patchscopes
cd /projectnb/cs505am/students/vishnuav/interpretability/patchscopes/code

# 1. smoke test on GPU: 3 images, 4 layers, writes overlays to check patch geometry
python visual_object_identification.py --coco_dir ../../../coco \
    --out_dir ./results/objid_smoke --n_images 3 --layer_step 8 --debug_overlays

# 2. full run
python visual_object_identification.py --coco_dir ../../../coco \
    --out_dir ./results/object_identification
```

COCO is downloaded automatically if absent (241MB annotation zip, then the ~170 images
individually). Point `--coco_dir` at an existing copy if the cluster already has one.

Scale of the full run: ~76k short generations (170 images × 6 patches × 32 layers × 2
prompts, plus random controls), batched at 64. Use `--layer_step 2` to halve it. Stages are
checkpointed: `patch_states.pt` is reused on re-runs unless `--overwrite` is passed.

## Check these before trusting the numbers

1. **Overlays** (`overlays/*.png`): the red patch grid must sit on the object. This is the
   check that was never done for the dog image.
2. **Verify lines** printed for the first image: captured states must equal HF's own
   `hidden_states`, and the visual tokens must start where expected (576 of them).
3. **`no_injection_baseline.csv`**: if the prompt alone already mentions class names, the
   accuracy numbers are contaminated.
4. **`outside` and `random` curves** in the plot: these should stay low. If the `object`
   curve is not clearly above them, there is no result yet.

## Outputs

| File | Contents |
|---|---|
| `samples.json` | the selected images, classes and patch indices |
| `patch_states.pt` | cached hidden states for the sampled patches (all layers) |
| `logit_lens.csv` | per patch × layer top-1/top-5 correctness |
| `patchscopes.csv` | every generation, with `correct` and `correct_strict` |
| `linear_probe.csv`, `no_injection_baseline.csv` | reference and control |
| `summary_by_layer.csv`, `object_identification.png` | the comparison table and plot |
| `headline.json` | the numbers to quote against Neo et al. |

## Known limitations

- Class-name matching is literal (plus simple plurals). "puppy" for a dog counts as wrong,
  so all methods are penalised, but Patchscopes generates free text and so is penalised
  hardest. If the numbers look low, inspect `patchscopes.csv` before concluding anything.
- The criteria are not identical across methods (single top token vs. text containing the
  name). Top-5 logit lens is reported to bracket this; state it plainly in any write-up.
- The probe is weak with ~170 images spread over many COCO classes; treat it as a rough
  reference, not a strong baseline.
- `test_visual_object_identification.py` runs the pipeline on a tiny random model, so it
  verifies plumbing (geometry, hook injection, batching, scoring), not correctness of results.
