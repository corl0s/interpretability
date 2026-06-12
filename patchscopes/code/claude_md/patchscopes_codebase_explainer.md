# Patchscopes Codebase Explainer

> **Codebase:** `interpretability/patchscopes/code/`  
> **Paper:** Ghandeharioun et al., ICML 2024 — arXiv:2401.06102  
> **Purpose:** Complete reference for understanding how the code implements the paper.

---

## 1. Repository Layout

```
code/
├── general_utils.py                  # Model loading + tokenization helpers
├── patchscopes_utils.py              # Core patching implementation
├── apply_delta.py                    # One-time script: assemble Vicuna model
├── download_the_pile_text_data.py    # One-time script: download The Pile dataset
├── requirements.txt                  # Python 3.10.13 environment
├── next_token_prediction.ipynb       # Experiment 1 (§4.1)
├── attribute_extraction.ipynb        # Experiment 2 (§4.2)
├── entity_processing.ipynb           # Experiment 3 (§4.3)
├── patch_cross_model.ipynb           # Experiment 4 (§4.4)
├── multihop-CoT.ipynb                # Experiment 5 (§5)
└── preprocessed_data/
    ├── commonsense/                  # 8 TSV datasets
    ├── factual/                      # 12 TSV datasets
    └── factual_multihop/             # 1 TSV dataset
```

---

## 2. What Patchscopes Does (Theory → Code)

### The Core Operation

The framework extracts a hidden vector from one forward pass and injects it into another, using the LLM's own computation as a "decoder" of that vector.

**Formally:**
1. Run source prompt **S** through model **M** → collect hidden state **h^ℓ_i** at layer `ℓ`, position `i`
2. Apply optional mapping **f(h)** (identity or learned affine)
3. Run target prompt **T** through model **M\*** → at layer `ℓ*`, position `i*`, **replace** the hidden state with **f(h^ℓ_i)**
4. Read the output after that layer

```
Source pass:   S → [layer 0] → ... → [layer ℓ] → h^ℓ_i  (captured)
                                                        ↓
                                                     f(h^ℓ_i)
                                                        ↓
Target pass:   T → [layer 0] → ... → [layer ℓ*] → REPLACE → [layer ℓ*+1] → ... → output
```

### Paper Parameters → Code Arguments

| Paper Symbol | Meaning | Code Argument |
|---|---|---|
| **S** | Source prompt | `prompt_source` |
| **i** | Source token position | `position_source` |
| **M** | Source model | `mt` (ModelAndTokenizer) |
| **ℓ** | Source layer | `layer_source` |
| **T** | Target prompt | `prompt_target` |
| **i\*** | Target token position | `position_target` |
| **M\*** | Target model | `mt` (same) or separate model |
| **ℓ\*** | Target layer | `layer_target` |
| **f** | Mapping function | `transform` (None = identity) |

---

## 3. `general_utils.py` — Model & Tokenization

### `ModelAndTokenizer`

The central wrapper object passed to every patching function.

```python
mt = ModelAndTokenizer("meta-llama/Llama-2-13b-hf")
# mt.model        → the CausalLM
# mt.tokenizer    → the tokenizer
# mt.layer_names  → ["model.layers.0", "model.layers.1", ...]
# mt.num_layers   → 40
```

**Architecture detection** — the constructor inspects the model class name to decide which attribute holds the layer list:

| Architecture | Layer path | Detection key |
|---|---|---|
| Llama / Vicuna | `model.model.layers` | `"llama"` in class name |
| GPT-J | `model.transformer.h` | `"gptj"` in class name |
| Pythia / NeoX | `model.gpt_neox.layers` | `"neox"` in class name |

### Helper Functions

| Function | Purpose |
|---|---|
| `make_inputs(tokenizer, prompts, device)` | Left-pads a list of strings → `input_ids` + `attention_mask` tensors |
| `decode_tokens(tokenizer, token_ids)` | Token ID list → list of string tokens |
| `find_token_range(tokenizer, token_array, substring)` | Returns `(start, end)` token indices for a substring |
| `predict_from_input(model, inp)` | Returns `(predictions, probabilities)` — the argmax token and its softmax score |
| `set_requires_grad(requires_grad, *models)` | Disables gradients for inference-only use |

---

## 4. `patchscopes_utils.py` — Patching Engine

This is the main file (1,157 lines). It has three layers of abstraction:

```
Hook Setters  →  inspect() / evaluate_*()  →  batch variants
(low level)      (single sample)               (10× faster)
```

### 4.1 Hook Setters

PyTorch `register_forward_hook` intercepts the output of a layer mid-pass and replaces the hidden state at the target position:

```python
def patch_hook(module, input, output):
    output[0][0, target_position, :] = patched_vector
    return output

hook = model.layers[target_layer].register_forward_hook(patch_hook)
outputs = model(**inputs)
hook.remove()
```

Six hook-setter functions handle architecture differences and single vs. batch mode:

| Function | Architecture | Mode |
|---|---|---|
| `set_hs_patch_hooks_neox()` | Pythia / NeoX | single sample |
| `set_hs_patch_hooks_llama()` | Llama / Vicuna | single sample |
| `set_hs_patch_hooks_gptj()` | GPT-J | single sample |
| `set_hs_patch_hooks_neox_batch()` | Pythia / NeoX | batch |
| `set_hs_patch_hooks_llama_batch()` | Llama / Vicuna | batch |
| `set_hs_patch_hooks_gptj_batch()` | GPT-J | batch |

**Batch hook config format** — one dict per sample in the batch:

```python
hs_patch_config = [
    {
        "batch_idx": 0,              # index within the batch
        "layer_target": 15,          # which layer to patch into
        "position_target": -1,       # token position (-1 = last)
        "hidden_rep": tensor,        # the vector to inject
        "skip_final_ln": False       # whether to skip the final layer norm
    },
    ...
]
```

Always call `remove_hooks(hooks)` after each forward pass to clean up.

### 4.2 Single-Sample Functions

#### `inspect()`
The most general function. Runs end-to-end patching and returns the model's output.

```python
output = inspect(
    mt,                    # ModelAndTokenizer
    prompt_source,         # string
    prompt_target,         # string
    layer_source,          # int
    layer_target,          # int
    position_source,       # int (negative = from end)
    position_target,       # int
    module="hs",           # "hs" = hidden state (only option used)
    generation_mode=False, # True → generate up to max_gen_len tokens
    max_gen_len=20,
    verbose=False,
    temperature=None
)
```

**Internal flow:**
1. Tokenize `prompt_source` → forward pass → capture hidden state at `[layer_source][position_source]`
2. Apply `transform` if provided
3. Tokenize `prompt_target` → register patch hook at `layer_target` / `position_target`
4. Run forward pass (or `model.generate()`) with hook active
5. Remove hook → return decoded output

#### `evaluate_patch_next_token_prediction()`
Returns `(is_correct, surprisal)` for a single source/target pair. Used in §4.1 experiments.

### 4.3 Batch Functions

Batch functions accept a **pandas DataFrame** where each row is one (sample, layer_source, layer_target) combination. They group rows into batches, run a single forward pass per batch, and return results.

| Function | Task | Output |
|---|---|---|
| `evaluate_patch_next_token_prediction_batch()` | Next-token accuracy | DataFrame with `is_correct`, `surprisal` columns |
| `evaluate_attriburte_exraction_batch()` | Attribute extraction via generation | DataFrame with `generation`, `is_correct` columns |
| `evaluate_attriburte_exraction_batch_multihop()` | Multi-hop reasoning | Same as above |
| `inspect_batch()` | Batch generation | DataFrame with `generation` column |
| `evaluate_patch_next_token_prediction_x_model()` | Cross-model patching | DataFrame with accuracy columns |

**Why batch is faster:** Multiple (source_layer, target_layer) combinations for the same prompt can be packed into one padded batch. The hook config carries per-sample metadata so each sample in the batch gets its own patch target.

---

## 5. Preprocessed Data

All datasets are TSV files with the same schema:

| Column | Type | Description |
|---|---|---|
| `sample_id` | int | Unique identifier |
| `prompt_source` | string | The source prompt (Wikipedia sentence containing the subject) |
| `position_source` | int | Token position of the subject in `prompt_source` |
| `prompt_target` | string | The target prompt (relation verbalization with placeholder) |
| `position_target` | int | Token position of the placeholder in `prompt_target` |
| `subject` | string | The entity being queried (σ) |
| `object` | string | The correct answer (ω) |
| `target_baseline` | string | Expected next token in the unpatched target prompt |
| `generations_baseline` | list | Generations from unpatched model |
| `is_correct_baseline` | bool | Whether unpatched model gets it right |
| `source_cropped_toks` | list | Tokenized source around position_source |

### Dataset Categories

**Commonsense** (8 datasets, ~500 samples each):
`fruit_inside_color`, `fruit_outside_color`, `object_superclass`, `substance_phase`,
`task_done_by_person`, `task_done_by_tool`, `word_sentiment`, `work_location`

**Factual** (12 datasets, ~500–1000 samples each):
`company_ceo`, `country_capital_city`, `country_currency`, `country_largest_city`,
`food_from_country`, `person_father`, `person_mother`, `person_plays_position_in_sport`,
`person_plays_pro_sport`, `pokemon_evolutions`, `product_by_company`, `star_constellation`,
`superhero_archnemesis`, `superhero_person`

**Multi-hop** (1 dataset):
`factual_multihop/multihop_CoT_vicuna-13b-v1.1.tsv` — chains two single-hop relations

---

## 6. The Five Experiments

### Experiment 1 — `next_token_prediction.ipynb` (§4.1)

**Goal:** At which layer does the model commit to its final prediction?

**Method:** Token Identity Patchscope
- Source: any prompt from The Pile
- Target: `"tok1 → tok1 ; tok2 → tok2 ; ... ; tokk → ?"`
- Position: last token (`position_target = -1`)
- No training required; `transform = None`

**Key result:** Setting `layer_target = layer_source` (same layer) outperforms Logit Lens and Tuned Lens by up to 98% gain in layers 18–22 across all tested models.

**Why:** Logit Lens projects from intermediate layers directly to vocabulary — it skips the remaining transformer computation. Using `ℓ* = ℓ` lets all remaining layers process the patched state, which recovers information that hasn't fully propagated to the final layer yet.

**Output:** CSV files with (layer_source × layer_target) accuracy matrices.

---

### Experiment 2 — `attribute_extraction.ipynb` (§4.2)

**Goal:** Can factual/commonsense attributes be decoded from a subject's hidden representation without any training?

**Method:** Zero-Shot Feature Extraction Patchscope
- Source: Wikipedia sentence containing subject σ; extract last-token hidden state of σ
- Target: relation verbalization, e.g., `"The largest city in x"` or `"x works as a"`
- `transform = None` (zero-shot, no learned mapping)
- Evaluate over all 20 datasets × all (layer_source, layer_target) combinations

**Scoring:** Does the true object ω appear anywhere in the 10-token generation?

**Key result:**
- Layers 1–10: Patchscopes >> linear probe (probe fails here)
- Layers 10–20: Patchscopes ≈ probe
- Layers 20+: probe ≥ Patchscopes (late layers shift toward next-token prediction, not attribute storage)

**Output:** Heatmaps of (layer_source × layer_target) accuracy per dataset.

---

### Experiment 3 — `entity_processing.ipynb` (§4.3)

**Goal:** How does a model gradually resolve an entity name as it processes through layers?

**Method:** Entity Description Patchscope
- Source: text containing entity name; extract last token of entity name
- Target: `"Syria: Country in the Middle East, Leonardo DiCaprio: American actor, x:"`
- Patch at position `x`; measure RougeL vs. Wikipedia description of the entity

**Example (Diana, Princess of Wales):**
| Layer | Generated | Interpretation |
|---|---|---|
| 1–2 | "Country in the United Kingdom" | Sees only "Wales" |
| 3 | "Country in Europe" | Still "Wales" |
| 4–5 | "Title held by female sovereigns" | Princess of Wales |
| 6+ | "Diana, Princess of Wales (1961–1997)..." | Fully resolved |

**Known limitation:** "Placeholder contamination" — the placeholder token `x` retains its own representation in early target layers, causing interference in later layers.

**Output:** CSV with per-layer RougeL, Rouge1, SBERT scores.

---

### Experiment 4 — `patch_cross_model.ipynb` (§4.4)

**Goal:** Can a larger model verbalize hidden representations extracted from a smaller model?

**Setup:** M = Vicuna-7B (source), M\* = Vicuna-13B (target)

**Method:**
1. Sample 100K pairs from The Pile
2. Run both models; collect hidden states from corresponding layers
3. Learn affine mapping `f(h) = Ah + b` via `numpy.linalg.lstsq`
4. Apply learned `f` as the `transform` argument during cross-model patching

**Key result:** Diagonal of (source_layer, target_layer) matrix performs best — there is layer-to-layer correspondence within model families. Patching into early layers of the larger model is most effective.

**Output:** Learned transformation matrices + cross-model accuracy CSVs.

---

### Experiment 5 — `multihop-CoT.ipynb` (§5)

**Goal:** Fix multi-hop reasoning failures using patchscopes.

**Setup:** Query = [π2][π1], e.g., "The current CEO of the company that created Visual Basic Script"
- τ1 = (Visual Basic Script, product of, Microsoft)
- τ2 = (Microsoft, CEO, Satya Nadella)

**Method (CoT Patchscope):**
1. Run π1 alone as source → extract last-token hidden state (encodes "Microsoft")
2. Target = full query [π2][π1]; patch at position just before π1
3. Attention mask: π1 tokens cannot attend to π2 tokens

**Results:**
| Method | Accuracy |
|---|---|
| Vanilla baseline | 19.57% |
| Chain-of-Thought ("Let's think step by step") | 35.71% |
| CoT Patchscope | **50.00%** |

**Output:** Heatmaps showing which (layer_source, layer_target) combinations enable multi-hop reasoning.

---

## 7. Supported Models

| Model | Size | Layers | Hidden Dim | Architecture | Hook path |
|---|---|---|---|---|---|
| LLaMA-2 | 13B | 40 | 5120 | Llama | `model.model.layers[i]` |
| Vicuna | 7B/13B | 32/40 | 4096/5120 | Llama (fine-tuned) | `model.model.layers[i]` |
| GPT-J | 6B | 28 | 4096 | GPT-J | `model.transformer.h[i]` |
| Pythia | 6.9B/12B | 32/36 | 4096/5120 | NeoX | `model.gpt_neox.layers[i]` |

---

## 8. Setup Instructions

```bash
# Install dependencies (Python 3.10.13, CUDA 11.3)
pip install -r requirements.txt

# Download The Pile data (needed for Experiments 1 and 4)
python3 download_the_pile_text_data.py \
    --output_path ./the_pile_deduplicated \
    --num_samples 200000

# Assemble Vicuna from delta weights (needed for Experiments 4 and 5)
python3 apply_delta.py \
    --base meta-llama/Llama-2-13b-hf \
    --target ./stable-vicuna-13b \
    --delta CarperAI/stable-vicuna-13b-delta

# Run any experiment
jupyter notebook attribute_extraction.ipynb
```

---

## 9. Typical Experiment Loop (Code Pattern)

Every notebook follows the same pattern:

```python
# 1. Load model
mt = ModelAndTokenizer("meta-llama/Llama-2-13b-hf")

# 2. Load dataset
df = pd.read_csv("preprocessed_data/factual/country_capital_city.tsv", sep="\t")

# 3. Build experiment DataFrame
# Each row = one (sample, layer_source, layer_target) combination
rows = []
for _, row in df.iterrows():
    for l_s in range(mt.num_layers):
        for l_t in range(mt.num_layers):
            rows.append({
                "prompt_source": row["prompt_source"],
                "prompt_target": row["prompt_target"],
                "position_source": row["position_source"],
                "position_target": row["position_target"],
                "layer_source": l_s,
                "layer_target": l_t,
                "object": row["object"],
            })
exp_df = pd.DataFrame(rows)

# 4. Run batched evaluation
results = evaluate_attriburte_exraction_batch(mt, exp_df, batch_size=256)

# 5. Visualize
pivot = results.groupby(["layer_source", "layer_target"])["is_correct"].mean().unstack()
sns.heatmap(pivot, cmap="RdYlGn")
```

---

## 10. Key Design Decisions

| Decision | Reason |
|---|---|
| PyTorch hooks instead of model modification | Non-invasive; works with any frozen HuggingFace model with no code changes |
| `ℓ* = ℓ` (same layer for source and target) | Lets remaining transformer computation act as a "decoder" — the main advantage over Logit Lens |
| Per-sample hook config in batch | Allows different (layer_source, layer_target) combinations in a single GPU forward pass |
| `transform = None` for most experiments | Zero-shot — no training data needed; learned affine only added for cross-model experiments |
| Negative position indexing | `-1` always refers to the last token regardless of prompt length — robust across variable-length inputs |

---

## 11. Known Limitations

1. **Placeholder contamination** — the `x` placeholder token in target prompts retains its own representation in early layers, causing interference in later layers during entity resolution experiments.
2. **Late layer degradation** — representations in layers 20+ shift toward next-token prediction; attribute information becomes harder to extract.
3. **Cross-architecture patching** — affine mapping works within families (7B→13B Vicuna) but hasn't been validated across different architectures (e.g., Llama→GPT-J).
4. **Early layer noise** — layers 1–10 perform poorly across all methods; input contextualization dominates there.
5. **Top-1 faithfulness** — the correct interpretation is not always the argmax prediction; it often appears in the top-10 tokens.
