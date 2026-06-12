# VLM Patchscopes: Complete Implementation Guide

> **What this document is:** A complete specification for adapting the Patchscopes mechanistic
> interpretability framework (Ghandeharioun et al., ICML 2024) from text-only LLMs to
> Vision-Language Models (VLMs). Written for Claude Code to implement from scratch.
>
> **Original codebase:** `interpretability/patchscopes/code/`  
> **Two core files:** `general_utils.py` + `patchscopes_utils.py` (1,157 lines)  
> **Target VLM architecture:** LLaVA-1.5 (llava-hf/llava-1.5-7b-hf and 13b variant)

---

## Part 1: What Patchscopes Does (The Original System)

Before describing changes, here is exactly what the original system does so every modification
makes sense in context.

### The Fundamental Operation

An LLM processes text by building up hidden vectors (also called hidden states or
representations) at every layer and every token position. These vectors are opaque
— they are just arrays of floating point numbers. The question Patchscopes asks is:
**what information is stored inside these vectors?**

The answer Patchscopes gives is: use the LLM itself to read its own vectors. You take
a hidden vector from one forward pass, inject it into a second forward pass that has
been designed to verbalize whatever is in that vector, and read what the model says.

Concretely, the operation has four steps:

```
Step 1: Run source prompt S through model M
        → capture hidden vector h at layer ℓ, token position i

Step 2: Optionally apply a mapping function f to h
        → f(h) is the vector you will inject
        → most experiments use f = identity (no change)

Step 3: Run target prompt T through model M*
        → at layer ℓ*, token position i*, REPLACE the hidden vector with f(h)
        → continue the forward pass from that layer onward

Step 4: Read what the model generates after the injection
        → this is the model's own "translation" of what was in h
```

The target prompt T is crafted to ask the question you want answered. For example:
- To ask "what entity is this vector about?": T = "Syria: Country in the Middle East,
  Leonardo DiCaprio: American actor, x:" — patch at position x
- To ask "what token will this predict?": T = "cat→cat; dog→dog; ?→" — patch at ?
- To ask "what is the capital of this country?": T = "The capital of x is" — patch at x

### The Two Real Files in the Codebase

**`general_utils.py`** — handles everything about loading models:
- The `ModelAndTokenizer` class wraps a HuggingFace CausalLM and its tokenizer
- It detects architecture automatically from the model class name
- It builds a list of layer names for each architecture
- Helper functions: tokenize inputs, find token positions, decode outputs

**`patchscopes_utils.py`** — handles everything about patching (1,157 lines):
- Six hook-setter functions (one per architecture × single/batch mode)
- `inspect()` — the main single-sample patching function
- Four batch evaluation functions (10x faster than single-sample loops)
- The batch functions accept a pandas DataFrame where each row is one
  (sample, layer_source, layer_target) combination

### How PyTorch Hooks Enable Patching

The patching is done with PyTorch forward hooks. A forward hook is a callback function
you attach to any PyTorch module. When that module finishes its forward computation,
the hook fires and can read or modify the output before it gets passed to the next layer.

This is the key mechanism:

```python
# The hook captures a reference to the vector we want to inject
# When the target layer finishes, this function fires automatically
def patch_hook(module, input, output):
    # output[0] has shape [batch_size, sequence_length, hidden_dim]
    # We replace just the target position with our injected vector
    output[0][0, target_position, :] = patched_vector
    return output

# Attach the hook to the specific layer we want to patch into
hook = model.layers[target_layer].register_forward_hook(patch_hook)

# Run the model — the hook fires automatically at the right moment
outputs = model(**inputs)

# CRITICAL: always remove the hook or it persists across future forward passes
hook.remove()
```

### The Six Existing Hook Setters

The codebase has six hook-setter functions, organized as two groups:

Group 1 — single sample (one prompt at a time):
- `set_hs_patch_hooks_llama()` — for LLaMA and Vicuna
- `set_hs_patch_hooks_neox()` — for Pythia
- `set_hs_patch_hooks_gptj()` — for GPT-J

Group 2 — batch mode (multiple prompts in one GPU forward pass):
- `set_hs_patch_hooks_llama_batch()`
- `set_hs_patch_hooks_neox_batch()`
- `set_hs_patch_hooks_gptj_batch()`

The reason there are separate functions per architecture is that different model families
store their transformer layers at different paths in the PyTorch module tree:
- LLaMA/Vicuna: `model.model.layers[i]`
- GPT-J: `model.transformer.h[i]`
- Pythia/NeoX: `model.gpt_neox.layers[i]`

### The Batch Config Format

The batch functions use a list of dicts — one per sample in the batch — to tell
the hook exactly what to do for each sample:

```python
hs_patch_config = [
    {
        "batch_idx": 0,           # which sample in the batch
        "layer_target": 15,       # which layer to patch into
        "position_target": -1,    # which token position (-1 = last token)
        "hidden_rep": tensor,     # the vector to inject (shape: [hidden_dim])
        "skip_final_ln": False    # whether to skip final layer norm
    },
    # ... one dict per sample in the batch
]
```

### The Standard Experiment Loop

Every notebook in the codebase follows the same five-step pattern:

```python
# Step 1: Load model
mt = ModelAndTokenizer("meta-llama/Llama-2-13b-hf")

# Step 2: Load dataset (TSV files in preprocessed_data/)
df = pd.read_csv("preprocessed_data/factual/country_capital_city.tsv", sep="\t")

# Step 3: Build experiment DataFrame
# One row per (sample, layer_source, layer_target) combination
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

# Step 4: Run batched evaluation
results = evaluate_attriburte_exraction_batch(mt, exp_df, batch_size=256)

# Step 5: Visualize as heatmap
pivot = results.groupby(["layer_source","layer_target"])["is_correct"].mean().unstack()
sns.heatmap(pivot, cmap="RdYlGn")
```

---

## Part 2: Why VLMs Are Different

A Vision-Language Model like LLaVA-1.5 has a fundamentally different architecture from
a text-only LLM. Understanding these differences is essential before touching any code.

### LLaVA-1.5 Architecture

```
Image Input
    ↓
ViT Encoder (CLIP-L/14@336px)
    ↓  produces 576 patch embeddings of dimension 1024
MLP Projector (2-layer MLP)
    ↓  maps each 1024-dim patch embedding → 4096-dim LLM token
Visual Tokens (576 tokens in the LLM's token space)
    ↓
    ↓  ←←← Text Tokens appended here
LLM Backbone (Vicuna-7B or 13B)
    ↓
Output Text
```

There are three structurally different stages:

**Stage 1 — Inside the ViT encoder:**
The image is split into a 24×24 grid of patches (576 patches for 336px input).
Each patch becomes a 1024-dimensional vector. These vectors "live in ViT space" —
they were trained by CLIP, not by the language model.

**Stage 2 — After the MLP projector:**
The projector maps each 1024-dim ViT vector to a 4096-dim vector that "lives in
LLM token space." From this point on, visual tokens are treated by the LLM backbone
exactly like text tokens — they participate in attention, go through transformer layers,
and get updated at each layer.

**Stage 3 — Inside the LLM backbone:**
Visual tokens and text tokens are processed together by the LLM's transformer layers.
At each layer, visual tokens attend to text tokens and vice versa. The representations
change as they propagate through the layers.

### Token Positions in LLaVA-1.5

When you give LLaVA an image and a text prompt, the combined input has this token layout:

```
Position 0:          BOS token
Positions 1–576:     Visual tokens (one per image patch, 24×24 grid)
Position 577:        Text token 1 (e.g., "USER")
Position 578:        Text token 2 (e.g., ":")
...and so on
```

This is critical for the patching operation. When you want to inspect what a specific
image patch "contains" according to the model, you need to address its token position
as `1 + patch_index` (where patch_index goes from 0 to 575).

### Three New Types of Representations

The original Patchscopes only deals with one type of hidden vector: text token
representations inside a text-only LLM. For VLMs, there are three types:

**Type 1: Visual token representations (post-projector, inside LLM backbone)**
These are the representations of visual tokens as they flow through the LLM backbone's
transformer layers. They have dimension 4096 (same as text tokens in Vicuna-7B).
These are the most natural to patch because they live in the same space as text tokens.

**Type 2: ViT hidden states (pre-projector, inside the vision encoder)**
These are the intermediate representations inside the ViT encoder itself.
They have dimension 1024 (CLIP-L hidden dimension).
To patch these into a text-only target prompt, you need a mapping function f
that maps from 1024-dim ViT space to 4096-dim LLM space.
The VLM's own MLP projector can serve as initialization for this mapping.

**Type 3: Cross-modal fused text token representations**
These are the representations of text tokens that have attended to visual tokens
inside the LLM backbone. For example, the word "it" in "describe it" after seeing
an image — this text token's hidden state encodes information absorbed from the
visual tokens through the attention mechanism.

### What Does NOT Change

The hook mechanism is completely model-agnostic. PyTorch hooks work on any nn.Module,
regardless of whether it is part of a VLM or a text-only LLM. The entire patching
logic — attaching hooks, injecting vectors, removing hooks — stays identical.

The target prompt is always text-only. The "target side" of a Patchscope is always
a text forward pass, because the goal is always to verbalize information in natural
language. This does not change for VLMs.

The DataFrame-based batch loop pattern stays identical. You still build a DataFrame
with one row per (sample, layer_source, layer_target) combination and pass it to a
batch evaluation function.

---

## Part 3: Every Change Required, Verbalized

### Change 1: Extend `ModelAndTokenizer` in `general_utils.py`

**What the current code does:**
The `ModelAndTokenizer` constructor takes a model name string, loads the HuggingFace
model and tokenizer, and detects the architecture by checking if certain strings appear
in the model class name. It builds a list of layer names based on the detected architecture.

**What needs to change:**
The constructor needs to also handle LLaVA-style VLMs. LLaVA is not a CausalLM —
it is a `LlavaForConditionalGeneration`. Its transformer layers live at
`model.language_model.model.layers`, not `model.model.layers`.

Additionally, for VLMs, the constructor needs to load a processor (not just a tokenizer).
A processor handles both image preprocessing (resizing, normalizing pixel values) and
text tokenization together. It is the LLaVA-specific replacement for a tokenizer.

The constructor also needs to store visual metadata so downstream code can work with
image patches: how many visual tokens are there (576 for LLaVA-1.5 at 336px), and
at which position do they start in the token sequence (position 1, after BOS).

**Concretely, add these attributes to `ModelAndTokenizer` for VLM models:**

```
mt.processor          → LlavaProcessor (handles image + text preprocessing)
mt.is_vlm             → True (boolean flag so other functions know this is a VLM)
mt.num_visual_tokens  → 576 (number of image patch tokens)
mt.visual_token_start → 1 (first visual token position, after BOS)
mt.vision_tower       → the ViT encoder module (model.vision_tower)
mt.projector          → the MLP projector module (model.multi_modal_projector)
mt.num_vision_layers  → number of layers in the ViT encoder
```

**Architecture detection addition:**

The current detection pattern:
```
if "llama" in class_name  →  layers at model.model.layers
if "gptj"  in class_name  →  layers at model.transformer.h
if "neox"  in class_name  →  layers at model.gpt_neox.layers
```

Add:
```
if "llava" in class_name  →  layers at model.language_model.model.layers
                               also load processor instead of tokenizer
                               also store visual metadata attributes
```

**Why this matters:** Every function in `patchscopes_utils.py` receives an `mt` object
and uses `mt.model` to access the model and `mt.layer_names` to navigate to layers.
If `ModelAndTokenizer` correctly handles LLaVA, everything downstream that uses `mt`
will automatically work with the VLM without needing model-specific code scattered
throughout the codebase.

---

### Change 2: Add Two New Hook Setter Functions in `patchscopes_utils.py`

**What the current code does:**
The six existing hook-setter functions all do the same conceptual thing — attach a
PyTorch hook to a specific transformer layer — but they navigate to that layer
differently depending on the model architecture.

For LLaMA, the path to layer i is `model.model.layers[i]`.
For GPT-J, the path is `model.transformer.h[i]`.
For Pythia, the path is `model.gpt_neox.layers[i]`.

**What needs to change:**
Add two new hook-setter functions for LLaVA:

**Function 1: `set_hs_patch_hooks_llava(mt, hs_patch_config)`**
For single-sample patching. Navigates to `model.language_model.model.layers[i]`
and attaches the same hook logic as the existing single-sample functions.
The hook itself is identical — it replaces `output[0][0, position_target, :]`
with the injected vector. Only the path to the layer changes.

**Function 2: `set_hs_patch_hooks_llava_batch(mt, hs_patch_config)`**
For batch-mode patching. Same as `set_hs_patch_hooks_llama_batch()` but navigating
to `model.language_model.model.layers[i]`.

**Why two functions instead of one:**
Single-sample mode and batch mode have different hook logic. In single-sample mode,
the hook replaces `output[0][0, position_target, :]` — always batch index 0.
In batch mode, the hook config contains a `batch_idx` field for each sample, and
the hook replaces `output[0][batch_idx, position_target, :]` for each sample
independently within the same forward pass.

**Important note about what does NOT change in the hook body:**
The actual replacement operation `output[0][batch_idx, position_target, :] = hidden_rep`
is identical for all architectures. The only thing that changes between architectures
is the path used to find and attach the hook. This is why the hook-setter functions
are separate from the hook body logic.

---

### Change 3: Add a New `get_hidden_state_vlm()` Function

**What the current code does:**
The existing code extracts hidden states implicitly inside `inspect()`. When you call
`inspect(mt, prompt_source, prompt_target, layer_source, layer_target, ...)`, the
function tokenizes the source prompt and runs a forward pass with
`output_hidden_states=True` to get all layer outputs, then indexes into
`outputs.hidden_states[layer_source][0, position_source, :]` to get the specific
hidden vector.

**What needs to change:**
For VLMs, extracting hidden states is more complex because:

1. The model takes both an image and text as input, so you need to use
   `processor(images=image, text=prompt, return_tensors="pt")` instead of just
   `tokenizer(prompt, return_tensors="pt")`.

2. For visual token positions, `position_source` is an index into the visual token
   region. The actual tensor position is `visual_token_start + position_source`
   (i.e., 1 + patch_index). For text token positions, you need to add an offset
   equal to the number of visual tokens to account for the fact that text tokens
   start after the visual tokens in the combined sequence.

3. Optionally, you may also want to extract hidden states from inside the ViT encoder
   itself (before the MLP projector). This requires a separate forward pass through
   `mt.vision_tower` with `output_hidden_states=True`, using only the image input
   without any text.

**New function signature:**

```python
def get_hidden_state_vlm(
    mt,                    # ModelAndTokenizer (VLM variant)
    image,                 # PIL Image
    text_prompt,           # string
    position,              # int — index within the modality-specific range
    layer,                 # int — which layer to extract from
    modality="visual"      # "visual", "text", or "vit" (pre-projector)
):
```

**How position is resolved based on modality:**

For `modality="visual"`:
The position argument is a patch index (0 to 575).
Actual tensor index = `mt.visual_token_start + position` = `1 + position`.
The hidden state is extracted from `outputs.hidden_states[layer][0, actual_position, :]`.
These are the visual token representations inside the LLM backbone (post-projector).

For `modality="text"`:
The position argument is a token index within the text portion.
Actual tensor index = `mt.visual_token_start + mt.num_visual_tokens + position`
= `1 + 576 + position` = `577 + position`.
These are text token representations inside the LLM backbone.

For `modality="vit"`:
The position argument is a patch index (0 to 575).
This requires running the vision tower separately:
`vit_outputs = mt.vision_tower(pixel_values, output_hidden_states=True)`
Then: `vit_outputs.hidden_states[layer][0, position, :]`
These are pre-projector representations inside the ViT encoder.
Their dimension is 1024 (CLIP-L hidden dim), not 4096.
A mapping function f is required to inject these into a text-conditioned target.

---

### Change 4: Add a New `inspect_vlm()` Function

**What the current code does:**
`inspect()` is the main single-sample patching function. It takes two string prompts
(source and target), extracts a hidden state from the source, and injects it into the
target. Everything is text-only.

**What needs to change:**
Add a new `inspect_vlm()` function that wraps `inspect()` for the VLM case.
The key difference is that the source side takes an image in addition to text,
and the source position is resolved differently depending on modality.
The target side stays text-only — you always inject into a text-conditioned forward pass.

**New function signature:**

```python
def inspect_vlm(
    mt,                    # ModelAndTokenizer (VLM variant)
    image,                 # PIL Image — the source image
    prompt_source,         # string — text part of source (e.g., "Describe this image")
    prompt_target,         # string — text-only target prompt (e.g., "Syria: ..., x:")
    layer_source,          # int — which LLM backbone layer to extract from
    layer_target,          # int — which layer to inject into
    position_source,       # int — patch index (0–575) or text position
    position_target,       # int — position in target prompt to inject at
    modality="visual",     # "visual", "text", or "vit"
    transform=None,        # mapping function f (None = identity)
    generation_mode=True,
    max_gen_len=50
):
```

**Internal flow of `inspect_vlm()`:**

Step 1: Extract hidden state using `get_hidden_state_vlm()` with the image,
text prompt, resolved position, and source layer. This gives you a vector h.

Step 2: Apply transform if provided: `h = transform(h)`.
For post-projector visual tokens, transform=None (identity) works because
the hidden state is already in 4096-dim LLM space.
For pre-projector ViT states (modality="vit"), a transform from 1024→4096 is required.

Step 3: Run the target prompt as a text-only input through the LLM backbone.
Attach the hook at layer_target, position_target.
Inject h at that position. Continue forward pass. Read output.

Step 4: Remove hooks. Return generated text.

**Why the target is always text-only:**
The whole point of Patchscopes is to verbalize hidden representations in natural language.
The target prompt is the "question" and the model's generation is the "answer."
Images are only on the source side — you are always asking the model to describe
something from an image using natural language, which means the target is a text prompt.

---

### Change 5: Add a New `evaluate_visual_attribute_extraction_batch()` Function

**What the current code does:**
`evaluate_attriburte_exraction_batch()` [note: the typo is in the original code]
accepts a pandas DataFrame, groups rows into batches, runs patched forward passes,
and returns a DataFrame with `generation` and `is_correct` columns.

**What needs to change:**
Add a new batch function for VLMs that handles the same flow but with images.
The DataFrame needs an additional `image_path` column so each row can specify
which image to load for that sample.

**New DataFrame schema for VLM experiments:**

```
Column            Type      Description
─────────────────────────────────────────────────────────────
image_path        string    Path to the source image file
prompt_source     string    Text prompt paired with the image
prompt_target     string    Text-only target/verbalization prompt
position_source   int       Patch index (0–575) or text position
position_target   int       Token position in target prompt
layer_source      int       Which LLM backbone layer to extract from
layer_target      int       Which layer to inject into
modality          string    "visual", "text", or "vit"
ground_truth      string    Expected attribute value (for scoring)
```

**Scoring for VLM attribute extraction:**
Same as the text case: does the ground truth string appear anywhere in the
first 20 generated tokens? (case-insensitive substring match)

---

### Change 6: Add a New `evaluate_visual_entity_resolution_batch()` Function

**What the current code does:**
`inspect_batch()` runs the entity description experiment for text entities.
For each (entity, layer) combination, it patches the entity's last-token hidden
state into a description target prompt and measures RougeL against Wikipedia.

**What needs to change:**
Add a visual analogue: for each (image_patch, layer) combination, patch the
visual patch token's hidden state into the description target prompt and
measure RougeL against a ground truth description of the image content.

**What this experiment measures:**
Just as the text version answers "how does the model's understanding of 'Diana,
Princess of Wales' evolve across layers?", the visual version answers
"how does the model's understanding of the image patch at position (x, y) evolve
across layers?" — starting from low-level visual descriptions ("a reddish circular
region") and progressing to semantic descriptions ("a Fuji apple on a wooden table").

**Ground truth for scoring:**
Use image captions (from COCO or Visual Genome) as the reference text for RougeL scoring,
the same way Wikipedia descriptions are used as reference in the text experiment.

---

### Change 7: Add a Mapping Function Builder for ViT→LLM Projection

**What this is:**
When extracting pre-projector ViT hidden states (modality="vit"), the hidden vectors
have dimension 1024 (CLIP-L) but the LLM backbone expects dimension 4096.
You cannot inject a 1024-dim vector into a position that expects 4096-dim without
first mapping it to the right dimension.

**The mapping function f:**
The simplest initialization for f is to use the VLM's own MLP projector, which
was already trained to map ViT space (1024-dim) to LLM token space (4096-dim).
This gives you a good starting point without any additional training.

For experiments requiring higher fidelity, you can fine-tune this mapping:
sample (ViT hidden state, LLM hidden state) pairs for the same semantic content
and optimize f to minimize reconstruction loss.

**Function to add:**

```python
def build_vit_to_llm_mapping(mt, mode="projector"):
    """
    Build a mapping function f: R^1024 → R^4096
    for cross-modal patching of pre-projector ViT states.

    mode="projector":  use the VLM's own MLP projector (no training needed)
    mode="affine":     learn a linear mapping via least squares (same as cross-model)
    mode="identity":   only works if vit_dim == llm_dim (not the case for LLaVA-1.5)
    """
```

---

### Change 8: Three New Experiment Notebooks

Three new Jupyter notebooks, each following the identical 5-step loop pattern
of the existing notebooks.

#### Notebook 1: `visual_concept_crystallization.ipynb`

**Analogue of:** `entity_processing.ipynb` (§4.3)

**Research question:**
How does a VLM's understanding of an image patch evolve across layers?
At which layer does a patch token representing "a dog's face" become semantically
recognizable as such? Does the model first see low-level visual properties
(color, texture) before recognizing objects, and then before recognizing
specific named entities?

**What the notebook does:**
1. Load LLaVA-1.5-7B
2. Load a dataset of images with associated region descriptions
   (Visual Genome has ~100k images with region-level descriptions)
3. For each image, select specific patch positions of interest (e.g., patches
   covering the main object)
4. For each patch × layer combination, run `inspect_vlm()` with the entity
   description target prompt: `"Syria: Country in the Middle East,
   Leonardo DiCaprio: American actor, x:"`
5. Measure RougeL between generated description and ground truth region description
6. Plot RougeL vs layer depth curves for popular vs rare visual entities

**Expected finding:**
Early LLM backbone layers (1-5): low-level descriptions ("a reddish circular shape")
Mid layers (5-15): object category ("a piece of fruit, possibly an apple")
Later layers (15+): specific entity ("a red Fuji apple on a wooden table")

This would be the first visualization of visual concept crystallization in VLMs,
directly analogous to the entity resolution experiment for text.

#### Notebook 2: `visual_attribute_extraction.ipynb`

**Analogue of:** `attribute_extraction.ipynb` (§4.2)

**Research question:**
Can we decode specific visual attributes (color, shape, object category, spatial
relationship, material) from a visual patch token's hidden state without training
any probes?

**Dataset:**
GQA or Visual Genome — both provide structured (image, region, attribute, value)
triplets, e.g., (image_001, region_045, color, red) or
(image_001, region_045, category, apple).

**Target prompts per attribute type:**
```
color:     "The color of the object in this region is:"
category:  "The object shown here is a type of:"
material:  "The material this object is made of is:"
shape:     "The shape of this object is:"
location:  "This object is located in the:"
relation:  "The spatial relationship shown here is:"
```

**Baseline comparisons:**
- Linear probe trained on visual token features (direct analogue of the text probe)
- CLIP zero-shot classification (baseline specific to visual models)
- Logit Lens applied to visual tokens (project via LLM unembedding matrix)

**Expected finding:**
Patchscopes outperforms linear probes in early and mid LLM backbone layers
(same pattern as text). Patchscopes strongly outperforms Logit Lens because visual
tokens were not trained to be directly decoded via the unembedding matrix.

#### Notebook 3: `cross_modal_fusion.ipynb`

**Analogue of:** `patch_cross_model.ipynb` (§4.4) but for a fundamentally new question

**Research question:**
At which LLM backbone layer do visual tokens and text tokens "fuse"? That is,
at which layer does a text token that refers to a visual entity (e.g., the pronoun
"it" referring to an object in the image) absorb visual information?

**How to measure fusion:**
Take the source prompt "There is an apple on the table. Describe it."
paired with an image of an apple.
Extract the hidden state of the word "it" at each layer.
Patch into target: "The object being referred to is:"
Track how the generated description changes from being a generic pronoun reference
to being a specific visual description ("a red Fuji apple").

The layer at which the generated description first matches the image content
is the "fusion point" — the layer where the text token absorbed visual information
through attention to visual patch tokens.

**Expected finding:**
The fusion point varies by model size and input complexity. For simple objects
it may happen in mid layers (~10-15 for a 32-layer model). For complex relational
descriptions it may happen later. This would be the first quantitative measurement
of cross-modal information flow in VLMs using natural language verbalization.

---

## Part 4: What the New Code Will Look Like End to End

Here is the complete flow of a visual attribute extraction experiment using the
new code, from raw inputs to results:

```
Input:
  image = PIL.Image.open("apple_on_table.jpg")
  text_prompt = "USER: Describe the image. ASSISTANT:"
  ground_truth_attribute = "red"
  attribute_type = "color"

Step 1: Load model
  mt = ModelAndTokenizer("llava-hf/llava-1.5-7b-hf")
  # mt.is_vlm = True
  # mt.num_visual_tokens = 576
  # mt.visual_token_start = 1
  # mt.num_layers = 32 (LLaVA-1.5-7B has 32 LLM backbone layers)

Step 2: Extract hidden state from visual patch token
  patch_index = 200  # the patch covering the apple (center of image)
  for layer_source in range(32):
      h = get_hidden_state_vlm(
          mt, image, text_prompt,
          position=patch_index,
          layer=layer_source,
          modality="visual"
      )
      # h.shape = [4096] — post-projector visual token representation

Step 3: Build target prompt
  target_prompt = "The color of the object in this region is:"
  position_target = -1  # last token position (the ":" before generation)

Step 4: Run patchscope for each (layer_source, layer_target) combination
  for layer_target in range(32):
      generated = inspect_vlm(
          mt,
          image=image,
          prompt_source=text_prompt,
          prompt_target=target_prompt,
          layer_source=layer_source,
          layer_target=layer_target,
          position_source=patch_index,
          position_target=position_target,
          modality="visual",
          transform=None,  # identity — post-projector, no dim mismatch
          generation_mode=True,
          max_gen_len=10
      )
      # generated might be: "red" / "reddish" / "green" / "fruit-colored"
      is_correct = ground_truth_attribute.lower() in generated.lower()

Step 5: Visualize
  # 32×32 heatmap of accuracy across (layer_source, layer_target) combinations
  # Same visualization as attribute_extraction.ipynb
```

---

## Part 5: File-by-File Summary of All Changes

```
general_utils.py
├── MODIFY: ModelAndTokenizer.__init__()
│   ├── ADD: "llava" architecture detection
│   ├── ADD: load LlavaProcessor instead of tokenizer for VLMs
│   ├── ADD: mt.is_vlm = True/False
│   ├── ADD: mt.processor attribute
│   ├── ADD: mt.num_visual_tokens = 576
│   ├── ADD: mt.visual_token_start = 1
│   ├── ADD: mt.vision_tower attribute
│   ├── ADD: mt.projector attribute
│   └── ADD: mt.num_vision_layers attribute
└── ADD: get_visual_token_position(mt, patch_index)
    └── returns actual tensor position: visual_token_start + patch_index

patchscopes_utils.py
├── ADD: set_hs_patch_hooks_llava(mt, hs_patch_config)
│   └── single-sample hook for LLaVA LLM backbone layers
├── ADD: set_hs_patch_hooks_llava_batch(mt, hs_patch_config)
│   └── batch hook for LLaVA LLM backbone layers
├── ADD: get_hidden_state_vlm(mt, image, text_prompt, position, layer, modality)
│   ├── handles modality="visual"  → LLM backbone visual token position
│   ├── handles modality="text"    → LLM backbone text token position
│   └── handles modality="vit"     → ViT encoder hidden state (pre-projector)
├── ADD: inspect_vlm(mt, image, prompt_source, prompt_target, ...)
│   └── VLM-aware wrapper around the core patching logic
├── ADD: evaluate_visual_attribute_extraction_batch(mt, df, batch_size)
│   └── batch version of inspect_vlm for attribute extraction experiments
├── ADD: evaluate_visual_entity_resolution_batch(mt, df, batch_size)
│   └── batch version for visual concept crystallization experiments
└── ADD: build_vit_to_llm_mapping(mt, mode="projector")
    └── builds mapping function f for pre-projector ViT patching

New notebooks (each follows the existing 5-step loop pattern):
├── visual_concept_crystallization.ipynb
│   └── how do visual patch representations evolve across LLM backbone layers?
├── visual_attribute_extraction.ipynb
│   └── can we decode color/shape/category from visual tokens without probes?
└── cross_modal_fusion.ipynb
    └── at which layer do text tokens absorb visual information?
```

---

## Part 6: What Makes This Novel Research

The key research contributions of this VLM adaptation are:

**Contribution 1: First natural-language visualization of visual concept crystallization**
Existing work can tell you "does layer 10 encode object category?" (binary probe answer).
This work shows you "here is what the model thinks this image patch contains at each layer,
expressed in natural language" — a continuous, expressive view of visual understanding.

**Contribution 2: Training-free visual attribute extraction**
Existing visual probing requires labeled training data for every attribute type.
Patchscopes requires zero training data and works with open vocabulary.

**Contribution 3: Cross-modal fusion localization in natural language**
Attention visualization shows WHERE the model looks. This work shows WHAT the model
understands, expressed in natural language, at each layer of the fusion process.

**Contribution 4: The first extension of the Patchscopes framework to multimodal models**
The formal framework (S, i, M, ℓ) → (T, i*, f, M*, ℓ*) is extended with a modality
dimension, establishing a new general framework for multimodal mechanistic interpretability.

---

## Part 7: Datasets Needed

| Dataset | Used In | What It Provides |
|---|---|---|
| Visual Genome | visual_attribute_extraction | (image, region, attribute, value) triplets |
| GQA | visual_attribute_extraction | Structured visual QA with attribute annotations |
| MS-COCO | visual_concept_crystallization | Image captions for RougeL scoring |
| OK-VQA | cross_modal_fusion | Questions requiring image+knowledge reasoning |
| VisualGenome regions | visual_concept_crystallization | Region-level descriptions for patches |

---

## Part 8: Environment

```
Base environment: same as original (Python 3.10.13, CUDA 11.3)

Additional dependencies for VLM adaptation:
pip install transformers>=4.36.0  # LLaVA support added in 4.36
pip install Pillow                # image loading
pip install rouge-score           # RougeL evaluation
pip install sentence-transformers # SBERT evaluation
pip install pycocoevalcap        # COCO caption evaluation (optional)

Model downloads:
huggingface-cli download llava-hf/llava-1.5-7b-hf
huggingface-cli download llava-hf/llava-1.5-13b-hf  # for cross-model experiments
```

---

## Part 9: Honest Assessment of Where This Will and Will Not Work

### Where VLM Patchscopes will clearly outperform baselines

- Early and mid LLM backbone layers for visual attribute extraction
  (same pattern as text — linear probes fail here, Patchscopes does not)
- Any question requiring open-vocabulary output
  (probes need a fixed class set; Patchscopes generates freely)
- Visual concept crystallization visualization
  (no existing method provides natural language layer-by-layer descriptions)
- Compared to Logit Lens on visual tokens
  (visual tokens are not trained to be decoded via the unembedding matrix)

### Where VLM Patchscopes may struggle

- Late LLM backbone layers
  (same late-layer degradation as text — representations shift toward next-token prediction)
- Fine-grained visual distinctions
  (counting exact numbers, distinguishing similar breeds — the LLM may lack resolution)
- Pre-projector patching without a trained mapping
  (using the raw projector as f without fine-tuning may give incoherent results)
- Rare visual concepts
  (analogous to rare entities in §4.3 — models trained on less data for unusual objects)
- Placeholder contamination
  (same limitation as text §4.3 — the "x" placeholder can interfere in later layers)

### The one genuinely unknown question

Whether the modality gap — the distributional difference between visual token
representations and text token representations even after projection — will cause
the patching to produce incoherent outputs when a visual hidden state is injected
into a text-conditioned target pass. This is the central empirical unknown.
The hypothesis (based on the success of cross-model patching in §4.4) is that
if the projector has done its job, visual tokens in mid-to-late LLM backbone layers
are close enough to text tokens in representation space that the patching will work.
This needs to be verified empirically and is itself a novel finding either way.
