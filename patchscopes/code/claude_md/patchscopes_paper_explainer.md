# Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models

> **Paper:** Ghandeharioun et al., ICML 2024  
> **arXiv:** 2401.06102  
> **Code:** https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code  
> **Purpose of this document:** Complete technical reference for implementing and extending Patchscopes, specifically for adapting it to Vision-Language Models (VLMs).

---

## 1. Core Idea in One Paragraph

LLMs are good at generating human-readable text. Instead of training external probes or projecting hidden states to vocabulary space to understand what information is stored inside an LLM, Patchscopes **uses the LLM itself as the inspection tool**. You extract a hidden representation (a vector) from one forward pass (the "source"), inject it into a second, carefully designed forward pass (the "target"), and read what the model generates. The target prompt is designed to ask exactly the question you want answered. The model's own forward computation after injection acts as a "translator" of the hidden vector into natural language.

---

## 2. The Formal Framework

### 2.1 Source Side
Defined by the tuple **(S, i, M, ℓ)**:
- **S** — source input prompt (e.g., "Amazon's former CEO attended the Oscars")
- **i** — token position to inspect (e.g., position of "CEO")
- **M** — source model (e.g., Vicuna-13B)
- **ℓ** — layer to extract hidden state from (e.g., layer 15)

This gives you a hidden vector **h^ℓ_i** of shape `[hidden_dim]`.

### 2.2 Target Side
Defined by the quintuplet **(T, i*, f, M*, ℓ*)**:
- **T** — target prompt designed to verbalize information (e.g., "cat→cat; dog→dog; ?→")
- **i*** — token position in T where the patch is injected (e.g., position of "?")
- **f** — mapping function applied to h^ℓ_i before injection (usually identity: f(h)=h)
- **M*** — target model (can be same as M, or a stronger model)
- **ℓ*** — layer in M* where injection happens

### 2.3 The Patching Operation
```
h̄^ℓ*_i* ← f(h^ℓ_i)
```
At layer ℓ* during M*'s forward pass on T, replace the hidden state at position i* with f(h^ℓ_i), then continue the forward computation from that layer onward and read the generated output.

### 2.4 Implementation via PyTorch Hooks
The patching is implemented using **PyTorch forward hooks** — callbacks that fire when a specific module completes its forward pass, allowing interception and replacement of activations without modifying model code:

```python
def patch_hook(module, input, output):
    # output[0] shape: [batch, seq_len, hidden_dim]
    output[0][0, target_position, :] = patched_vector
    return output

hook = model.layers[target_layer].register_forward_hook(patch_hook)
outputs = model.generate(**inputs)
hook.remove()  # ALWAYS remove after use
```

---

## 3. How Prior Methods Are Special Cases

| Method | S | T | f | M* | ℓ* |
|---|---|---|---|---|---|
| **Logit Lens** | any | any (doesn't matter) | identity | M | L (final layer) |
| **Tuned Lens** | any | any | learned affine Aℓh+bℓ | M | L (final layer) |
| **Future Lens** | any | fixed soft prompt | learned linear | M | ℓ (same as source) |
| **Causal Tracing** | S+noise | S+Gaussian noise | identity | M | ℓ |
| **Attention Knockout** | any | S | f(h)=0 | M | multiple layers |
| **Token Identity (new)** | any | "tok1→tok1; tok2→tok2; ?" | identity | M | ℓ (same as source) |
| **Entity Description (new)** | any | "Syria: Country..., x:" | identity | M | ℓ |
| **X-Model (new)** | any | "Syria: Country..., x:" | learned affine | larger M* | ℓ |

**Key insight:** Prior methods are constrained to either T=S (same prompt) or ℓ*=L (final layer). Patchscopes removes both constraints.

---

## 4. Experiments and Results

### 4.1 Next-Token Prediction (§4.1)
**Question:** At which layer does the model "commit" to its final prediction?

**Token Identity Patchscope:**
- T = `"tok1 → tok1 ; tok2 → tok2 ; ... ; tokk → ?"` (random tokens as demonstrations)
- i* = position of "?"
- ℓ* = ℓ (same as source layer — key difference from baselines)
- f = identity
- No training required

**Result:** From layer 10 upward, Token Identity Patchscope outperforms Logit Lens and Tuned Lens by up to 98% gain in layers 18-22 across LLaMA2-13B, Vicuna-13B, Pythia-12B, GPT-J-6B.

**Why it works:** Setting ℓ*=ℓ means the model's full remaining computation runs, which does important processing that jumping straight to the unembedding matrix (ℓ*=L) skips.

**Metrics:**
- Precision@1 (↑): does argmax of estimated distribution match argmax of true final distribution?
- Surprisal (↓): -log p^L_t̃ where t̃ = argmax of estimated distribution

### 4.2 Attribute Extraction (§4.2)
**Question:** Can we decode factual attributes from a subject's hidden representation without training?

**Setup:** Knowledge triplets (σ, ρ, ω) e.g., (United States, largest city, New York City). Source S is a Wikipedia sentence containing σ. Extract last-token representation of σ.

**Zero-Shot Feature Extraction Patchscope:**
- T = relation verbalization with placeholder, e.g., `"The largest city in x"`
- i* = position of "x"
- ℓ* ∈ [1,...,L*] (searched over all combinations)
- f = identity

**Result:** Outperforms logistic regression probe on 6/12 tasks (p<1e-5) with NO training data. Works comparably on 5/6 remaining tasks. Consistently better in early/mid layers; worse only in late layers (where representations shift toward next-token prediction).

**Key finding about layer patterns:**
- Early layers (1-10): Patchscope >> Probe (probe fails here, Patchscope doesn't)
- Mid layers (10-20): Patchscope ≥ Probe
- Late layers (20+): Probe ≥ Patchscope (representations shift toward next-token prediction)

### 4.3 Entity Resolution in Early Layers (§4.3)
**Question:** How does an LLM gradually resolve an entity name across layers?

**Entity Description Patchscope:**
- T = `"Syria: Country in the Middle East, Leonardo DiCaprio: American actor, Samsung: South Korean multinational..., x:"`
- i* = position of "x" (last position)
- ℓ* = ℓ (same layer)
- Source position = last token of entity name in S

**Example result for "Diana, Princess of Wales":**
| Layer | Generated | Interpretation |
|---|---|---|
| 1-2 | "Country in the United Kingdom" | Model only sees "Wales" |
| 3 | "Country in Europe" | Still just Wales |
| 4 | "Title held by female sovereigns" | Princess of Wales (unspecific) |
| 5 | "Title given to wife of Prince of Wales" | More specific |
| 6 | "Diana, Princess of Wales (1961-1997)..." | Fully resolved |

**Quantitative:** RougeL between generated descriptions and Wikipedia descriptions peaks around layer 5, then decreases (due to "placeholder contamination" — see §4.3 limitations).

**Placeholder contamination:** The placeholder token "x" in T retains its representation in early layers, which can interfere with generation in later layers, causing the model to describe "x" itself rather than the patched entity. This is a known limitation.

### 4.4 Cross-Model Patching (§4.4)
**Question:** Can a stronger model M* verbalize representations from a weaker model M?

**Setup:** M=Vicuna-7B, M*=Vicuna-13B. Learn affine mapping f between corresponding layers. f(h) = Aℓh + bℓ trained to align representations.

**Result:** Cross-model patching improves entity resolution for both popular and rare entities. Diagonal of (source_layer, target_layer) matrix consistently performs best — there is layer-to-layer correspondence within model families. Patching into early layers of the larger model is most effective.

### 4.5 Multi-Hop Reasoning Correction (§5)
**Question:** Can Patchscopes fix multi-hop reasoning failures?

**Setup:** Query = [π2][π1], e.g., "The current CEO of" + "the company that created Visual Basic Script"
- τ1 = (Visual Basic, product of, Microsoft)
- τ2 = (Microsoft, CEO, Satya Nadella)

**CoT Patchscope:**
- Source: run on π1 alone; extract last-token hidden state (encodes "Microsoft")
- Target: full query [π2][π1]; patch at position just before π1
- Attention mask: π1 tokens cannot see π2 tokens and vice versa

**Results:**
- Vanilla baseline: 19.57%
- Chain-of-Thought baseline: 35.71%
- CoT Patchscope: **50% accuracy**

---

## 5. Models Used in Paper

| Model | Size | Layers | Hidden Dim | Architecture |
|---|---|---|---|---|
| LLaMA2 | 13B | 40 | 5120 | LLaMA2 (Meta) |
| Vicuna | 13B | 40 | 5120 | LLaMA1 fine-tuned on ShareGPT |
| GPT-J | 6B | 28 | 4096 | EleutherAI |
| Pythia | 12B | 36 | 5120 | EleutherAI |

**Layer path in PyTorch model tree:**
- Vicuna/LLaMA: `model.model.layers[i]`
- GPT-J: `model.transformer.h[i]`
- Pythia: `model.gpt_neox.layers[i]`

---

## 6. Known Limitations

1. **Placeholder contamination:** The placeholder token "x" in T retains its own representation in early target layers, potentially interfering with generation. Affects multi-token generation scenarios.

2. **Late layer degradation:** Representations in late layers shift toward next-token prediction, making attribute information less accessible to Patchscopes.

3. **Cross-family patching:** Cross-model patching works well within model families (7B→13B Vicuna) but requires learned mappings and hasn't been validated across different architectures.

4. **Early layer noise (layers 1-10):** All methods perform poorly in very early layers — this is where input contextualization happens, not prediction formation.

5. **Numerically unstable outputs:** In early layers, interpretation reflects current token; in later layers, next token. Top-1 prediction is not always the most faithful interpretation (often within top-10).

---

## 7. Extending Patchscopes to Vision-Language Models (VLMs)

> This section describes the research goal: adapting Patchscopes to VLMs.

### 7.1 Target Architecture: LLaVA-style VLMs
```
Image → ViT Encoder → MLP Projector → Visual Tokens
                                            ↓
                                    LLM Backbone (Vicuna/LLaMA)
                                            ↑
                                       Text Tokens
                                            ↓
                                         Output
```

### 7.2 Extended Formal Framework
**Source:** **(S_text, S_image, i, M_VLM, ℓ, modality)**
- S_image: the input image
- modality ∈ {visual, text, fused}: what kind of token is at position i

**Target:** **(T, i*, f, M*, ℓ*)** — target is always text-only for verbalization

**Three types of source representations to inspect:**
1. **Pre-projector ViT hidden states** — from vision encoder before MLP projection (dim=1024 for CLIP-L)
2. **Post-projector visual token hidden states** — visual tokens inside LLM backbone (dim=4096)
3. **Cross-modal fused text token hidden states** — text tokens that have attended to visual tokens

### 7.3 Visual Token Positions in LLaVA
In LLaVA-1.5:
- Position 0: BOS token
- Positions 1 to 576: visual patch tokens (24×24 patches from CLIP-L/14@336)
- Positions 577+: text tokens

```python
NUM_VISUAL_TOKENS = 576  # for LLaVA-1.5 with 336px input
VISUAL_TOKEN_START = 1   # position 0 is BOS
```

### 7.4 Key Implementation Changes Needed

**Change 1: `get_hidden_state()` must handle image inputs**
```python
def get_hidden_state_vlm(model, processor, image, text_prompt,
                          position, layer, modality="visual"):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if modality == "visual":
        actual_position = VISUAL_TOKEN_START + position  # patch index
    else:
        actual_position = position
    return outputs.hidden_states[layer][0, actual_position, :]
```

**Change 2: Mapping function f for cross-modal patching**
- Post-projector (dim matches): f = identity works
- Pre-projector (dim mismatch 1024→4096): f = learned MLP, initialize from VLM's own projector
- Cross-model: f = learned affine mapping

**Change 3: Model layer path for LLaVA**
```python
# LLaVA-1.5 layer navigation
model.language_model.model.layers[i]  # LLM backbone layers
model.vision_tower.vision_model.encoder.layers[i]  # ViT layers
model.multi_modal_projector  # the MLP projector
```

**Change 4: Hook mechanism stays identical** — PyTorch hooks are architecture-agnostic

### 7.5 New Target Prompts for VLM Experiments

**Visual concept crystallization (analogue of §4.3):**
```
T = "Syria: Country in the Middle East, Leonardo DiCaprio: American actor, x:"
```
Patch visual patch token hidden state at position x. Track how generated description changes across layers.

**Visual attribute extraction (analogue of §4.2):**
```
T_color    = "The color of the object in this region is:"
T_category = "The object shown here is a type of:"
T_relation = "The spatial relationship shown is:"
```

**Visual entity resolution:**
```
T = "subject1: description1, subject2: description2, x:"
```

### 7.6 Baseline Methods to Compare Against
1. **Linear probes on visual features** — direct analogue of probing in §4.2
2. **Logit Lens on visual tokens** — project visual token hidden states via unembedding matrix
3. **Attention visualization / GradCAM** — tells you *where* model looks, not *what* it understands
4. **CLIP zero-shot classification** — baseline for visual concept recognition
5. **Causal tracing in VLMs** — binary signal only, no natural language output

### 7.7 Expected Winning Conditions for VLM Patchscopes
- **Vs. linear probes:** Early and mid LLM backbone layers (same reason as text paper)
- **Vs. Logit Lens:** Visual tokens were never trained for direct unembedding — projecting them to vocabulary space produces noise; Patchscopes avoids this
- **Vs. attention visualization:** Patchscopes gives semantic content ("a red apple on a table"), not just a heatmap
- **Novel capability:** Layer-by-layer visual concept crystallization in natural language — no prior method can do this

---

## 8. Datasets Referenced in Paper

| Dataset | Used For | Description |
|---|---|---|
| The Pile (eval split) | §4.1 | 2000 samples for next-token prediction evaluation |
| Hernandez et al. 2023b dataset | §4.2, §5 | Knowledge triplets (σ,ρ,ω) for factual/commonsense reasoning |
| WikiText-103 | §4.2 | Source for Wikipedia sentences containing subjects |
| PopQA | §4.3 | 200 most popular + 200 least popular entities |
| Multi-hop constructed | §5 | 1,104 multi-hop queries, 46 filtered for evaluation |

---

## 9. Evaluation Metrics

| Metric | Used In | Definition |
|---|---|---|
| Precision@1 | §4.1 | Does argmax(p̃ℓ) == argmax(pL)? |
| Surprisal | §4.1 | -log p^L_t̃ where t̃ = argmax(p̃ℓ) |
| Accuracy | §4.2, §5 | Does true object ω appear in generated text (up to 20 tokens)? |
| RougeL | §4.3 | Overlap between generated description and Wikipedia description |
| Rouge1 | §4.3 | Unigram overlap variant |
| SBERT | §4.3 | Sentence-BERT semantic similarity score |

---

## 10. Citation

```bibtex
@inproceedings{ghandeharioun2024patchscopes,
  title={Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models},
  author={Ghandeharioun, Asma and Caciularu, Avi and Pearce, Adam and Dixon, Lucas and Geva, Mor},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://arxiv.org/abs/2401.06102}
}
```
