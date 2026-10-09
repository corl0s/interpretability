"""Model-specific capture of visual-token hidden states.

capture_states(mt, processor, image) returns [n_layers, n_visual_tokens, hidden] (fp16, CPU), where
index l is the OUTPUT of decoder block l (pre final-norm), and visual tokens are in the order of
the model's geometry (vtr.geometry), so token i matches coverage[i].

  llava-1.5   576 tokens (24x24), via visual_object_identification.capture_visual_states
  qwen2.5-vl  rows x cols tokens (28-px cells of the resized image, row-major); the grid reported
              by the processor is checked against the geometry for every image
"""

import torch

from visual_object_identification import capture_visual_states, get_decoder

QWEN_SOURCE_TEXT = "Describe the image."


def text_position_ids(mt, start, length, batch, device):
  """Explicit position ids for text-only forward passes, or None to let the model decide.

  Qwen2.5-VL uses 3-component multimodal RoPE; for text all three components equal the 1-D
  position. Passing them explicitly avoids depending on rope state cached by earlier calls.
  LLaVA keeps its default behaviour (None).
  """
  if getattr(mt, "family", "llava") != "qwen2.5-vl":
    return None
  pos = torch.arange(start, start + length, device=device)
  return pos.view(1, 1, -1).expand(3, batch, -1)


def capture_qwen(mt, processor, image, geometry, verify=False):
  text = processor.PROMPT.format(text=QWEN_SOURCE_TEXT)
  inputs = processor(images=[image], text=[text], return_tensors="pt")
  grid_t, grid_h, grid_w = [int(x) for x in inputs["image_grid_thw"][0]]
  rows, cols = geometry.grid(image.height, image.width)
  if (grid_h // 2, grid_w // 2) != (rows, cols) or grid_t != 1:
    raise ValueError(f"processor grid {(grid_t, grid_h, grid_w)} != geometry {(rows, cols)}")
  inputs = {k: v.to(mt.device) for k, v in inputs.items()}
  ids = inputs["input_ids"][0]
  positions = (ids == processor.image_token_id).nonzero(as_tuple=True)[0]
  if len(positions) != rows * cols or (positions[-1] - positions[0] + 1) != len(positions):
    raise ValueError(f"expected {rows * cols} contiguous image tokens, found {len(positions)}")
  start = int(positions[0])

  layers = get_decoder(mt).layers
  captured = {}
  handles = [l.register_forward_hook(
      lambda m, i, o, k=k: captured.__setitem__(k, (o[0] if isinstance(o, tuple) else o)[0]
                                                .detach()))
      for k, l in enumerate(layers)]
  try:
    with torch.no_grad():
      out = mt.model(**inputs, output_hidden_states=verify, use_cache=False)
  finally:
    for h in handles:
      h.remove()
  states = torch.stack([captured[k][start:start + rows * cols] for k in range(len(layers))])
  if verify:
    for k in (0, len(layers) // 2, len(layers) - 2):
      hf = out.hidden_states[k + 1][0, start:start + rows * cols]
      ok = torch.allclose(hf.float(), states[k].float(), atol=1e-3)
      print(f"  [verify] layer {k}: captured == hidden_states[{k + 1}]: {ok}")
    print(f"  [verify] {rows}x{cols} = {rows * cols} image tokens from position {start}")
  keep = torch.bfloat16 if states.dtype == torch.bfloat16 else torch.float16
  return states.to(keep).cpu()


def capture_states(mt, processor, image, geometry, verify=False):
  if getattr(mt, "family", "llava") == "qwen2.5-vl":
    return capture_qwen(mt, processor, image, geometry, verify)
  return capture_visual_states(mt, processor, image, verify)
