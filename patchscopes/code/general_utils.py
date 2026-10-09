# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import re

import torch
import transformers


class QwenVLImageTextProcessor:
  """Image-only processor for Qwen2.5-VL (no video processor, so no torchvision needed).

  Mirrors what Qwen2_5_VLProcessor does for images: the image processor returns pixel patches
  and image_grid_thw, and each <|image_pad|> in the text is expanded to one token per merged
  (merge_size x merge_size) patch cell.
  """

  # Qwen2.5-VL chat format with its default system prompt, one image then the question.
  PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>\n"
            "<|im_start|>assistant\n")

  def __init__(self, image_processor, tokenizer):
    self.image_processor = image_processor
    self.tokenizer = tokenizer
    self.image_token = "<|image_pad|>"
    self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

  def set_pixel_bounds(self, min_pixels=None, max_pixels=None):
    ip = self.image_processor
    if min_pixels is not None:
      ip.min_pixels = min_pixels
      ip.size["shortest_edge"] = min_pixels
    if max_pixels is not None:
      ip.max_pixels = max_pixels
      ip.size["longest_edge"] = max_pixels

  def __call__(self, images=None, text=None, return_tensors="pt"):
    out = {}
    texts = [text] if isinstance(text, str) else list(text)
    if images is not None:
      images = images if isinstance(images, (list, tuple)) else [images]
      vis = self.image_processor(images=images, return_tensors=return_tensors)
      out["pixel_values"], out["image_grid_thw"] = vis["pixel_values"], vis["image_grid_thw"]
      merge = self.image_processor.merge_size ** 2
      counts = [int(g.prod()) // merge for g in vis["image_grid_thw"]]
      expanded, k = [], 0
      for t in texts:
        while self.image_token in t:
          t = t.replace(self.image_token, "<|placeholder|>" * counts[k], 1)
          k += 1
        expanded.append(t.replace("<|placeholder|>", self.image_token))
      texts = expanded
    enc = self.tokenizer(texts, return_tensors=return_tensors, padding=True)
    out.update(enc)
    return out


class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      use_fast=True,
      device="cuda",
      ):
    is_vlm = (
        model_name is not None and "llava" in model_name.lower()
    ) or (
        model is not None and "llava" in type(model).__name__.lower()
    )
    is_qwen_vl = (
        model_name is not None and "qwen" in model_name.lower() and "vl" in model_name.lower()
    ) or (
        model is not None and "qwen2_5_vl" in type(model).__name__.lower()
    )

    if is_qwen_vl:
      from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
      from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
          Qwen2VLImageProcessor)
      if tokenizer is None:
        assert model_name is not None
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = QwenVLImageTextProcessor(
            Qwen2VLImageProcessor.from_pretrained(model_name), tokenizer)
      else:
        processor = None
      if model is None:
        assert model_name is not None
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype)
        if device is not None:
          model.to(device)
        set_requires_grad(False, model)
        model.eval()
      self.tokenizer = tokenizer
      self.model = model
      self.device = device
      self.processor = processor
      self.is_vlm = True
      self.family = "qwen2.5-vl"
      self.num_visual_tokens = None  # depends on the image size
      self.vision_tower = model.model.visual
      self.layer_names = [
          n for n, _ in model.named_modules()
          if re.match(r"^model\.language_model\.layers\.\d+$", n)
      ]
    elif is_vlm:
      from transformers import LlavaForConditionalGeneration, LlavaProcessor
      if tokenizer is None:
        assert model_name is not None
        processor = LlavaProcessor.from_pretrained(model_name)
        tokenizer = processor.tokenizer
      else:
        processor = None
      if model is None:
        assert model_name is not None
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype
        )
        if device is not None:
          model.to(device)
        set_requires_grad(False, model)
        model.eval()
      self.tokenizer = tokenizer
      self.model = model
      self.device = device
      self.processor = processor
      self.is_vlm = True
      self.family = "llava"
      self.num_visual_tokens = 576
      self.visual_token_start = 1
      # transformers >= 4.5x nests vision_tower / multi_modal_projector /
      # language_model under model.model (LlavaModel base), with
      # LlavaForConditionalGeneration only adding lm_head on top. The
      # nested language_model is the bare decoder (no lm_head of its own).
      self.vision_tower = model.model.vision_tower
      self.projector = model.model.multi_modal_projector
      self.num_vision_layers = len([
          n for n, _ in model.model.vision_tower.named_modules()
          if re.match(r".*encoder\.layers\.\d+$", n)
      ])
      self.layer_names = [
          n for n, _ in model.named_modules()
          if re.match(r"^model\.language_model\.layers\.\d+$", n)
      ]
    else:
      if tokenizer is None:
        assert model_name is not None
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
      if model is None:
        assert model_name is not None
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype
        )
        if device is not None:
          model.to(device)
        set_requires_grad(False, model)
        model.eval()
      self.tokenizer = tokenizer
      self.model = model
      self.device = device
      self.processor = None
      self.is_vlm = False
      self.family = "lm"
      self.layer_names = [
          n
          for n, _ in model.named_modules()
          if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
      ]

    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)


def get_visual_token_position(mt, patch_index):
  """Return the absolute token position of a visual patch in the LLM sequence."""
  return mt.visual_token_start + patch_index
