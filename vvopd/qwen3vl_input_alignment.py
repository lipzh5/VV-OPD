#!/usr/bin/env python3
"""Align Qwen3-VL multimodal inputs with appended reasoning tokens.

Extend token-level modality metadata when candidate reasoning steps
are appended to the multimodal prefix, ensuring consistent sequence
lengths and correct multimodal positional encoding.

Preserve the original visual inputs and prefix metadata.
"""
from __future__ import annotations
import torch

ALIGNMENT_VERSION = 'qwen3vl_mm_token_type_ids_text_suffix_v1'


def align_appended_text_step(model_inputs: dict, prefix_len: int, suffix_len: int) -> dict:
    if suffix_len <= 0 or prefix_len <= 0:
        raise ValueError('V4 modality alignment requires nonempty prefix and step')
    ids = model_inputs.get('input_ids')
    attention = model_inputs.get('attention_mask')
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('V4 modality alignment requires input_ids shape [1,seq]')
    length = prefix_len + suffix_len
    if ids.shape[1] != length:
        raise ValueError(f'V4 input_ids length mismatch: got={ids.shape[1]} expected={length}')
    if not isinstance(attention, torch.Tensor) or attention.shape != ids.shape:
        raise ValueError(f'V4 attention_mask length mismatch: expected={tuple(ids.shape)}')
    if 'mm_token_type_ids' not in model_inputs:
        if 'image_grid_thw' in model_inputs or 'video_grid_thw' in model_inputs:
            raise ValueError('V4 Qwen3-VL multimodal input has no mm_token_type_ids')
    for key in ('mm_token_type_ids', 'token_type_ids'):
        if key not in model_inputs:
            continue
        mod = model_inputs[key]
        if not isinstance(mod, torch.Tensor) or mod.ndim != 2 or mod.shape != (1, prefix_len):
            raise ValueError(f'V4 {key} should describe prefix [1,{prefix_len}], '
                             f'got={getattr(mod,"shape",None)}')
        # 0 = text; preserve all multimodal prefix values unchanged.
        suffix = torch.zeros((1,suffix_len), dtype=mod.dtype, device=mod.device)
        model_inputs[key] = torch.cat((mod, suffix), dim=-1)
        if model_inputs[key].shape != ids.shape:
            raise AssertionError(f'V4 {key} extension failed')
    return model_inputs
