#!/usr/bin/env python3
"""Construct edit-token masks for verified contrastive reasoning pairs.

Identify token-level differences between positive and negative
reasoning steps, enabling localized visual consistency supervision
over the edited regions.
"""

from __future__ import annotations
from difflib import SequenceMatcher


def paired_masks(tokenizer, chosen_step: str, rejected_step: str, max_step: int):
    a = list(tokenizer.encode(chosen_step, add_special_tokens=False))
    b = list(tokenizer.encode(rejected_step, add_special_tokens=False))
    if not a or not b:
        raise ValueError('V3 MASK: empty chosen/rejected step')
    if len(a) > max_step or len(b) > max_step:
        raise ValueError(f'V3 MASK: step too long (chosen={len(a)}, rejected={len(b)}, max={max_step}); refuse truncation')
    if a == b:
        raise ValueError('V3 MASK: chosen/rejected token sequences identical; no visual contrast available')
    aa, bb = [False]*len(a), [False]*len(b)
    fallback_a, fallback_b = None, None
    for tag, i0, i1, j0, j1 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        for i in range(i0, i1):aa[i]=True
        for j in range(j0, j1):bb[j]=True
        # Positions closest to edit boundary, used only if a side has no edit.
        fallback_a = min(i0, len(a)-1)
        fallback_b = min(j0, len(b)-1)
    anchor_a = anchor_b = False
    if not any(aa):
        assert fallback_a is not None
        aa[fallback_a]=True;anchor_a=True
    if not any(bb):
        assert fallback_b is not None
        bb[fallback_b]=True;anchor_b=True
    return aa, bb, (anchor_a or anchor_b)


def candidate_mask(tokenizer, pair, step_text: str, max_step: int):
    """Same function signature as QA-fast hook's existing candidate_mask call."""
    chosen, rejected = str(pair['chosen_step']), str(pair['rejected_step'])
    aa, bb, _ = paired_masks(tokenizer, chosen, rejected, int(max_step))
    if step_text == chosen and step_text == rejected:
        raise ValueError('V3 MASK: identical chosen/rejected step text')
    if step_text == chosen: return aa
    if step_text == rejected:return bb
    raise ValueError(f'V3 MASK: candidate text does not exactly match pair_id={pair.get("pair_id", "?")}')
