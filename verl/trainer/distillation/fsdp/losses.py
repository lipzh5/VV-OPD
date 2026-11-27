# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import os
import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.

    Returns:
        [B, T, K] tensor with the same dtype as `logits`.
    """
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=logits.dtype, device=logits.device)

    out = torch.empty((N, K), dtype=logits.dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(logits.dtype)
    return out.reshape(B, T, K)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _compute_forward_kl_topk_single(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    if teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested:
        teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
        teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)
    else:
        # Internal dual-teacher path passes flat dense [1,total_nnz,K].
        assert teacher_topk_log_probs.ndim == teacher_topk_ids.ndim == 3

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits)).
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        )
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    cvga_step_ids: torch.Tensor | None = None,
    cvga_visual_signal: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Preserve legacy top-k path or calculate uniform two-teacher loss.

    Dual input is [S,2K] per sample, packed as [original K | visual K].
    We deliberately do NOT normalize the teachers' truncated probabilities:
    the original baseline uses raw prompt_logprobs and clamps token losses.
    """
    # CVGA_REAL_TWO_TEACHER_TOKEN_MIN_NORM_V1
    # Bypass legacy single-8B/two-prompt CVGA paths. Use *two model weights*.
    import os
    if os.environ.get('CVGA_TWO_REAL_TEACHERS_MIN_NORM', '0') == '1':
        if not getattr(config, 'uniform_dual_teacher', False):
            raise RuntimeError('Real two-teacher MGDA requires uniform_dual_teacher=True')
        if get_ulysses_sequence_parallel_world_size() != 1:
            raise RuntimeError('Two-teacher logit min-norm supports only sequence_parallel_size=1')
        if not teacher_topk_ids.is_nested or not teacher_topk_log_probs.is_nested:
            raise RuntimeError('Packed two-teacher inputs must be nested')
        ids = teacher_topk_ids.values()
        logps = teacher_topk_log_probs.values()
        k = config.distillation_loss.topk
        if not isinstance(k, int) or k < 1 or ids.shape != logps.shape or ids.ndim != 2 or ids.shape[-1] != 2*k:
            raise RuntimeError(f'Expected [N,2*K] packed Top-K from two distinct models; got {ids.shape}')
        from importlib import import_module
        choose_alpha = import_module('cvpr_exp3.3_two_teacher_token_min_norm').choose_alpha
        # Genuine independent teacher top-k tensors; logit-space gradients are
        # analytical proxies, not 2B parameter-space gradients.
        alpha, diagnostics = choose_alpha(
            student_logits,
            ids[:, :k].unsqueeze(0), logps[:, :k].unsqueeze(0),
            ids[:, k:].unsqueeze(0), logps[:, k:].unsqueeze(0),
            conflict_threshold=float(os.environ.get('CVGA_MIN_NORM_CONFLICT_THRESHOLD', '0')),
            base_alpha_original=float(os.environ.get('CVGA_MIN_NORM_BASE_ALPHA', '0.5')),
            chunk_tokens=int(os.environ.get('CVGA_MIN_NORM_CHUNK_TOKENS', '128')),
            log_prob_min_clamp=config.distillation_loss.log_prob_min_clamp,
            loss_max_clamp=config.distillation_loss.loss_max_clamp,
        )
        orig = _compute_forward_kl_topk_single(
            student_logits, logps[:, :k].unsqueeze(0), ids[:, :k].unsqueeze(0), config, data_format
        )
        priv = _compute_forward_kl_topk_single(
            student_logits, logps[:, k:].unsqueeze(0), ids[:, k:].unsqueeze(0), config, data_format
        )
        result = {}
        for key in orig:
            a, b = orig[key], priv[key]
            if key == 'distillation_losses':
                limit = config.distillation_loss.loss_max_clamp
                a = a.clamp(min=0.0, max=limit) if limit is not None else a.clamp_min(0.0)
                b = b.clamp(min=0.0, max=limit) if limit is not None else b.clamp_min(0.0)
                w = alpha.to(device=a.device, dtype=a.dtype)
                if w.shape != a.shape:
                    raise RuntimeError(f'Token alpha {w.shape} != OPD loss {a.shape}; no broadcasting allowed')
                result[key] = w * a + (1.0-w) * b
            else:
                result[key] = 0.5 * a + 0.5 * b
        # Sparse diagnostics only (not a second backward pass).
        global _cvga_real_two_teacher_step
        _cvga_real_two_teacher_step = globals().get('_cvga_real_two_teacher_step', 0) + 1
        if _cvga_real_two_teacher_step <= 3 or _cvga_real_two_teacher_step % 25 == 0:
            print('CVGA_REAL_TWO_TEACHER_MIN_NORM ' +
                  ' '.join(f'{key}={value:.6g}' for key, value in diagnostics.items()), flush=True)
        return result
    # CVGA_CF_GATE_V4_LOSS: opt-in (CVGA_CF_GATE_ENABLED=1 and CVGA_CF_GATE_VERSION=4); V3 below is untouched.
    if os.environ.get('CVGA_CF_GATE_ENABLED', '0') == '1' and os.environ.get('CVGA_CF_GATE_VERSION', '') == '4':
        if not getattr(config, 'uniform_dual_teacher', False):
            raise RuntimeError('CF gate V4 requires uniform_dual_teacher=True')
        if os.environ.get('CVGA_TWO_REAL_TEACHERS_MIN_NORM', '0') != '0':
            raise RuntimeError('CF gate V4 must not run with min-norm enabled')
        if get_ulysses_sequence_parallel_world_size() != 1:
            raise RuntimeError('CF gate V4 requires SP=1')
        if not teacher_topk_ids.is_nested or not teacher_topk_log_probs.is_nested:
            raise RuntimeError('CF gate V4 expected nested teacher packing')
        from importlib import import_module as _cf4_import
        _cf4 = _cf4_import('cvpr_exp3.3_cf_gate_runtime_v4')
        _k = config.distillation_loss.topk
        _ids, _lp = teacher_topk_ids.values(), teacher_topk_log_probs.values()
        if not isinstance(_k, int) or _ids.ndim != 2 or _lp.shape != _ids.shape or _ids.shape[-1] != 2 * _k + 2:
            raise RuntimeError(f'CF gate V4 expected [N,2*K+2] teacher scores, got {tuple(_ids.shape)}')
        return _cf4.mixed_loss(student_logits, _ids, _lp, _k, config, data_format, _compute_forward_kl_topk_single)
    # CVGA_CF_GATE_LOSS_V1 / CVGA_REAL_CF_ACTION_LOGP_GATE_V1
    # New exclusive loss branch BEFORE uniform (0.5/0.5) path. Two independent
    # actual teachers still provide top-K; no-image Original is gate-only.
    if os.environ.get('CVGA_CF_GATE_ENABLED','0') == '1':
        if not getattr(config,'uniform_dual_teacher',False):
            raise RuntimeError('CF action gate requires uniform_dual_teacher=True')
        if os.environ.get('CVGA_TWO_REAL_TEACHERS_MIN_NORM','0') != '0':
            raise RuntimeError('CF gate must not run with min-norm enabled')
        if os.environ.get('CVGA_AP_ENABLED','0') == '1' or os.environ.get('CVGA_CF_MODE',''):
            raise RuntimeError('CF gate cannot combine with CVGA-AP or legacy CF')
        if get_ulysses_sequence_parallel_world_size()!=1:
            raise RuntimeError('CF gate currently requires SP=1')
        if not teacher_topk_ids.is_nested or not teacher_topk_log_probs.is_nested:
            raise RuntimeError('CF gate expected nested teacher packing')
        from importlib import import_module
        cf=import_module('cvpr_exp3.3_cf_gate_runtime_v1')
        flat_ids=teacher_topk_ids.values()
        flat_lp=teacher_topk_log_probs.values()
        k=config.distillation_loss.topk
        if not isinstance(k,int) or k<1 or flat_ids.ndim!=2 or flat_lp.shape!=flat_ids.shape or flat_ids.shape[-1]!=2*k+2:
            raise RuntimeError(f'CF gate expected actual [N,2*K+2] teacher scores, got {flat_ids.shape}')
        tau=float(os.environ.get('CVGA_CF_GATE_TAU','0.7'))
        alpha,stats=cf.cf_token_weights(flat_ids[:,2*k:],flat_lp[:,2*k:],tau)
        orig=_compute_forward_kl_topk_single(student_logits,flat_lp[:,:k].unsqueeze(0),flat_ids[:,:k].unsqueeze(0),config,data_format)
        grpo=_compute_forward_kl_topk_single(student_logits,flat_lp[:,k:2*k].unsqueeze(0),flat_ids[:,k:2*k].unsqueeze(0),config,data_format)
        out={}
        for key in orig:
            a,b=orig[key],grpo[key]
            if key=='distillation_losses':
                limit=config.distillation_loss.loss_max_clamp
                a=a.clamp(min=0.0,max=limit) if limit is not None else a.clamp_min(0.0)
                b=b.clamp(min=0.0,max=limit) if limit is not None else b.clamp_min(0.0)
                w=alpha.to(device=a.device,dtype=a.dtype).unsqueeze(0)
                if w.shape!=a.shape or b.shape!=a.shape:
                    raise RuntimeError(f'CF gate loss shape mismatch: {w.shape} / {a.shape} / {b.shape}')
                out[key]=w*a+(1.0-w)*b
            else:
                out[key]=0.5*a+0.5*b
        global _cvga_cf_real_loss_calls
        _cvga_cf_real_loss_calls=globals().get('_cvga_cf_real_loss_calls',0)+1
        if _cvga_cf_real_loss_calls<=3 or _cvga_cf_real_loss_calls%25==0:
            print('[CF_GATE_LOSS] '+' '.join(f'{n}={v:.6g}' for n,v in stats.items())+
                  f' tau={tau:.6g} call={_cvga_cf_real_loss_calls}',flush=True)
        return out
    cvga_ap = os.environ.get("CVGA_AP_ENABLED", "0") == "1"
    if not (config.uniform_dual_teacher or cvga_ap):
        return _compute_forward_kl_topk_single(
            student_logits, teacher_topk_log_probs,
            teacher_topk_ids, config, data_format,
        )
    if not teacher_topk_log_probs.is_nested or not teacher_topk_ids.is_nested:
        raise ValueError("Uniform dual-teacher mode expects nested top-k tensors")
    k = config.distillation_loss.topk
    if not isinstance(k, int) or k < 1:
        raise ValueError("Uniform dual-teacher requires a positive top-k")
    flat_ids = teacher_topk_ids.values()
    flat_logps = teacher_topk_log_probs.values()
    if flat_ids.shape != flat_logps.shape or flat_ids.ndim != 2 or flat_ids.shape[-1] != (4*k if os.environ.get("CVGA_CF_MODE", "") == "no_image" else 2*k):
        raise ValueError(f"Expected packed [N,{2*k}] teacher tensors, got {flat_ids.shape} and {flat_logps.shape}")

    original = _compute_forward_kl_topk_single(
        student_logits, flat_logps[:, :k].unsqueeze(0),
        flat_ids[:, :k].unsqueeze(0), config, data_format,
    )
    visual = _compute_forward_kl_topk_single(
        student_logits, flat_logps[:, k:2*k].unsqueeze(0),
        flat_ids[:, k:2*k].unsqueeze(0), config, data_format,
    )

    # Existing upstream compute_forward_kl_topk() applies clamp_min(0)
    # *after* computing its per-token top-k divergence. Apply it per teacher,
    # then average. The upstream outer clamp becomes idempotent.
    # Also reproduce configured loss_max_clamp per teacher, if present.
    max_clamp = config.distillation_loss.loss_max_clamp
    # CVGA_AP_EXPLICIT_ABLATION_V1: exact, auditable teacher-only ablations.
    # The teacher server still scores original and annotated prompts where
    # annotations exist; only the FSDP training objective is selected here.
    ablation_mode = os.environ.get("CVGA_AP_ABLATION_MODE", "").strip().lower()
    if ablation_mode:
        if ablation_mode not in ("cc_only", "uniform"):
            raise ValueError(f"Unknown CVGA_AP_ABLATION_MODE={ablation_mode!r}")
        if not cvga_ap:
            raise RuntimeError("Explicit ablations require CVGA_AP_ENABLED=1 for annotation scoring")
        if getattr(config, "uniform_dual_teacher", False):
            raise RuntimeError("Set distillation.uniform_dual_teacher=False: one shared 8B, two prompts")
        # No gradient cosine, min-norm, step aggregation, or visual preference.
        result = {}
        for key in original:
            original_value, visual_value = original[key], visual[key]
            if key == "distillation_losses":
                if max_clamp is not None:
                    original_value = original_value.clamp(min=0.0, max=max_clamp)
                    visual_value = visual_value.clamp(min=0.0, max=max_clamp)
                else:
                    original_value = original_value.clamp_min(0.0)
                    visual_value = visual_value.clamp_min(0.0)
            if ablation_mode == "cc_only":
                # IMPORTANT: no multiplication by 0 on original loss tensor.
                result[key] = visual_value
            else:
                result[key] = 0.5 * original_value + 0.5 * visual_value

        global _cvga_explicit_ablation_calls
        _cvga_explicit_ablation_calls = globals().get("_cvga_explicit_ablation_calls", 0) + 1
        if _cvga_explicit_ablation_calls <= 3 or _cvga_explicit_ablation_calls % 25 == 0:
            # Diagnostic is an actual loss difference, not merely an env flag.
            with torch.no_grad():
                a_loss = original["distillation_losses"].detach()
                b_loss = visual["distillation_losses"].detach()
                mean_delta = (a_loss - b_loss).abs().float().mean().item()
            print(
                f"CVGA_ABLATION mode={ablation_mode} "
                f"alpha_original={0.0 if ablation_mode == 'cc_only' else 0.5:.5f} "
                f"mean_abs_raw_teacher_loss_difference={mean_delta:.6g}",
                flush=True,
            )
        return result

    # Stage-1 CVGA arbitration weights, calculated WITHOUT autograd.
    alpha_original = None
    if cvga_ap or getattr(config, "cvga_enabled", False):
        from verl.trainer.distillation.fsdp.cvga_logit_arbitration import choose_alpha
        if get_ulysses_sequence_parallel_world_size() != 1:
            raise RuntimeError("Stage-1 CVGA supports SP=1 only")
        alpha_original, cvga_diag = choose_alpha(
            student_logits=student_logits,
            original_ids=flat_ids[:, :k].unsqueeze(0),
            original_log_probs=flat_logps[:, :k].unsqueeze(0),
            visual_ids=flat_ids[:, k:2*k].unsqueeze(0),
            visual_log_probs=flat_logps[:, k:2*k].unsqueeze(0),
            conflict_threshold=float(os.environ.get("CVGA_AP_CONFLICT_THRESHOLD", "0.0")) if cvga_ap else config.cvga_conflict_threshold,
            base_alpha_original=float(os.environ.get("CVGA_AP_ALPHA_ORIGINAL", "0.5")) if cvga_ap else config.cvga_base_alpha_original,
            chunk_tokens=int(os.environ.get("CVGA_AP_CHUNK_TOKENS", "128")) if cvga_ap else config.cvga_chunk_tokens,
            log_prob_min_clamp=config.distillation_loss.log_prob_min_clamp,
            loss_max_clamp=config.distillation_loss.loss_max_clamp,
            step_ids=cvga_step_ids if cvga_ap else None,
            visual_signal=cvga_visual_signal if cvga_ap else None,
            signal_original_weight=float(os.environ.get("CVGA_AP_SIGNAL_ORIGINAL_WEIGHT", "0.25")),
            signal_strength=float(os.environ.get("CVGA_AP_SIGNAL_STRENGTH", "0.5")),
        
            # CVGA_CF_NO_IMAGE_EQ21_V1 additional counterfactual scores.
            cf_original_ids=flat_ids[:,2*k:3*k].unsqueeze(0),
            cf_original_log_probs=flat_logps[:,2*k:3*k].unsqueeze(0),
            cf_visual_ids=flat_ids[:,3*k:4*k].unsqueeze(0),
            cf_visual_log_probs=flat_logps[:,3*k:4*k].unsqueeze(0),
)
        # One print per actor forward. Useful for verifying the feature is live.
        global _cvga_stage1_loss_calls
        _cvga_stage1_loss_calls = globals().get("_cvga_stage1_loss_calls", 0) + 1
        # CVGA_GEO3K_GROUP_LOG_EVERY_V1
        if (__import__('os').environ.get('CVGA_DIAG_LOG_EVERY') == '1'
                or _cvga_stage1_loss_calls <= 3 or _cvga_stage1_loss_calls % 25 == 0):
            print("CVGA_STAGE1 " + " ".join(f"{key}={v:.5f}" for key,v in cvga_diag.items()), flush=True)

    out = {}
    for name in original:
        a = original[name]
        b = visual[name]
        if name == "distillation_losses":
            a = a.clamp(min=0.0, max=max_clamp) if max_clamp is not None else a.clamp_min(0.0)
            b = b.clamp(min=0.0, max=max_clamp) if max_clamp is not None else b.clamp_min(0.0)
        if name == "distillation_losses" and alpha_original is not None:
            w = alpha_original.to(device=a.device, dtype=a.dtype)
            if w.shape != a.shape:
                raise RuntimeError(f"CVGA weight shape {w.shape} != loss shape {a.shape}")
            out[name] = w * a + (1.0 - w) * b
        else:
            out[name] = 0.5 * a + 0.5 * b
    return out
