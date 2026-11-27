"""Stage-1 CVGA ablation: conflict-aware min-norm mixing in STUDENT-LOGIT space.

This is NOT full counterfactual visual-consistency CVGA, and is NOT
parameter-gradient arbitration or step-level arbitration.

Inputs are student logits [B,T,V] and both teachers' aligned top-k scores
[B,T,K] for identical on-policy student prefixes.

The analytical gradient of truncated teacher forward-KL (when unclipped) is
    g = sum_i(q_i_active)*softmax(z_student) - scatter(q_i_active),
where q_i_active excludes log-probabilities clipped below log_prob_min_clamp.
We respect optional per-token [0,loss_max_clamp] clipping by zeroing a
teacher's diagnostic gradient at saturated tokens.

For conflict cos(g_original,g_visual) < threshold, choose the minimum-norm
convex combination; otherwise retain the baseline alpha_original.
The resulting weight is detached and used to combine the two *original*
FSDP loss tensors; thus teacher KL definitions and student autograd are intact.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F


def choose_alpha(
    student_logits: torch.Tensor,
    original_ids: torch.Tensor,
    original_log_probs: torch.Tensor,
    visual_ids: torch.Tensor,
    visual_log_probs: torch.Tensor,
    *,
    conflict_threshold: float = 0.0,
    base_alpha_original: float = 0.5,
    log_prob_min_clamp: float | None = -10.0,
    loss_max_clamp: float | None = 10.0,
    chunk_tokens: int = 256,
    step_ids: torch.Tensor | None = None,
    visual_signal: torch.Tensor | None = None,
    signal_original_weight: float = 0.25,
    signal_strength: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return detached alpha_original [B,T] and scalar diagnostics."""
    if student_logits.ndim != 3:
        raise ValueError("student_logits must be [B,T,V]")
    if not (0.0 <= base_alpha_original <= 1.0):
        raise ValueError("base_alpha_original must be in [0,1]")
    if not (-1.0 <= conflict_threshold < 1.0):
        raise ValueError("conflict_threshold must be in [-1,1)")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    shape = student_logits.shape[:2]
    if any(x.shape[:2] != shape for x in
           (original_ids, original_log_probs, visual_ids, visual_log_probs)):
        raise ValueError("All tensors must be aligned on [B,T]")
    if original_ids.shape != original_log_probs.shape or visual_ids.shape != visual_log_probs.shape:
        raise ValueError("ID/logprob shape mismatch")
    if original_ids.shape[-1] != visual_ids.shape[-1]:
        raise ValueError("Both teachers must have the same top-k")
    if not (student_logits.device == original_ids.device == original_log_probs.device
            == visual_ids.device == visual_log_probs.device):
        raise ValueError("All tensors must be on the same device")

    b, t, v = student_logits.shape
    k = original_ids.shape[-1]
    n = b * t
    flat_z = student_logits.detach().reshape(n, v)
    oi = original_ids.detach().reshape(n, k).long()
    vi = visual_ids.detach().reshape(n, k).long()
    ol = original_log_probs.detach().reshape(n, k)
    vl = visual_log_probs.detach().reshape(n, k)
    alpha_all = torch.empty(n, dtype=torch.float32, device=flat_z.device)
    # CVGA_AP_STEP_LEVEL_MODE_V1: inactive unless explicitly selected.
    cvga_mode = os.getenv("CVGA_AP_MODE", "token_level")
    if cvga_mode not in ("token_level", "step_level", "step_level_visual_prior"):
        raise ValueError(f"Unsupported CVGA_AP_MODE={cvga_mode!r}")
    if cvga_mode != "token_level":
        # Small O(B*T) diagnostic storage, not [B,T,V] and not model gradients.
        o2_all = torch.empty(n, dtype=torch.float32, device=flat_z.device)
        v2_all = torch.empty(n, dtype=torch.float32, device=flat_z.device)
        cross_all = torch.empty(n, dtype=torch.float32, device=flat_z.device)
        valid_all = torch.empty(n, dtype=torch.bool, device=flat_z.device)
        conflict_all = torch.empty(n, dtype=torch.bool, device=flat_z.device)
    num_conflict = 0
    num_valid = 0
    alpha_conflict_sum = 0.0
    eps = 1e-12

    with torch.no_grad():
        for start in range(0, n, chunk_tokens):
            end = min(n, start + chunk_tokens)
            # Detach diagnostics: only the preexisting two teacher losses
            # participate in autograd and backpropagation.
            lp = F.log_softmax(flat_z[start:end].float(), dim=-1)
            p = lp.exp()
            p_norm2 = (p * p).sum(dim=-1)
            ids_o = oi[start:end]
            ids_v = vi[start:end]
            logq_o = ol[start:end].float()
            logq_v = vl[start:end].float()
            logp_o = lp.gather(-1, ids_o)
            logp_v = lp.gather(-1, ids_v)
            if log_prob_min_clamp is not None:
                logq_o = logq_o.clamp_min(log_prob_min_clamp)
                logq_v = logq_v.clamp_min(log_prob_min_clamp)
                effective_o = logp_o > log_prob_min_clamp
                effective_v = logp_v > log_prob_min_clamp
                used_logp_o = logp_o.clamp_min(log_prob_min_clamp)
                used_logp_v = logp_v.clamp_min(log_prob_min_clamp)
            else:
                effective_o = torch.ones_like(logp_o, dtype=torch.bool)
                effective_v = torch.ones_like(logp_v, dtype=torch.bool)
                used_logp_o = logp_o
                used_logp_v = logp_v
            raw_qo = logq_o.exp()
            raw_qv = logq_v.exp()
            qo = raw_qo * effective_o
            qv = raw_qv * effective_v

            # Mirror the two-teacher baseline's independently clamped KL.
            loss_o = (raw_qo * (logq_o - used_logp_o)).sum(dim=-1)
            loss_v = (raw_qv * (logq_v - used_logp_v)).sum(dim=-1)
            active_o = loss_o > 0
            active_v = loss_v > 0
            if loss_max_clamp is not None:
                active_o = active_o & (loss_o < loss_max_clamp)
                active_v = active_v & (loss_v < loss_max_clamp)
            qo = qo * active_o.unsqueeze(-1)
            qv = qv * active_v.unsqueeze(-1)

            mo = qo.sum(dim=-1)
            mv = qv.sum(dim=-1)
            po = p.gather(-1, ids_o)
            pv = p.gather(-1, ids_v)
            o_norm2 = (mo * mo * p_norm2 - 2 * mo * (qo * po).sum(-1) + (qo * qo).sum(-1)).clamp_min(0)
            v_norm2 = (mv * mv * p_norm2 - 2 * mv * (qv * pv).sum(-1) + (qv * qv).sum(-1)).clamp_min(0)

            # Sparse q_original . q_visual via sort+searchsorted, O(KlogK)
            # instead of a [tokens,K,K] equality tensor.
            sorted_o, order_o = ids_o.sort(dim=-1)
            sorted_qo = qo.gather(-1, order_o)
            ix = torch.searchsorted(sorted_o.contiguous(), ids_v.contiguous()).clamp(max=k-1)
            matching = sorted_o.gather(-1, ix) == ids_v
            q_cross = (sorted_qo.gather(-1, ix) * qv * matching).sum(dim=-1)
            cross = mo * mv * p_norm2 - mo * (qv * pv).sum(-1) - mv * (qo * po).sum(-1) + q_cross
            valid = (o_norm2 > eps) & (v_norm2 > eps)
            cos = cross / (o_norm2 * v_norm2).clamp_min(eps).sqrt()
            conflict = valid & (cos < conflict_threshold)

            # Minimum-norm combination of the two gradient vectors.
            denominator = (o_norm2 + v_norm2 - 2 * cross).clamp_min(eps)
            alpha_mgda = ((v_norm2 - cross) / denominator).clamp(0, 1)
            alpha = torch.where(conflict, alpha_mgda, torch.full_like(alpha_mgda, base_alpha_original))
            alpha_all[start:end] = alpha
            if cvga_mode != "token_level":
                o2_all[start:end] = o_norm2.detach()
                v2_all[start:end] = v_norm2.detach()
                cross_all[start:end] = cross.detach()
                valid_all[start:end] = valid.detach()
                conflict_all[start:end] = conflict.detach()
            num_conflict += int(conflict.sum().item())
            num_valid += int(valid.sum().item())
            alpha_conflict_sum += float(alpha[conflict].sum().item())
            del lp, p

    if cvga_mode != "token_level":
        if step_ids is None:
            raise RuntimeError("Step-Level CVGA requires cvga_step_ids; no silent fallback")
        from cvpr_exp.cvga_step_level import arbitrate_steps
        return arbitrate_steps(
            alpha_token=alpha_all, step_ids=step_ids, visual_signal=visual_signal,
            o2=o2_all, v2=v2_all, cross=cross_all,
            valid=valid_all, token_conflict=conflict_all,
            b=b, t=t, base_alpha=base_alpha_original,
            token_conflict_rate=num_conflict / max(1, num_valid),
            token_conflict_count=num_conflict, valid_token_count=num_valid,
            alpha_token_conflicts_mean=alpha_conflict_sum / max(1, num_conflict),
            signal_strength=signal_strength,
            signal_original_weight=signal_original_weight,
            mode=cvga_mode,
        )

    # Aggregate token-level min-norm arbitration to step-level weights.
    # CVGA_PURE_TOKEN_MIN_NORM_V1
    # CVGA_CFP_MATCHED_TOKEN_STEP_V1
    pure_token_min_norm = os.environ.get("CVGA_AP_PURE_TOKEN_MIN_NORM", "0") == "1"
    if pure_token_min_norm and cvga_mode != "token_level":
        raise RuntimeError("Pure token min-norm requires CVGA_AP_MODE=token_level")
    if pure_token_min_norm and not globals().get("_cvga_pure_token_announced", False):
        print("CVGA_PURE_TOKEN_ACTIVE step_average=OFF", flush=True)
        globals()["_cvga_pure_token_announced"] = True
    if not (0 <= signal_strength <= 1 and 0 <= signal_original_weight <= 1):
        raise ValueError("Counterfactual visual preference strengths/targets must be in [0,1]")
    # alpha_all is already the min-norm output for each token; all computations
    # below are no-grad so only the original KL losses participate in backward.
    # IMPORTANT: apply identical visual preference BEFORE optional step-average.
    # This ensures the two ablations differ ONLY by post-arbitration step averaging.
    step_count = 0
    preferred_steps = 0
    if signal_strength > 0 and (step_ids is None or visual_signal is None):
        raise RuntimeError("CVGA CFP requires both aligned step_ids and visual_signal; "
                           "refusing to silently disable counterfactual preference")
    if step_ids is not None:
        segment_ids = step_ids.values() if step_ids.is_nested else step_ids.reshape(-1)
        if segment_ids.numel() != n:
            raise RuntimeError(f"CVGA step alignment {segment_ids.numel()} != logits {n}")
        if visual_signal is not None:
            signal = visual_signal.values() if visual_signal.is_nested else visual_signal.reshape(-1)
            if signal.numel() != n:
                raise RuntimeError(f"CVGA visual signal alignment {signal.numel()} != logits {n}")
        else:
            signal = torch.zeros_like(segment_ids)
        # The step ids are contiguous by construction, with negative ids for
        # prompt/non-step tokens. Any positive signal tags the entire reasoning
        # step, matching legacy CFP semantics.
        ids_cpu = segment_ids.detach().to(device="cpu").tolist()
        with torch.no_grad():
            left = 0
            while left < n:
                right = left + 1
                while right < n and ids_cpu[right] == ids_cpu[left]:
                    right += 1
                if ids_cpu[left] >= 0:
                    vals = alpha_all[left:right]
                    if signal_strength > 0 and bool((signal[left:right] > 0).any().item()):
                        vals.mul_(1.0 - signal_strength).add_(signal_strength * signal_original_weight)
                        preferred_steps += 1
                    if not pure_token_min_norm:
                        vals.fill_(vals.mean().item())
                    step_count += 1
                left = right
    if preferred_steps > 0 and not globals().get("_cvga_cfp_active_announced", False):
        print(f"CVGA_CFP_ACTIVE step_average={'OFF' if pure_token_min_norm else 'ON'} "
              f"signal_strength={signal_strength} preferred_steps={preferred_steps}", flush=True)
        globals()["_cvga_cfp_active_announced"] = True

    metrics = {
        "cvga_step_count": float(step_count),
        "cvga_visual_preferred_steps": float(preferred_steps),
        "conflict_rate_valid": num_conflict / max(1, num_valid),
        "conflict_count": float(num_conflict),
        "valid_token_count": float(num_valid),
        "alpha_original_mean": float(alpha_all.mean().item()) if n else base_alpha_original,
        "alpha_original_conflicts_mean": alpha_conflict_sum / max(1, num_conflict),
    }
    return alpha_all.reshape(b, t).detach(), metrics


# CVGA_CF_NO_IMAGE_EQ21_V1: Preserve legacy algorithm when no-image mode is OFF.
_choose_alpha_legacy_noimage_cfp = choose_alpha

def choose_alpha(*args, **kwargs):
    import os
    if os.environ.get("CVGA_CF_MODE", "") != "no_image":
        for key in ("cf_original_ids","cf_original_log_probs","cf_visual_ids","cf_visual_log_probs"):
            kwargs.pop(key, None)
        return _choose_alpha_legacy_noimage_cfp(*args, **kwargs)
    if args:
        raise RuntimeError("Fixed no-image CVGA requires keyword-only choose_alpha args")
    from cvpr_exp.cfp_no_image_math import arbitrate
    keys=("student_logits","original_ids","original_log_probs","visual_ids","visual_log_probs",
          "cf_original_ids","cf_original_log_probs","cf_visual_ids","cf_visual_log_probs","step_ids")
    missing=[k for k in keys if kwargs.get(k) is None]
    if missing:
        raise RuntimeError(f"CF teacher tensors missing: {missing}; will not fall back to old visual_signal")
    pure=os.environ.get("CVGA_AP_PURE_TOKEN_MIN_NORM", "0") == "1"
    mode=os.environ.get("CVGA_CF_ARBITRATION_MODE", "pure_token" if pure else "step_average")
    if (mode == "pure_token") != pure and mode != "step_objective":
        raise RuntimeError(f"Mode conflict: {mode=} {pure=}")
    alpha,met=arbitrate(*[kwargs[k] for k in keys],
        mode=mode,
        tau_p=float(os.environ.get("CVGA_CF_TAU_P","0.5")),
        preference_lambda=float(os.environ.get("CVGA_CF_PREFERENCE_LAMBDA","0.5")),
        chunk_tokens=int(kwargs.get("chunk_tokens",128)),
        log_prob_min_clamp=kwargs.get("log_prob_min_clamp",-10.0),
        loss_max_clamp=kwargs.get("loss_max_clamp",10.0))
    global _cvga_cf_noimage_calls
    _cvga_cf_noimage_calls=globals().get("_cvga_cf_noimage_calls",0)+1
    if _cvga_cf_noimage_calls <= 3 or _cvga_cf_noimage_calls % 25 == 0:
        print(f"CVGA_CF_NO_IMAGE_ACTIVE mode={mode} "
              f"preferred_steps={met['preferred_steps']:.0f} "
              f"evidence_steps={met['steps_with_valid_evidence']:.0f} "
              f"mean_lambda={met['mean_lambda']:.6f}",flush=True)
    return alpha,met
