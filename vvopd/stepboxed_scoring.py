"""Multimodal reasoning-step scoring for visual contrastive pairs.

Provide image-conditioned likelihood scoring for candidate
reasoning steps under a shared multimodal reasoning prefix,
with support for edit-token masking.
"""

from __future__ import annotations
import json, os, math, re, hashlib, random, time
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image


# VIPOD_VERIFIED_PRIORITY_V2
def _vipod_v2_original_score_step(engine, model_inputs, step_ids, prefix_len):
    out = engine.module(**model_inputs, use_cache=False)
    logits = out.logits
    start = prefix_len - 1
    if start < 0:
        raise ValueError('Prefix must contain at least one token')
    n = step_ids.shape[0]
    relevant = logits[:, start:start+n, :].float()
    if relevant.shape[1] != n:
        raise ValueError('VCP causal sequence misaligned')
    labels = step_ids.view(1, -1)
    return -F.cross_entropy(relevant.reshape(-1, relevant.shape[-1]), labels.reshape(-1), reduction='mean')

def _score_step(engine, model_inputs, step_ids, prefix_len):
    """Optional token-masked score; loss backprop only through changed labels."""
    _e4_mask = model_inputs.get('_e4_vcp_visual_mask', None)
    if _e4_mask is None:
        return _vipod_v2_original_score_step(engine, model_inputs, step_ids, prefix_len)
    _clean = {k: v for k,v in model_inputs.items() if k != '_e4_vcp_visual_mask'}
    out = engine.module(**_clean, use_cache=False)
    k = step_ids.shape[0]
    relevant = out.logits[:, prefix_len-1:prefix_len-1+k, :].float()
    if relevant.shape[1] != k or _e4_mask.numel() != k:
        raise ValueError('VCP masked causal token alignment mismatch')
    _e4_mask = _e4_mask.to(device=relevant.device, dtype=torch.bool).reshape(-1)
    if not torch.any(_e4_mask):
        raise ValueError('VCP mask empty')
    relevant = relevant[:, _e4_mask, :]
    labels = step_ids.reshape(1, -1)[:, _e4_mask]
    return -torch.nn.functional.cross_entropy(relevant.reshape(-1,relevant.shape[-1]),
                                               labels.reshape(-1), reduction='mean')



def pairwise_loss(pos, neg, beta):
    return F.softplus(-beta * (pos - neg))


def pairs_for_minibatch(question_ids, pair_by_source, update=0):
    """Dedup rollout copies, match question; cycle ONE pair per question/update.

    pair_by_source maps int -> list[pair]. If multiple validated corrections
    exist for the same question, round-robin across optimizer updates without
    giving that question a larger effective preference weight.
    """
    if question_ids is None:
        raise RuntimeError('E4 QA: question IDs missing; refusing random VCP fallback')
    if len(question_ids) == 0:
        raise RuntimeError('E4 QA: empty mini-batch question IDs')
    # VIPOD_QA_PADDING_UID_V1: -1 is reserved solely for tagged synthetic padding.
    _source_rows = [int(q) for q in question_ids]
    if any(q < -1 for q in _source_rows):
        raise RuntimeError('E4 QA: unexpected negative question index')
    unique = list(dict.fromkeys(q for q in _source_rows if q != -1))
    matched = [pair_by_source[q][update % len(pair_by_source[q])]
               for q in unique if q in pair_by_source]
    return unique, matched


class PairState:
    def __init__(self, engine):
        self.path = Path(os.environ['E4_VCP_ASSETS']).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.pairs = [json.loads(line) for line in self.path.read_text(encoding='utf8').splitlines() if line.strip()]
        if not self.pairs:
            raise RuntimeError('E4 QA: empty VCP assets')
        self.pair_by_source = {}
        for pair in self.pairs:
            key = int(pair['source_row_idx'])
            self.pair_by_source.setdefault(key, []).append(pair)
        self.root = self.path.parent
        self.processor = engine.model_config.get_processor()
        self.tok = getattr(self.processor, 'tokenizer', None)
        if self.tok is None:
            raise RuntimeError('E4 QA: processor has no tokenizer')
        self.step = 0
        self._cache = {}
        self._step_ids_cache = {}
        self.beta = float(os.getenv('E4_VCP_BETA', '2.0'))
        self.weight = float(os.getenv('E4_VCP_WEIGHT', '0.01'))
        self.max_tokens = int(os.getenv('E4_VCP_MAX_SEQ_TOKENS', '4096'))
        self.max_step = int(os.getenv('E4_VCP_MAX_STEP_TOKENS', '128'))
        # 0 means all matched questions; positive value bounds GPU VCP work.
        self.max_matched = int(os.getenv('E4_VCP_MAX_MATCHED_PER_UPDATE', '0'))
        self.backward_mode = os.getenv('E4_VCP_BACKWARD_MODE', 'sequential').lower()
        if self.max_matched < 0 or self.backward_mode not in ('sequential', 'joint'):
            raise ValueError('E4 QA FAST: invalid max matched or backward mode')
        self.seed = int(os.getenv('E4_VCP_SEED', '2027'))
        self.exposure = {}
        self.cache_index = None
        self.cache_dir = None
        self._offline_verified = False
        self.validate_cache_first = os.getenv('E4_VCP_VALIDATE_CACHE_FIRST', '1') == '1'
        if os.getenv('E4_VCP_USE_DISK_CACHE', '0') == '1':
            idx_path = Path(os.getenv('E4_VCP_CACHE_INDEX', ''))
            if not idx_path.is_file():
                raise FileNotFoundError(f'E4 QA FAST: missing pretokenized index {idx_path}')
            idx = json.loads(idx_path.read_text(encoding='utf8'))
            manifest_digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if idx.get('version') != 1 or idx.get('manifest_sha256') != manifest_digest:
                raise RuntimeError('E4 QA FAST: stale VCP cache manifest; rerun preprocessing')
            model_name = os.getenv('STUDENT_MODEL', 'Qwen/Qwen3-VL-2B-Instruct')
            if idx.get('model') != model_name or int(idx.get('max_step', -1)) != self.max_step:
                raise RuntimeError('E4 QA FAST: stale processor model or max_step; rebuild cache')
            if len(idx.get('files', {})) != len(self.pairs):
                raise RuntimeError('E4 QA FAST: incomplete cached pair count')
            self.cache_index = idx['files']
            self.cache_dir = idx_path.parent
            print(f'[E4 QA FAST CACHE] {len(self.cache_index)} pretokenized pairs; model={model_name}',flush=True)
        if not (math.isfinite(self.beta) and self.beta > 0 and
                math.isfinite(self.weight) and 0 < self.weight <= 1 and
                self.max_tokens > 0 and self.max_step > 0):
            raise ValueError('E4 QA: invalid VCP hyperparameters')

    def _prefix(self, pair):
        pid = pair['pair_id']
        if pid in self._cache:
            return self._cache[pid]
        if self.cache_index is not None:
            record = self.cache_index.get(str(pid))
            if not record:
                raise RuntimeError(f'E4 QA FAST: no pretokenized pair {pid}')
            cache_file = (self.cache_dir / record).resolve()
            if not cache_file.is_relative_to(self.cache_dir.resolve()):
                raise RuntimeError(f'E4 QA FAST: invalid cache path {record}')
            loaded = torch.load(cache_file, map_location='cpu', weights_only=True)
            cpu = loaded['prefix']
            self._step_ids_cache[pid] = loaded['steps']
            if len(self._cache) >= 64:
                old_pid = next(iter(self._cache))
                self._cache.pop(old_pid)
                self._step_ids_cache.pop(old_pid, None)
            if self.validate_cache_first and not self._offline_verified:
                # Compare ONCE against the live engine processor, to detect
                # cross-version offline preprocessing mismatch before updates.
                _index = self.cache_index
                self.cache_index = None
                self._cache.pop(pid, None)
                try:
                    live = self._prefix(pair)
                finally:
                    self.cache_index = _index
                if set(cpu) != set(live) or any(
                    not torch.equal(cpu[k], live[k]) for k in cpu
                ):
                    raise RuntimeError('E4 QA FAST: offline prefix != live processor output; cache invalid')
                self._offline_verified = True
                print('[E4 QA FAST CACHE] first-pair exact live-processor check PASS', flush=True)
            self._cache[pid] = cpu
            return cpu
        images = []
        for rel in pair['image_paths']:
            with Image.open(self.root / rel) as img:
                images.append(img.convert('RGB'))
        parts = re.split(r'(<image\s*/?>)', pair['math_prompt'], flags=re.I)
        user, nimg = [], 0
        for part in parts:
            if re.fullmatch(r'<image\s*/?>', part or '', re.I):
                user.append({'type': 'image'})
                nimg += 1
            elif part:
                user.append({'type': 'text', 'text': part})
        if nimg != len(images):
            raise ValueError(f'E4 QA image mismatch: {nimg} markers, {len(images)} images, pair={pid}')
        base = self.processor.apply_chat_template([{'role': 'user', 'content': user}],
                                                   tokenize=False, add_generation_prompt=True)
        history = str(pair['previous_reasoning']).replace('<think>', '').replace('</think>', '').strip()
        if history and '<step>' not in history:
            history = '<step>' + history + '</step>'
        # EXP6_STEPBOXED_V1: this prompt does not request <think> output.
        prefix = base + ('\n' + history if history else '') + '\n<step>'
        tensors = self.processor(text=[prefix], images=images, return_tensors='pt')
        ids = tensors['input_ids']
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Unexpected processor input_ids shape')
        cpu = {k: v.detach().cpu() for k, v in tensors.items() if torch.is_tensor(v)}
        if len(self._cache) >= 64:
            old_pid = next(iter(self._cache))
            self._cache.pop(old_pid)
            self._step_ids_cache.pop(old_pid, None)
        self._cache[pid] = cpu
        return cpu

    def _input(self, pair, step_text):
        cpu = self._prefix(pair)
        if self.cache_index is not None:
            if step_text == pair['chosen_step']:
                step_ids = self._step_ids_cache[pair['pair_id']]['chosen']
            elif step_text == pair['rejected_step']:
                step_ids = self._step_ids_cache[pair['pair_id']]['rejected']
            else:
                raise ValueError('VCP step differs from pretokenized verified Pair')
            step_ids = step_ids.tolist()
        else:
            step_ids = self.tok.encode(step_text, add_special_tokens=False)[:self.max_step]
        if not step_ids:
            raise ValueError(f'Empty tokenized step in pair {pair["pair_id"]}')
        if cpu['input_ids'].shape[-1] + len(step_ids) > self.max_tokens:
            raise ValueError(f'Pair {pair["pair_id"]} exceeds E4_VCP_MAX_SEQ_TOKENS; do not truncate prefix')
        device = torch.device('cuda', torch.cuda.current_device())
        processed = {k: v.to(device) for k, v in cpu.items()}
        prefix_ids = processed['input_ids']
        added = torch.tensor(step_ids, dtype=prefix_ids.dtype, device=device).view(1, -1)
        processed['input_ids'] = torch.cat([prefix_ids, added], dim=-1)
        processed['attention_mask'] = torch.ones_like(processed['input_ids'])
        processed.pop('position_ids', None)
        if os.environ.get("E4_VCP_VISUAL_MASK", "0") == "1":
            import importlib as _v2_importlib
            _v2_mask_mod = _v2_importlib.import_module("vvopd.edit_token_mask")
            _v2_mask = _v2_mask_mod.candidate_mask(self.tok, pair, step_text, self.max_step)
            processed["_e4_vcp_visual_mask"] = torch.tensor(_v2_mask, dtype=torch.bool,
                                                    device=processed["input_ids"].device)
        return processed, added[0], prefix_ids.shape[-1]


def choose_matched_pairs(matched, max_matched, exposure, seed, update):
    """Choose only pairs already matching this optimizer mini-batch.

    max_matched==0 uses ALL pairs, preserving strict question-aligned semantics.
    Positive cap implements cheaper sparse VCP; not every eligible question gets
    VCP on that particular update (KL remains active for every question).
    """
    if max_matched == 0 or len(matched) <= max_matched:
        selected = list(matched)
    else:
        shuffled = list(matched)
        random.Random(seed + update).shuffle(shuffled)
        shuffled.sort(key=lambda p: exposure.get(int(p['source_row_idx']), 0))
        selected = shuffled[:max_matched]
    for pair in selected:
        key = int(pair['source_row_idx'])
        exposure[key] = exposure.get(key, 0) + 1
    return selected


def compute_pair_loss(pos_score, neg_score, beta):
    return pairwise_loss(pos_score, neg_score, beta)


def apply_aux_gradients(engine):
    if not getattr(engine, '_e4_vcp_active', False) or os.getenv('E4_VCP_ENABLED') != '1':
        return
    if os.getenv('E4_VCP_QUESTION_ALIGNED') != '1':
        raise RuntimeError('E4 QA FAST: question-aligned flag missing')
    if not hasattr(engine, '_e4_vcp_state'):
        engine._e4_vcp_state = PairState(engine)
        st = engine._e4_vcp_state
        print(f'[E4 QA FAST] init pairs={len(st.pairs)} mode={st.backward_mode} '
              f'cap={st.max_matched} cache={st.cache_index is not None}',flush=True)
    st = engine._e4_vcp_state
    current = getattr(engine, '_e4_current_source_row_idxs', None)
    questions, eligible = pairs_for_minibatch(current, st.pair_by_source, update=st.step)
    selected = choose_matched_pairs(eligible, st.max_matched, st.exposure, st.seed, st.step)
    st.step += 1
    engine._e4_qa_total_questions = len(questions)
    # This is the number actually supervised, NOT the full eligible count when cap>0.
    engine._e4_qa_matched_questions = len(selected)
    engine._e4_vcp_last_weighted_loss = 0.0
    if not selected:
        if st.step <= 3 or st.step % 10 == 0:
            print(f'[E4 QA FAST] update={st.step} unique_questions={len(questions)} '
                  f'eligible=0 used=0 KL_ONLY=True', flush=True)
        return

    dtype = getattr(engine, '_autocast_dtype', torch.bfloat16)
    ctx = lambda: nullcontext() if dtype == torch.float32 else torch.autocast('cuda',dtype=dtype)
    scale = getattr(engine, 'scaler', None)
    # VIPOD_V2_SAFE_PAIR_PREFILTER_V1
    # Filter BEFORE backward so skipped pairs do not dilute the effective
    # VCP preference mean. Never truncate candidate tokens or mask lengths.
    start = time.perf_counter()
    effective_pairs = []
    # MASKED_V3_NO_NONMINIMAL_FILTER: editing score and quality gating are independent.
    _v2_mask_enabled = (os.environ.get("E4_VCP_FILTER_NONMINIMAL", "0") == "1")
    if _v2_mask_enabled:
        import importlib as _v2_importlib
        _v2_mask_mod = _v2_importlib.import_module("vvopd.edit_token_mask")
    for pair in selected:
        if _v2_mask_enabled:
            try:
                _v2_quality = _v2_mask_mod.pair_quality(st.tok, pair, st.max_step)
            except ValueError as _v2_exc:
                # Only the known length-policy failure becomes a skip.
                # Unknown tokenizer/mask/data errors must still fail closed.
                if str(_v2_exc) != 'Candidate step exceeds max_step; refuse masked preference':
                    raise
                _v2_quality = {
                    'eligible': False,
                    'reason': 'candidate_step_too_long',
                    'pair_id': str(pair.get('pair_id', '')),
                }
            if not _v2_quality['eligible']:
                _v2_reason = str(_v2_quality['reason'])
                if _v2_reason == 'candidate_step_too_long':
                    st._v2_too_long_skipped = getattr(st, '_v2_too_long_skipped', 0) + 1
                    _v2_skip_count = st._v2_too_long_skipped
                else:
                    st._v2_nonminimal_skipped = getattr(st, '_v2_nonminimal_skipped', 0) + 1
                    _v2_skip_count = st._v2_nonminimal_skipped
                if _v2_skip_count <= 3 or _v2_skip_count % 100 == 0:
                    print('[VIPOD V2 SKIP] reason=' + _v2_reason +
                          ' pair=' + str(_v2_quality['pair_id']) +
                          ' max_step=' + str(st.max_step) +
                          ' cumulative_reason_count=' + str(_v2_skip_count), flush=True)
                continue
        effective_pairs.append(pair)

    # This is the number ACTUALLY used for VCP supervision, not the
    # pre-filter candidate count. The trainer consumes this coverage value.
    engine._e4_qa_matched_questions = len(effective_pairs)
    if not effective_pairs:
        if st.step <= 5 or st.step % 10 == 0:
            print(f'[E4 QA FAST] update={st.step} unique_questions={len(questions)} '
                  f'eligible={len(eligible)} selected={len(selected)} used=0 '
                  f'KL_ONLY=True filtered={len(selected)}', flush=True)
        return

    per_pair_weight = st.weight / len(effective_pairs)
    losses = []
    for pair in effective_pairs:
        pos, pos_ids, pos_start = st._input(pair, pair['chosen_step'])
        neg, neg_ids, neg_start = st._input(pair, pair['rejected_step'])
        if st.backward_mode == 'joint':
            # True preference loss, two forwards and ONE backward.
            # Requires both Qwen3-VL FSDP activation graphs to be live: pilot first.
            with ctx():
                pos_val = _score_step(engine, pos, pos_ids, pos_start)
                neg_val = _score_step(engine, neg, neg_ids, neg_start)
                weighted = per_pair_weight * compute_pair_loss(pos_val, neg_val, st.beta)
            losses.append(float(weighted.detach().item()))
            if scale is not None:
                scale.scale(weighted).backward()
            else:
                weighted.backward()
            del pos_val, neg_val, weighted
        else:
            # Memory-conservative 3F/2B mode. Equivalent gradients at fixed
            # model state, except for stochastic dropout / forward state.
            with torch.no_grad(),ctx():
                neg_val = _score_step(engine, neg, neg_ids, neg_start).detach()
            with ctx():
                pos_val = _score_step(engine, pos, pos_ids, pos_start)
                weighted = per_pair_weight * compute_pair_loss(pos_val, neg_val, st.beta)
            losses.append(float(weighted.detach().item()))
            if scale is not None:
                scale.scale(weighted).backward()
            else:
                weighted.backward()
            with torch.no_grad():
                coeff = (per_pair_weight * st.beta
                         * torch.sigmoid(-st.beta * (pos_val.detach() - neg_val)))
            del pos_val, weighted
            with ctx():
                neg_grad = _score_step(engine, neg, neg_ids, neg_start)
                neg_term = coeff.detach() * neg_grad
            if scale is not None:
                scale.scale(neg_term).backward()
            else:
                neg_term.backward()
            del neg_grad, neg_term
        del pos, neg
    engine._e4_vcp_last_weighted_loss = sum(losses)
    elapsed = time.perf_counter() - start
    if st.step <= 5 or st.step % 10 == 0:
        print(f'[E4 QA FAST] update={st.step} unique_questions={len(questions)} '
              f'eligible={len(eligible)} selected={len(selected)} 'f'used={len(effective_pairs)} filtered={len(selected)-len(effective_pairs)} '
              f'KL_ONLY=False mode={st.backward_mode} vcp_seconds={elapsed:.2f} '
              f'weighted_vcp_loss={engine._e4_vcp_last_weighted_loss:.6f}', flush=True)
