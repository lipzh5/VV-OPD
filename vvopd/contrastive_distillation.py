"""VV-OPD: reward-gated visual contrastive distillation.

Apply verified contrastive supervision to intermediate reasoning
steps using edit-token-masked preference ranking.

The hook implements:
    - Question-aligned verified contrastive pair selection.
    - Rollout reward-based gating.
    - Edit-token-masked likelihood scoring.
    - Batched multimodal forward and backward computation.
    - Static-weight auxiliary gradient integration without projection.
"""

from __future__ import annotations
import json, os, math, re, hashlib, random, time
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from importlib import import_module
ALIGN = import_module("vvopd.qwen3vl_input_alignment")



# VIRL862_VCP_IMAGE_FORWARD_AUDIT_V2: demonstrate images reach actual model forward.
def _vcp_image_forward_audit(engine, model_inputs, stage):
    """Strictly check image patches and image grid before a real model forward.

    First successful model forward logs a concise certificate. Does not alter
    any tensor, gradient, loss or input. One log line per actor process.
    """
    if os.environ.get('E4_VCP_IMAGE_FORWARD_AUDIT','1') != '1':
        return None
    required={'input_ids','pixel_values','image_grid_thw'}
    if not required <= set(model_inputs):
        raise RuntimeError(f'[VCP IMAGE ABORT] {stage}: missing {required-set(model_inputs)}')
    pixels=model_inputs['pixel_values']
    grid=model_inputs['image_grid_thw']
    ids=model_inputs['input_ids']
    if not all(torch.is_tensor(x) for x in (pixels,grid,ids)):
        raise RuntimeError(f'[VCP IMAGE ABORT] {stage}: multimodal inputs not tensors')
    if pixels.ndim!=2 or pixels.numel()==0 or grid.ndim!=2 or grid.shape[-1]!=3 or grid.numel()==0:
        raise RuntimeError(f'[VCP IMAGE ABORT] {stage}: empty or invalid vision tensors')
    if grid.shape[0]<1 or ids.shape[0]<1 or ids.numel()==0:
        raise RuntimeError(f'[VCP IMAGE ABORT] {stage}: no image or text tokens')
    if pixels.device != ids.device or grid.device != ids.device:
        raise RuntimeError(f'[VCP IMAGE ABORT] {stage}: text/image tensors on different devices')
    if getattr(engine,'_e4_vcp_image_forward_audit_logged',False):
        return None
    return (stage,tuple(pixels.shape),tuple(grid.shape),str(pixels.device))


def _vcp_image_forward_log_after_success(engine, info):
    if info is not None and not getattr(engine,'_e4_vcp_image_forward_audit_logged',False):
        stage,pixel_shape,grid_shape,device=info
        print(f'[VCP IMAGE FORWARD PASS] stage={stage} pixel_values={pixel_shape} '
              f'image_grid_thw={grid_shape} device={device} '
              'engine.module forward succeeded',flush=True)
        engine._e4_vcp_image_forward_audit_logged=True


# VIPOD_VERIFIED_PRIORITY_V2
def _vipod_v2_original_score_step(engine, model_inputs, step_ids, prefix_len):
    _vision_info=_vcp_image_forward_audit(engine,model_inputs,'v2_reference')
    out = engine.module(**model_inputs, use_cache=False)
    _vcp_image_forward_log_after_success(engine,_vision_info)
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
    """Score the exact V3 visual edit tokens without CUDA boolean masking.

    The source V3 uses logits[:, boolean_mask, :].  V4 builds nonzero token
    indices once on CPU before any forward, then uses CUDA index_select; the
    selected tokens and length-normalized log likelihood are identical.
    """
    indices = model_inputs.get('_e4_vcp_visual_indices')
    if indices is None:
        return _vipod_v2_original_score_step(engine, model_inputs, step_ids, prefix_len)
    inputs = {k: v for k, v in model_inputs.items() if k != '_e4_vcp_visual_indices'}
    _vision_info=_vcp_image_forward_audit(engine,inputs,'vcp_single')
    out = engine.module(**inputs, use_cache=False)
    _vcp_image_forward_log_after_success(engine,_vision_info)
    k = step_ids.shape[0]
    relevant = out.logits[:, prefix_len-1:prefix_len-1+k, :]
    if relevant.shape[1] != k or indices.numel() == 0:
        raise ValueError('PLAIN VCP: empty/misaligned causal selected indices')
    if indices.dtype != torch.long or indices.device != relevant.device:
        raise ValueError('PLAIN VCP: selected indices device/dtype mismatch')
    relevant = relevant.index_select(1, indices).float()
    labels = step_ids.index_select(0, indices).reshape(1, -1)
    return -F.cross_entropy(relevant.reshape(-1, relevant.shape[-1]),
                            labels.reshape(-1), reduction='mean')



def pairwise_loss(pos, neg, beta):
    """Legacy unreferenced VCP for local diagnostic comparisons only."""
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
        self._v4_visual_token_cache = {}  # immutable per-pair token IDs + mask indices
        self.beta = float(os.getenv('E4_VCP_BETA', '2.0'))
        self.weight = float(os.getenv('E4_VCP_WEIGHT', '0.01'))
        self.max_tokens = int(os.getenv('E4_VCP_MAX_SEQ_TOKENS', '4096'))
        self.max_step = int(os.getenv('E4_VCP_MAX_STEP_TOKENS', '2048'))
        # Plain VCP: no teacher preference scores, no error-margin, no ref cache.
        if os.getenv('E4_VCP_VISUAL_MASK') != '1':
            raise RuntimeError('Plain VCP requires E4_VCP_VISUAL_MASK=1')
        if os.getenv('E4_VCP_FILTER_NONMINIMAL') != '0':
            raise RuntimeError('Plain VCP requires E4_VCP_FILTER_NONMINIMAL=0')
        if os.getenv('E4_VCP_PRIORITY_ENABLED') != '0' or os.getenv('E4_KLFIRST_ENABLED') != '0':
            raise RuntimeError('Plain VCP forbids KL/VCP gradient projection')
        print(f'[PLAIN VCP NOREF] pairs={len(self.pairs)} teacher_reference=OFF '
              f'margin=0 projection=OFF beta={self.beta} lambda={self.weight}', flush=True)
        # 0 means all matched questions; positive value bounds GPU VCP work.
        self.max_matched = int(os.getenv('E4_VCP_MAX_MATCHED_PER_UPDATE', '0'))
        self.backward_mode = os.getenv('E4_VCP_BACKWARD_MODE', 'batched').lower()
        self.batch_pairs = int(os.getenv('E4_VCP_BATCH_PAIRS', '4'))
        self.batch_max_padded_tokens = int(os.getenv('E4_VCP_BATCH_MAX_PADDED_TOKENS', '8192'))
        self.batch_verify_first = os.getenv('E4_VCP_BATCH_VERIFY_FIRST', '1') == '1'
        self.batch_verified = False
        if not 1 <= self.batch_pairs <= 16 or self.batch_max_padded_tokens < 256:
            raise ValueError('V45 invalid batching limits')
        if self.max_matched < 0 or self.backward_mode not in ('sequential', 'joint', 'batched'):
            raise ValueError('BASE2B PLAIN VCP: invalid max matched or backward mode')
        self.seed = int(os.getenv('E4_VCP_SEED', '2027'))
        self.exposure = {}
        self.cache_index = None
        self.cache_dir = None
        self._offline_verified = False
        self.validate_cache_first = os.getenv('E4_VCP_VALIDATE_CACHE_FIRST', '1') == '1'
        if os.getenv('E4_VCP_USE_DISK_CACHE', '0') == '1':
            idx_path = Path(os.getenv('E4_VCP_CACHE_INDEX', ''))
            if not idx_path.is_file():
                raise FileNotFoundError(f'BASE2B PLAIN VCP: missing pretokenized index {idx_path}')
            idx = json.loads(idx_path.read_text(encoding='utf8'))
            manifest_digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if idx.get('version') != 1 or idx.get('manifest_sha256') != manifest_digest:
                raise RuntimeError('BASE2B PLAIN VCP: stale VCP cache manifest; rerun preprocessing')
            model_name = os.getenv('STUDENT_MODEL', 'Qwen/Qwen3-VL-2B-Instruct')
            if idx.get('model') != model_name or int(idx.get('max_step', -1)) != self.max_step:
                raise RuntimeError('BASE2B PLAIN VCP: stale processor model or max_step; rebuild cache')
            if len(idx.get('files', {})) != len(self.pairs):
                raise RuntimeError('BASE2B PLAIN VCP: incomplete cached pair count')
            self.cache_index = idx['files']
            self.cache_dir = idx_path.parent
            print(f'[E4 QA FAST CACHE] {len(self.cache_index)} pretokenized pairs; model={model_name}',flush=True)
        if not (math.isfinite(self.beta) and self.beta > 0 and
                math.isfinite(self.weight) and 0 < self.weight <= 1 and
                self.max_tokens > 0 and self.max_step > 0):
            raise ValueError('E4 QA: invalid VCP hyperparameters')

        # The experiment uses a fixed scalar VCP weight; EMA is deliberately absent.
        if os.environ.get('E4_VCP_EMA_ENABLED', '0') != '0':
            raise RuntimeError('Base2B PlainVCP NoProj run forbids EMA; E4_VCP_EMA_ENABLED must be 0')
        if self.backward_mode != 'batched':
            raise RuntimeError('Base2B PlainVCP NoProj requires V4.5 batched backward')

    def _prefix(self, pair):
        pid = pair['pair_id']
        if pid in self._cache:
            return self._cache[pid]
        if self.cache_index is not None:
            record = self.cache_index.get(str(pid))
            if not record:
                raise RuntimeError(f'BASE2B PLAIN VCP: no pretokenized pair {pid}')
            cache_file = (self.cache_dir / record).resolve()
            if not cache_file.is_relative_to(self.cache_dir.resolve()):
                raise RuntimeError(f'BASE2B PLAIN VCP: invalid cache path {record}')
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
                    raise RuntimeError('BASE2B PLAIN VCP: offline prefix != live processor output; cache invalid')
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
        pid = str(pair['pair_id'])
        if pid not in self._v4_visual_token_cache:
            # Same V3 all-edited-token mask including the insertion/deletion
            # boundary fallback. Build it once per pair, not every forward.
            mask_module = import_module('vvopd.edit_token_mask')
            plus_mask, minus_mask, _ = mask_module.paired_masks(
                self.tok, str(pair['chosen_step']), str(pair['rejected_step']), self.max_step)
            if self.cache_index is not None:
                plus_ids = self._step_ids_cache[pid]['chosen'].tolist()
                minus_ids = self._step_ids_cache[pid]['rejected'].tolist()
            else:
                plus_ids = self.tok.encode(str(pair['chosen_step']), add_special_tokens=False)
                minus_ids = self.tok.encode(str(pair['rejected_step']), add_special_tokens=False)
            if len(plus_ids) != len(plus_mask) or len(minus_ids) != len(minus_mask):
                raise ValueError('PLAIN VCP: token mask/cache length mismatch')
            plus_indices = [i for i, bit in enumerate(plus_mask) if bit]
            minus_indices = [i for i, bit in enumerate(minus_mask) if bit]
            if not plus_indices or not minus_indices:
                raise ValueError(f'PLAIN VCP: zero selected tokens for {pid}')
            self._v4_visual_token_cache[pid] = (
                (plus_ids, plus_indices), (minus_ids, minus_indices))
        if step_text == pair['chosen_step']:
            step_ids, selected_indices = self._v4_visual_token_cache[pid][0]
        elif step_text == pair['rejected_step']:
            step_ids, selected_indices = self._v4_visual_token_cache[pid][1]
        else:
            raise ValueError('VCP step differs from pretokenized verified Pair')
        if not step_ids:
            raise ValueError(f'Empty tokenized step in pair {pid}')
        if cpu['input_ids'].shape[-1] + len(step_ids) > self.max_tokens:
            raise ValueError(f'Pair {pid} exceeds E4_VCP_MAX_SEQ_TOKENS; do not truncate prefix')
        device = torch.device('cuda', torch.cuda.current_device())
        processed = {k: v.to(device) for k, v in cpu.items()}
        prefix_ids = processed['input_ids']
        added = torch.tensor(step_ids, dtype=prefix_ids.dtype, device=device).view(1, -1)
        processed['input_ids'] = torch.cat([prefix_ids, added], dim=-1)
        processed['attention_mask'] = torch.ones_like(processed['input_ids'])
        processed.pop('position_ids', None)
        ALIGN.align_appended_text_step(processed, prefix_ids.shape[-1], added.shape[-1])
        processed['_e4_vcp_visual_indices'] = torch.tensor(
            selected_indices, dtype=torch.long, device=device)
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


def compute_pair_loss(pos_score, neg_score, beta, pair, state):
    """Plain masked visual-consistency preference; no teacher-reference margin."""
    return F.softplus(-beta * (pos_score - neg_score))


def reference_threshold(pair, state):
    """All verified pairs have ZERO margin; no reference scores are loaded."""
    return 0.0


# V45_QWEN3VL_IMAGE_BATCH_V1: strict image/prefix/step-score preserving VCP batching.
_V45_TEXT_KEYS = frozenset(('input_ids', 'attention_mask', 'mm_token_type_ids', 'token_type_ids'))
_V45_IMAGE_KEYS = frozenset(('pixel_values', 'image_grid_thw'))
_V45_ALLOWED_KEYS = _V45_TEXT_KEYS | _V45_IMAGE_KEYS


def _v45_pack_multimodal(rows):
    """Right-pad text; concatenate image patches/grids IN ROW ORDER.

    Each row has independently aligned Qwen3-VL multimodal positions. Model's
    get_rope_index uses the attention mask to remove right padding and consumes
    image_grid_thw in the order of image-token appearances across batch rows.
    No image is dropped, averaged, or shifted between chosen/rejected rows.
    """
    if not rows:
        raise ValueError('V45: no rows')
    keys = set(rows[0]) - {'_e4_vcp_visual_indices'}
    if not {'input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw'} <= keys:
        raise RuntimeError(f'V45 requires image-conditioned Qwen3-VL inputs; keys={sorted(keys)}')
    if keys - _V45_ALLOWED_KEYS:
        raise RuntimeError(f'V45 unsupported processor keys: {sorted(keys - _V45_ALLOWED_KEYS)}')
    for i, row in enumerate(rows):
        row_keys = set(row) - {'_e4_vcp_visual_indices'}
        if row_keys != keys:
            raise RuntimeError(f'V45: inconsistent multimodal keys in row {i}')
        ids = row['input_ids']
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('V45: row input_ids not [1,L]')
        if row['attention_mask'].shape != ids.shape:
            raise ValueError('V45: attention mask has wrong shape')
        for k in keys & _V45_TEXT_KEYS:
            if row[k].shape != ids.shape:
                raise ValueError(f'V45: {k} has mismatched sequence length')
        if row['image_grid_thw'].ndim != 2 or row['image_grid_thw'].shape[1] != 3:
            raise ValueError('V45: incorrect image_grid_thw')
        if row['pixel_values'].ndim != 2:
            raise ValueError('V45: pixel_values must be flattened Qwen3-VL image patches')
    max_len=max(int(x['input_ids'].shape[1]) for x in rows)
    n=len(rows)
    packed={}
    for k in keys & _V45_TEXT_KEYS:
        first=rows[0][k]
        arr=torch.full((n,max_len), fill_value=0, dtype=first.dtype, device=first.device)
        for i,row in enumerate(rows):
            size=row[k].shape[1]
            arr[i,:size] = row[k][0]
        packed[k]=arr
    for k in keys & _V45_IMAGE_KEYS:
        packed[k]=torch.cat([x[k] for x in rows],dim=0)
    return packed


def _v45_score_rows(logits, sides):
    """Exactly V43's masked MEAN log-p for each row; no averaging across rows."""
    row_indices=[]
    time_indices=[]
    labels=[]
    counts=[]
    for row_id,(processed,step_ids,prefix_len) in enumerate(sides):
        ix=processed['_e4_vcp_visual_indices']
        if ix.dtype != torch.long or ix.numel() == 0:
            raise ValueError('V45: invalid/empty V3 visual indices')
        k=step_ids.shape[0]
        if int(prefix_len)+k != processed['input_ids'].shape[1]:
            raise ValueError('V45: step boundary mismatch')
        row_indices.append(torch.full((ix.numel(),),row_id,dtype=torch.long,device=ix.device))
        time_indices.append(ix + int(prefix_len) - 1)
        labels.append(step_ids.index_select(0,ix))
        counts.append(ix.numel())
    r=torch.cat(row_indices)
    t=torch.cat(time_indices)
    target=torch.cat(labels)
    if r.device != logits.device or t.device != logits.device or target.device != logits.device:
        raise RuntimeError('V45: visual-mask indices and logits on different devices')
    selected=logits[r,t,:].float()
    loss_each=F.cross_entropy(selected,target,reduction='none')
    totals=loss_each.new_zeros(len(sides)).index_add(0,r,loss_each)
    denoms=loss_each.new_tensor(counts)
    return -totals / denoms


def _v45_backward(engine, weighted, scale):
    if scale is not None:
        scale.scale(weighted).backward()
    else:
        weighted.backward()


def _v45_joint_single(engine, pair, pos, pos_ids, pos_start,
                      neg, neg_ids, neg_start, per_pair_weight, st, ctx, scale):
    """Unmodified V43 two-forward/one-backward fallback for outlier lengths."""
    with ctx():
        pos_val=_score_step(engine,pos,pos_ids,pos_start)
        neg_val=_score_step(engine,neg,neg_ids,neg_start)
        weighted=per_pair_weight*compute_pair_loss(pos_val,neg_val,st.beta,pair,st)
    _v45_backward(engine,weighted,scale)
    return weighted.detach()


def _v45_execute_chunk(engine, chunk, per_pair_weight, st, ctx, scale):
    """One Qwen3-VL forward + one backward for 1..N independent pairs."""
    rows=[]
    pairs=[]
    for pair,pos,neg in chunk:
        pairs.append(pair)
        rows.extend((pos,neg))
    inputs=_v45_pack_multimodal([r[0] for r in rows])
    reference=[]
    # First-chunk fail-closed numerical check against the unbatched V43 scorer.
    # Train model normally contains no dropout for Qwen3-VL. Disable with
    # E4_VCP_BATCH_VERIFY_FIRST=0 only after independent GPU equivalence tests.
    if st.batch_verify_first and not st.batch_verified:
        with torch.no_grad(), ctx():
            reference=[_score_step(engine,*rows[j]).detach() for j in (0,1)]
    with ctx():
        _vision_info=_vcp_image_forward_audit(engine,inputs,'vcp_batched')
        logits=engine.module(**inputs,use_cache=False).logits
        _vcp_image_forward_log_after_success(engine,_vision_info)
        scores=_v45_score_rows(logits,rows)
        if reference:
            reference=torch.stack(reference).float()
            actual=scores[:2].detach().float()
            difference=(reference-actual).abs().max().item()
            if not torch.allclose(reference,actual,atol=0.06,rtol=0.015):
                raise RuntimeError('V45 batch does NOT match V43 single-pair scores: '
                                   f'max_abs={difference:.6f}; ABORT before optimizer update')
            st.batch_verified=True
            print(f'[V45 BATCH EQUIVALENCE PASS] first_pair_max_abs={difference:.6f}',flush=True)
        thresholds=scores.new_tensor([reference_threshold(pair,st) for pair in pairs])
        pair_losses=F.softplus(-st.beta*(scores[::2]-scores[1::2]-thresholds))
        weighted=per_pair_weight*pair_losses.sum()
    _v45_backward(engine,weighted,scale)
    return weighted.detach()


def _v45_batch_pairs(engine,effective_pairs,st,per_pair_weight,ctx,scale):
    """Greedy padded-token budget; long or unsupported pairs use V43 fallback."""
    losses=[]
    chunk=[]
    n_chunks=0
    n_fallback=0
    n_forward=0
    n_backward=0
    padded_tokens=0
    total_tokens=0
    max_len=0
    prep_seconds=0.0
    compute_seconds=0.0

    def flush():
        nonlocal chunk,n_chunks,n_forward,n_backward,padded_tokens,compute_seconds
        if not chunk:
            return
        t=time.perf_counter()
        losses.append(_v45_execute_chunk(engine,chunk,per_pair_weight,st,ctx,scale))
        compute_seconds+=time.perf_counter()-t
        n_chunks+=1
        n_forward+=1
        n_backward+=1
        padded_tokens+=2*len(chunk)*max(
            max(int(x[1][0]['input_ids'].shape[1]),int(x[2][0]['input_ids'].shape[1]))
            for x in chunk)
        chunk=[]

    for pair in effective_pairs:
        t=time.perf_counter()
        pos=st._input(pair,pair['chosen_step'])
        neg=st._input(pair,pair['rejected_step'])
        prep_seconds+=time.perf_counter()-t
        pair_len=max(int(pos[0]['input_ids'].shape[1]),int(neg[0]['input_ids'].shape[1]))
        total_tokens+=pair_len
        max_len=max(max_len,pair_len)
        if pair_len*2>st.batch_max_padded_tokens:
            flush()
            t=time.perf_counter()
            losses.append(_v45_joint_single(engine,pair,*pos,*neg,
                           per_pair_weight,st,ctx,scale))
            compute_seconds+=time.perf_counter()-t
            n_fallback+=1
            n_forward+=2
            n_backward+=1
            continue
        next_max=max([pair_len]+[
            max(int(x[1][0]['input_ids'].shape[1]),int(x[2][0]['input_ids'].shape[1]))
            for x in chunk])
        if chunk and (len(chunk)>=st.batch_pairs or
                      2*(len(chunk)+1)*next_max>st.batch_max_padded_tokens):
            flush()
        chunk.append((pair,pos,neg))
    flush()
    return (losses,prep_seconds,compute_seconds,total_tokens,max_len,
            n_chunks,n_fallback,n_forward,n_backward,padded_tokens)



# Save KL gradients, compute an independently differentiated RAW VCP gradient,
# then ADD lambda * VCP to the original KL, with NO projection. The coefficient is constant;
# no gradient-norm EMA, adaptive scaling, or projection on correct-only batches.

def _v46_trainable_params(engine):
    parameters=[p for p in engine.module.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError('Fixed VCPfirst: no trainable parameters exposed by FSDP engine')
    if any(p.grad is not None and p.grad.is_sparse for p in parameters):
        raise RuntimeError('Fixed VCPfirst: sparse gradients are unsupported')
    return parameters


def _v46_before_vcp(st, engine):
    """Snapshot the pre-VCP KL-side gradients; isolate raw VCP.

    Called after VERL's KL backward and before VCP backward. The pre-VCP
    gradient norm is the effective KL-side gradient visible to this optimizer
    update. VCP is isolated by zeroing the saved grads before its backward.
    """
    params=_v46_trainable_params(engine)
    with torch.no_grad():
        before=[]
        sq_kl=None
        for p in params:
            g=p.grad
            clone=None if g is None else g.detach().clone()
            before.append(clone)
            if clone is None:
                continue
            gf=clone.float()
            term=(gf*gf).sum(dtype=torch.float32)
            sq_kl=term if sq_kl is None else sq_kl+term
            # Preserve the existing tensor allocation if possible, but do not
            # destroy the snapshot needed to restore accumulated gradients.
            g.zero_()
        if sq_kl is None:
            raise RuntimeError('Fixed VCPfirst: KL gradient is entirely absent; wrong hook order')
        norm_kl=float(torch.sqrt(sq_kl).item())
        if not math.isfinite(norm_kl) or norm_kl<=0:
            raise RuntimeError(f'Fixed VCPfirst: invalid KL norm {norm_kl}')
    return params,before,norm_kl


def _fixed_vcpfirst_finish(st, params, before, norm_kl, raw_loss):
    """g = g_KL + lambda*g_V; NO projection, even when dot(g_KL,g_V)<0.

    The raw VCP gradient is computed separately from the saved mini-batch KL
    gradient; the final update adds lambda * VCP, without projection.
    On an all-correct mini-batch, this function is NEVER invoked.
    """
    lam = st.weight
    with torch.no_grad():
        dot_tensor = None
        v2_tensor = None
        for p, kl in zip(params, before):
            v = p.grad
            if v is None:
                continue
            vf = v.detach().float()
            v2 = (vf*vf).sum()
            v2_tensor = v2 if v2_tensor is None else v2_tensor+v2
            if kl is not None:
                d = (kl.detach().float()*vf).sum()
                dot_tensor = d if dot_tensor is None else dot_tensor+d
        if v2_tensor is None:
            raise RuntimeError('Base2B PlainVCP NoProj: no raw VCP gradient produced')
        if dot_tensor is None:
            # All KL slots may be absent for parameters hit by VCP. A valid
            # KL gradient elsewhere still exists; dot is zero in this case.
            dot_tensor = torch.zeros_like(v2_tensor)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(dot_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(v2_tensor, op=torch.distributed.ReduceOp.SUM)
        dot = float(dot_tensor.item())
        v2 = float(v2_tensor.item())
        if not math.isfinite(dot) or not math.isfinite(v2) or v2 <= 1e-24:
            raise RuntimeError(f'Base2B PlainVCP NoProj: degenerate dot/norm ({dot}, {v2})')
        norm_vcp = math.sqrt(v2)
        # Exp B: NO PROJECTION. Exactly g_total = g_KL + lambda*g_VCP.
        # Dot/cosine are monitoring only and NEVER alter the update.
        projection = 0.0
        effective_coeff = lam
        for p, kl in zip(params,before):
            if p.grad is None:
                if kl is not None:
                    p.grad=kl
            else:
                p.grad.mul_(effective_coeff)
                if kl is not None:
                    p.grad.add_(kl)
        metrics = {
          'e4/grad_noproj_lambda':float(lam),
          'e4/grad_noproj_conflict':float(dot<0.),
          'e4/grad_noproj_kl_vcp_cosine':float(dot/(norm_kl*norm_vcp+1e-24)),
          'e4/grad_noproj_projection_boost':float(-projection),
          'e4/grad_noproj_effective_vcp_coeff':float(effective_coeff),
          'e4/grad_noproj_effective_ratio':float(effective_coeff*norm_vcp/(norm_kl+1e-24)),
          'e4/grad_noproj_kl_norm':float(norm_kl),
          'e4/grad_noproj_vcp_norm':float(norm_vcp),
        }
        st.last_vcpfirst_metrics=metrics
    return float(raw_loss)*lam,metrics


def _fixed_run_vcpfirst_batch(engine, st, pairs, ctx, scale, start):
    params,before,norm_kl=_v46_before_vcp(st,engine)
    try:
        # Backward unweighted VCP so projection uses its TRUE direction.
        (losses,prep_s,compute_s,tokens,max_len,chunks,fallback,fwd,bwd,padded)=_v45_batch_pairs(
            engine,pairs,st,1./len(pairs),ctx,scale)
        raw_loss=float(torch.stack(losses).sum().item())
        weighted_loss,metrics=_fixed_vcpfirst_finish(st,params,before,norm_kl,raw_loss)
    except BaseException:
        with torch.no_grad():
            for p,old in zip(params,before):
                p.grad=None if old is None else old
        raise
    engine._e4_vcp_last_weighted_loss=weighted_loss
    engine._e4_vcp_last_grad_diag=metrics
    elapsed=time.perf_counter()-start
    if st.step<=5 or st.step%10==0:
        print(f'[BASE2B PLAIN VCP NOPROJ] update={st.step} pairs={len(pairs)} '
              f'lambda={st.weight:.6g} conflict={int(metrics["e4/grad_noproj_conflict"])} '
              f'projection_boost={metrics["e4/grad_noproj_projection_boost"]:.5g} '
              f'effective_ratio={metrics["e4/grad_noproj_effective_ratio"]:.5g} '
              f'weighted_loss={weighted_loss:.6g}',flush=True)
        print(f'[BASE2B PLAIN VCP PERF] update={st.step} chunks={chunks} pairs={len(pairs)} '
              f'compute_wall_s={compute_s:.2f} total_s={elapsed:.2f}',flush=True)


def apply_aux_gradients(engine):
    if not getattr(engine, '_e4_vcp_active', False) or os.getenv('E4_VCP_ENABLED') != '1':
        return
    if os.getenv('E4_VCP_PRIORITY_ENABLED', '0') != '0':
        raise RuntimeError('Plain VCP forbids Priority gradient projection')
    if os.getenv('E4_KLFIRST_ENABLED', '0') == '1':
        raise RuntimeError('Plain VCP forbids KL-first projection')
    if os.getenv('E4_VCP_QUESTION_ALIGNED') != '1':
        raise RuntimeError('BASE2B PLAIN VCP: question-aligned flag missing')
    if not hasattr(engine, '_e4_vcp_state'):
        engine._e4_vcp_state = PairState(engine)
        st = engine._e4_vcp_state
        print(f'[E4 QA FAST] init pairs={len(st.pairs)} mode={st.backward_mode} '
              f'cap={st.max_matched} cache={st.cache_index is not None}',flush=True)
    st = engine._e4_vcp_state
    current = getattr(engine, '_e4_current_source_row_idxs', None)
    questions, eligible = pairs_for_minibatch(current, st.pair_by_source, update=st.step)
    selected = choose_matched_pairs(eligible, st.max_matched, st.exposure, st.seed, st.step)
    # No fallback to neutral reward: the worker must attach true per-response
    # math accuracy before the KL backward.  A question is VCP-eligible if at
    # least one of its actual on-policy rollouts received acc=0.
    if os.getenv('E4_REWARDGATE_ENABLED','0') != '1':
        raise RuntimeError('RewardGate VCPfirst requires E4_REWARDGATE_ENABLED=1')
    wrong = getattr(engine, '_e4_rewardgate_wrong_questions', None)
    correct_flags = getattr(engine, '_e4_rewardgate_acc_flags', None)
    if wrong is None or correct_flags is None or len(correct_flags) != len(current):
        raise RuntimeError('RewardGate VCPfirst: accurate per-response reward flags missing')
    selected = [pair for pair in selected if int(pair['source_row_idx']) in wrong]
    engine._e4_rewardgate_vcp_question_count = len(selected)
    if st.step < 5 or st.step % 10 == 0:
        print('[REWARDGATE]', 'samples=',len(correct_flags),
              'correct=',sum(correct_flags),'incorrect=',len(correct_flags)-sum(correct_flags),
              'wrong_questions=',len(wrong),'vcp_pairs=',len(selected),flush=True)
    st.step += 1
    engine._e4_qa_total_questions = len(questions)
    # This is the number actually supervised, NOT the full eligible count when cap>0.
    engine._e4_qa_matched_questions = len(selected)
    engine._e4_vcp_last_weighted_loss = 0.0
    engine._e4_vcp_last_grad_diag = None  # no stale gradient telemetry
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

    return _fixed_run_vcpfirst_batch(engine,st,effective_pairs,ctx,scale,start)
