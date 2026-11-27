#!/usr/bin/env python3
"""Reward-based gating for verified visual contrastive supervision.

Evaluate on-policy rollout correctness and identify questions
eligible for visual consistency supervision based on their
observed rollout outcomes.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

ID_FIELDS = ("source_row_idx", "e4_source_row_idx", "source_idx", "row_idx", "original_source_row_idx")


def source_id(extra):
    if not isinstance(extra, dict):
        raise RuntimeError(f"RewardGate: extra_info must be dict; found {type(extra).__name__}")
    for k in ID_FIELDS:
        if k in extra and extra[k] is not None:
            try:
                candidate = int(extra[k])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"RewardGate: invalid {k}={extra[k]!r}") from exc
            if str(extra[k]).strip() != str(candidate):
                raise RuntimeError(f"RewardGate: noncanonical ID {k}={extra[k]!r}")
            return candidate
    raise RuntimeError(f"RewardGate: no explicit source ID in extra_info keys={list(extra)}; cannot join by row order")



def load_labels(path):
    import pyarrow.parquet as pq
    src = Path(path).resolve(strict=True)
    pf = pq.ParquetFile(src)
    required = {"extra_info", "reward_model"}
    if not required.issubset(pf.schema_arrow.names):
        raise RuntimeError(f"RewardGate: {src} missing {required - set(pf.schema_arrow.names)}")
    targets = {}
    counters = Counter()
    for batch in pf.iter_batches(batch_size=512, columns=["extra_info", "reward_model"]):
        for item in batch.to_pylist():
            extra, reward = item["extra_info"], item["reward_model"]
            qid = source_id(extra)
            if qid in targets:
                raise RuntimeError(f"RewardGate: duplicate authoritative source ID {qid}: cannot join")
            if not isinstance(reward, dict) or "ground_truth" not in reward:
                raise RuntimeError(f"RewardGate: missing reward_model.ground_truth for question={qid}")
            truth = reward["ground_truth"]
            if truth is None or str(truth).strip()=="":
                raise RuntimeError(f"RewardGate: empty ground_truth for question={qid}")
            targets[qid] = str(truth)
            counters["rows"]+=1
    if not targets:
        raise RuntimeError("RewardGate: empty dataset")
    return targets


@lru_cache(maxsize=2)
def cached_labels(path):
    return load_labels(path)


@lru_cache(maxsize=2)
def cached_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)


def accuracy(text, gt):
    # Exactly the Geo3K accuracy term, NOT combined reward or format bonus.
    from verl.utils.reward_score import geo3k
    value = float(geo3k.acc_reward(text, gt))
    if value not in (0.,1.):
        raise RuntimeError(f"RewardGate: geo3k.acc_reward returned nonbinary {value!r}")
    return int(value)


def score_batch(question_ids, decoded_responses, label_map):
    if len(question_ids)!=len(decoded_responses):
        raise RuntimeError('RewardGate: tensor rows do not match question source IDs')
    flags=[]; wrong_questions=set(); qstats={}
    for idx,(qid,resp) in enumerate(zip(question_ids,decoded_responses)):
        qid=int(qid)
        if qid not in label_map:
            raise RuntimeError(f'RewardGate: mini-batch source ID {qid} absent from train reward_model')
        v=accuracy(resp,label_map[qid])
        flags.append(v)
        if v==0: wrong_questions.add(qid)
        qstats.setdefault(qid,[]).append(v)
    mixed=sum(len(set(v))>1 for v in qstats.values())
    return tuple(flags),wrong_questions,mixed




def _response_token_rows(responses):
    """Decode dense or NestedTensor rows without unsafe tensor.tolist()."""
    import torch
    if not isinstance(responses, torch.Tensor) or responses.ndim != 2:
        raise RuntimeError('RewardGate: responses must be a 2-D torch tensor')
    result=[]
    for i,row in enumerate(responses.unbind(0)):
        if row.ndim!=1 or row.layout!=torch.strided:
            raise RuntimeError(f'RewardGate: invalid 1-D response row {i}')
        ids=row.detach().to('cpu').tolist()
        if not all(isinstance(v,int) and v>=0 for v in ids):
            raise RuntimeError(f'RewardGate: invalid token IDs at row {i}')
        result.append(ids)
    return result


# VIRL2477_GATE_DIRECT_MODULE_V2: no modifications to verl/workers/engine_workers.py.
_GATE_DISPATCH_LOGGED=False


# VIPOD_REWARDGATE_PADDING_V106
# A validated synthetic padding UID is represented as -1 by the existing
# TransferQueue trainer mapping (VIPOD_QA_PADDING_UID_V1). Real IDs are >=0.
# The original scorer remains unchanged for all actual questions.
_e4_score_batch_without_padding_v106 = score_batch

def score_batch(qs, output, labels):
    from collections.abc import Mapping as _V106Mapping
    _n = len(qs)
    _ids = [int(_qid) for _qid in qs]
    if any(_qid < -1 for _qid in _ids):
        raise RuntimeError('RewardGate V106: source ID smaller than -1')
    _padding = [i for i, _qid in enumerate(_ids) if _qid == -1]
    if not _padding:
        return _e4_score_batch_without_padding_v106(qs, output, labels)
    if len(output) != _n:
        raise RuntimeError('RewardGate V106: response/ID lengths differ')
    _real = [i for i, _qid in enumerate(_ids) if _qid != -1]
    if isinstance(labels, _V106Mapping):
        _real_labels = labels   # Reward-model lookup keyed by source_row_idx.
    else:
        if len(labels) != _n:
            raise RuntimeError('RewardGate V106: labels neither mapping nor response-aligned')
        _real_labels = [labels[i] for i in _real]
    if not _real:
        # Ask the existing scorer to construct its own empty collections. This
        # avoids guessing the type of its 'mixed' output on all-pad batches.
        _f, _wrong, _mixed = _e4_score_batch_without_padding_v106([], [], _real_labels)
        if len(_f) != 0:
            raise RuntimeError('RewardGate V106: original scorer returned nonempty flags for empty input')
        return [True] * _n, _wrong, _mixed
    _real_flags, _wrong, _mixed = _e4_score_batch_without_padding_v106(
        [_ids[i] for i in _real], [output[i] for i in _real], _real_labels
    )
    if len(_real_flags) != len(_real):
        raise RuntimeError('RewardGate V106: real flags / real IDs length mismatch')
    if -1 in _wrong:
        raise RuntimeError('RewardGate V106: synthetic padding leaked into wrong questions')
    # The ViPOD hook requires exactly one flag per rollout. A synthetic pad is
    # neutral and can never gate VCP; mark True as a placeholder, NOT accuracy.
    _flags = [True] * _n
    for _i, _correct in zip(_real, _real_flags):
        _flags[_i] = _correct
    print(f'[REWARDGATE PADDING V106] ignored={len(_padding)} real={len(_real)} '
          'pad_flags=neutral_true (NOT accuracy)', flush=True)
    return _flags, _wrong, _mixed

def attach(engine,mini_batch):
    global _GATE_DISPATCH_LOGGED
    requested=os.environ.get('E4_REWARDGATE_MODULE','').strip()
    if requested:
        supported = {'cvpr_exp7.3_virl2477_rewardgate',
                     'cvpr_exp7.4_mix_geo_virl_rewardgate'}
        if requested not in supported:
            raise RuntimeError(f'RewardGate: unsupported module override {requested!r}')
        if os.environ.get('E4_REWARDGATE_ENABLED')!='1':
            return None
        if not _GATE_DISPATCH_LOGGED:
            print('[VIRL2477 GATE DIRECT] source-aware grader active (ViRL or MIX)',flush=True)
            _GATE_DISPATCH_LOGGED=True
        return importlib.import_module(requested).attach(engine,mini_batch)
    if os.environ.get('E4_REWARDGATE_ENABLED')!='1':
        return
    dataset_path=os.environ['E4_REWARDGATE_SOURCE']
    tokenizer_path=os.environ['E4_REWARDGATE_TOKENIZER']
    labels=cached_labels(dataset_path)
    tok=cached_tokenizer(tokenizer_path)
    source=mini_batch.get('e4_source_row_idx')
    responses=mini_batch.get('responses')
    if source is None or responses is None or responses.ndim!=2 or source.ndim!=1:
        raise RuntimeError('RewardGate: need e4_source_row_idx[B], responses[B,L] in actor mini-batch')
    # REWARDGATE_NESTED_TOKEN_ROWS_V1: support nested/nonuniform response lengths.
    token_rows=_response_token_rows(responses)
    if int(source.shape[0])!=len(token_rows):
        raise RuntimeError('RewardGate: source IDs and response rows disagree')
    qs=[int(v) for v in source.detach().cpu().tolist()]
    output=tok.batch_decode(token_rows,skip_special_tokens=True)
    flags, wrong, mixed=score_batch(qs,output,labels)
    engine._e4_rewardgate_acc_flags=flags
    engine._e4_rewardgate_wrong_questions=wrong
    engine._e4_rewardgate_mixed_questions=mixed
    # Never use an existing neutral OPD reward: scoring uses actual decoded
    # rollout and the exact training parquet's reward_model.ground_truth.
    return flags


def audit(args):
    labels=load_labels(args.train)
    pair_rows=[]
    if args.assets:
        with Path(args.assets).open() as f:
            for line in f:
                if not line.strip():continue
                obj=json.loads(line)
                if 'source_row_idx' not in obj:
                    raise RuntimeError('RewardGate: verified assets missing source_row_idx')
                pair_rows.append(int(obj['source_row_idx']))
    distinct=set(pair_rows)
    missing=distinct-set(labels)
    print(f'[REWARDGATE DATA] authoritative_questions={len(labels)} verified_pairs={len(pair_rows)} paired_questions={len(distinct)}')
    print('[REWARDGATE SOURCE] explicit extra_info.source_row_idx ↔ e4_source_row_idx; NOT positional join')
    if missing:
        raise RuntimeError(f'RewardGate: verified pair IDs missing in train parquet: {sorted(missing)[:15]} (total {len(missing)})')
    print('[REWARDGATE SOURCE PASS] every verified question has authoritative ground_truth')
    # Verify local-only tokenizer, official 0/1 accuracy grader, synthetic sanity.
    tokenizer_id=os.environ.get('E4_REWARDGATE_TOKENIZER','')
    if tokenizer_id:
        tok=cached_tokenizer(tokenizer_id)
        encoded=tok(r'<step>Test</step> \boxed{42}',add_special_tokens=False).input_ids
        assert r'\boxed{42}' in tok.decode(encoded,skip_special_tokens=True)
        print('[REWARDGATE TOKENIZER PASS]',tokenizer_id)
    try:
        assert accuracy(r'<step>Done</step> \boxed{42}', '42')==1
        assert accuracy(r'<step>Done</step> \boxed{41}', '42')==0
    except Exception as exc:
        raise RuntimeError('RewardGate: Geo3K official accuracy grader sanity check failed: '+str(exc)) from exc
    print('[REWARDGATE ACC PASS] official Geo3K acc_reward is binary and responds to correct/incorrect answer')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit',action='store_true')
    ap.add_argument('--repo',default='.')
    ap.add_argument('--train',required=True)
    ap.add_argument('--assets')
    args=ap.parse_args()
    if not args.audit:ap.error('use --audit')
    audit(args)
