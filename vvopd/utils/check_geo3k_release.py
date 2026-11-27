"""Validate the Geo3K training release for VV-OPD.

Check dataset integrity, verified contrastive pair records,
and question-level alignment before training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROMPT_SUFFIX = (
    'Solve the problem step by step.\n'
    'Enclose each reasoning step in <step> </step> tags.\n'
    'Put your final answer in \\boxed{}.'
)
EXPECTED_HIST = {0: 691, 1: 1587, 2: 329, 3: 95}
EXPECTED_ROWS = 2702
EXPECTED_PAIRS = 2530
EXPECTED_PAIRED_QUESTIONS = 2011


def ensure(ok, msg):
    if not ok:
        raise ValueError('[GEO3K RELEASE AUDIT ABORT] ' + str(msg))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    with path.open(encoding='utf-8') as fp:
        for line_no, line in enumerate(fp, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no} invalid JSONL: {exc}') from exc


def audit_release(root):
    import pyarrow.parquet as pq
    root = root.resolve(strict=True)
    ensure(root.name == 'geo3k2702_2530pair',
           f'wrong release directory name: {root.name}')
    parquet = root/'train_geo3k_qa_aligned.parquet'
    pair_file = root/'train_vcp_multi.jsonl'
    index_file = root/'pair_index_multi.jsonl'
    for path in (parquet, pair_file, index_file):
        ensure(path.is_file() and path.stat().st_size > 0, f'missing release source: {path}')
    sha_file = root/'SHA256SUMS'
    if sha_file.is_file():
        for raw in sha_file.read_text(encoding='utf-8').splitlines():
            if not raw.strip():
                continue
            fields = raw.split(maxsplit=1)
            ensure(len(fields) == 2, f'bad SHA256SUMS entry: {raw}')
            wanted, filename = fields
            target = root/filename.lstrip('*')
            ensure(target.is_file() and digest(target) == wanted,
                   f'input release checksum changed: {target}')
        print('[SHA256SUMS PASS]')

    pf = pq.ParquetFile(parquet)
    ensure(pf.metadata.num_rows == EXPECTED_ROWS,
           f'Geo3K row count is {pf.metadata.num_rows}, expected {EXPECTED_ROWS}')
    need = {'prompt', 'images', 'extra_info', 'reward_model',
            'vipod_pair_count', 'vipod_has_vcp_pair'}
    ensure(need.issubset(set(pf.schema_arrow.names)),
           f'Geo3K missing fields: {need-set(pf.schema_arrow.names)}')

    qids = set()
    counts = {}
    hist = Counter()
    rows = 0
    # Don't read image payload bytes just to verify prompt/ID/counts.
    cols = ['prompt','extra_info','vipod_pair_count','vipod_has_vcp_pair']
    first = None
    for batch in pf.iter_batches(batch_size=64, columns=cols):
        for row in batch.to_pylist():
            msgs = row['prompt']
            ensure(isinstance(msgs, list) and len(msgs) == 1 and
                   isinstance(msgs[0], dict) and msgs[0].get('role') == 'user',
                   f'row {rows}: malformed user prompt')
            prompt = msgs[0].get('content')
            ensure(isinstance(prompt,str) and prompt.endswith(PROMPT_SUFFIX),
                   f'row {rows}: not the exact three-line pure step-boxed prompt')
            ensure('<think>' not in prompt and '</think>' not in prompt,
                   f'row {rows}: legacy think tag found')
            ensure(prompt.count('<image>') >= 1, f'row {rows}: missing image token')
            question = prompt[:-len(PROMPT_SUFFIX)].strip()
            ensure(question and '<image>' in question,
                   f'row {rows}: empty or marker-less question')
            info = row['extra_info']
            ensure(isinstance(info, dict) and info.get('source_row_idx') is not None,
                   f'row {rows}: missing extra_info.source_row_idx')
            qid = int(info['source_row_idx'])
            ensure(qid not in qids, f'duplicate question ID {qid}')
            qids.add(qid)
            count = int(row['vipod_pair_count'] or 0)
            ensure(0 <= count <= 3, f'row {rows}: invalid VCP pair count {count}')
            ensure(bool(row['vipod_has_vcp_pair']) == (count > 0),
                   f'row {rows}: VCP row flag mismatch')
            counts[qid] = count
            hist[count] += 1
            rows += 1
            if first is None:
                first = prompt
    ensure(rows == EXPECTED_ROWS and len(qids) == EXPECTED_ROWS,
           'unexpected number of unique training questions')
    ensure(dict(hist) == EXPECTED_HIST,
           f'VCP count histogram differs: {dict(hist)} vs {EXPECTED_HIST}')
    ensure(sum(i * n for i,n in hist.items()) == EXPECTED_PAIRS,
           'Parquet VCP count does not sum to 2530')
    ensure(rows - hist[0] == EXPECTED_PAIRED_QUESTIONS,
           'wrong number of paired questions')

    real = Counter()
    pairids = set()
    paircount = 0
    for rec in load_jsonl(pair_file):
        ensure(isinstance(rec,dict), 'invalid pair JSONL record')
        qid = int(rec['source_row_idx'])
        pid = str(rec['pair_id'])
        ensure(qid in qids and pid and pid not in pairids,
               f'pair {pid} has missing question, empty ID or duplicate ID')
        for field in ('chosen_step','rejected_step'):
            ensure(isinstance(rec.get(field),str) and rec[field].strip(),
                   f'pair {pid} missing {field}')
        pairids.add(pid)
        real[qid] += 1
        paircount += 1
    ensure(paircount == EXPECTED_PAIRS, f'VCP JSONL has {paircount}, expected 2530')
    ensure(all(real[q] == v for q,v in counts.items()),
           'VCP JSONL question alignment differs from Parquet pair counts')
    index_len = sum(1 for _ in load_jsonl(index_file))
    ensure(index_len == EXPECTED_PAIRS,
           f'Geo3K pair_index_multi length {index_len} != 2530')
    print(f'[GEO3K RELEASE AUDIT PASS] rows={rows} VCP={paircount} paired_questions={rows-hist[0]}')
    print('[PAIR HISTOGRAM]', dict(sorted(hist.items())))
    print('[DATA]', parquet)
    print('[VCP]', pair_file)
    print('[PROMPT EXAMPLE]', repr(first[-230:]))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', type=Path,
                    default=Path('data/train/release/geo3k2702_2530pair'))
    a = ap.parse_args()
    audit_release(a.release)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
