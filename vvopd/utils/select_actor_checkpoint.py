#!/usr/bin/env python3
"""Read-only, fail-closed structural selector of VERL FSDP actor checkpoints.

Prints: <step> TAB <absolute actor path> TAB <metadata signature>
Only structural completeness is checked; loading the weights is validated by model merger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SHARD = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt\Z")
STEP = re.compile(r"global_step_(\d+)\Z")


def audit(d: Path):
    actor = d / "actor"
    mandatory = [actor / "fsdp_config.json", actor / "huggingface" / "config.json"]  # model-only checkpoint permitted
    for p in mandatory:
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"missing/empty {p.relative_to(d)}")
    for p in mandatory:
        json.loads(p.read_text(encoding="utf8"))
    shard_info = []
    for p in actor.iterdir():
        m = SHARD.fullmatch(p.name)
        if m:
            shard_info.append((int(m.group(1)), int(m.group(2)), p))
    if not shard_info:
        raise ValueError("no actor FSDP model shards")
    worlds = {w for w, _, _ in shard_info}
    if len(worlds) != 1:
        raise ValueError("mixed model world sizes")
    world = next(iter(worlds))
    ranks = [r for _, r, _ in shard_info]
    if world < 1 or len(ranks) != world or set(ranks) != set(range(world)):
        raise ValueError(f"incomplete rank set {ranks}, expected 0..{world-1}")
    if any(p.stat().st_size == 0 for _, _, p in shard_info):
        raise ValueError("zero-length FSDP model shard")
    # Cheap provenance signature: NOT a cryptographic hash of model contents.
    files = mandatory + [p for _, _, p in sorted(shard_info)]
    fingerprint = [
        (str(p.relative_to(d)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in files
    ]
    signature = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
    return str(actor.resolve()), signature, world


def choose(root: Path, requested: str):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint root absent: {root}")
    if requested != "auto":
        n = int(requested)
        dirs = [(n, root / f"global_step_{n}")]
    else:
        dirs = sorted(
            ((int(m.group(1)), p) for p in root.iterdir()
             if p.is_dir() and (m := STEP.fullmatch(p.name))),
            reverse=True,
        )
    if not dirs:
        raise ValueError(f"no global_step_N directories: {root}")
    for step, d in dirs:
        try:
            actor, sig, world = audit(d)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[CKPT SKIP] step {step}: {e}", file=sys.stderr)
            continue
        print(f"[CKPT SELECT] step={step}, world_size={world}, actor={actor}", file=sys.stderr)
        return step, actor, sig
    raise ValueError(f"no structurally complete checkpoint under {root} (step={requested})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--step", default="auto", help="auto or exact nonnegative integer")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print("[SELECTOR SELF-TEST] use dedicated fixture test; no filesystem modifications")
        return
    if a.step != "auto" and not re.fullmatch(r"\d+", a.step):
        ap.error("--step must be auto or a nonnegative integer")
    try:
        step, actor, sig = choose(a.root, a.step)
    except Exception as e:
        raise SystemExit(f"[CKPT ABORT] {e}") from e
    print(f"{step}\t{actor}\t{sig}")


if __name__ == "__main__":
    main()
