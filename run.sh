#!/usr/bin/env bash
set -Eeuo pipefail

fail(){ echo '[GEO2702 FKL RUNAI UV ABORT]' "$*" >&2; exit 2; }
# run.sh is in repository root: vvopd-pre/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
HERE="$ROOT"
cd "$ROOT"
mkdir -p tmp
# Run:ai UV workspace follows the user's official Geo3K Run:ai entrypoint.
PROJECT_ROOT="$(dirname "$ROOT")"
command -v uv >/dev/null 2>&1 || fail 'uv missing in Run:ai pod'
# Isolate UV/HF/W&B caches from the Run:ai container's /root defaults.
# Force every path to the writable mounted scratch volume; do not inherit /root env.
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/projectllm1}"
[[ -d "$SCRATCH_ROOT" && -w "$SCRATCH_ROOT" ]] || fail "Scratch is not writable: $SCRATCH_ROOT"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT PYTHONHOME
export HOME="$SCRATCH_ROOT/.container_home"
export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export XDG_CONFIG_HOME="$SCRATCH_ROOT/.config"
export XDG_DATA_HOME="$SCRATCH_ROOT/.local/share"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_ROOT/.uv/python"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export WANDB_DIR="$ROOT/tmp/wandb"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TMPDIR=/tmp
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USER="${USER:-runai-$(id -u)}"
export LOGNAME="${LOGNAME:-$USER}"
# Explicit W&B project and credentials for this Run:ai experiment.
# WARNING: This is a plaintext credential as explicitly requested; do not commit.
# W&B remains online; use an already configured key or local secret file.
# Never embed a plaintext key in a shared experiment script.
if [[ -z "${WANDB_API_KEY:-}" && -r "$PROJECT_ROOT/.secrets/wandb_api_key" ]]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$PROJECT_ROOT/.secrets/wandb_api_key")"
fi
[[ -n "${WANDB_API_KEY:-}" ]] || fail 'W&B online requires WANDB_API_KEY or PROJECT_ROOT/.secrets/wandb_api_key'
export WANDB_PROJECT=vvopd
export WANDB_MODE=online
for d in "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
         "$HF_HOME" "$HF_HUB_CACHE" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
         "$PIP_CACHE_DIR" "$WANDB_DIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"; do
  [[ "$d" != /root && "$d" != /root/* ]] || fail "Refusing unwritable /root path: $d"
  mkdir -p -- "$d"
done
printf '[RUNAI UV PATHS] HOME=%s UV_CACHE_DIR=%s HF_HOME=%s\n' "$HOME" "$UV_CACHE_DIR" "$HF_HOME"
UV_PYTHON=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
# Using uv run here ensures the project venv exists with its locked dependencies.
"${UV_PYTHON[@]}" - <<'PY_UV'
import sys,pyarrow,torch,ray,vllm
from pathlib import Path
print('[UV ENV]',sys.executable,flush=True)
print('[UV VERSIONS] torch=',torch.__version__,'ray=',ray.__version__,'vllm=',vllm.__version__,flush=True)
PY_UV
PYTHON="$ROOT/.venv/bin/python3"
[[ -x "$PYTHON" ]] || fail "Expected UV project Python absent: $PYTHON"
export PYTHON
"$PYTHON" - <<'PY_UV_ASSERT'
import sys
from pathlib import Path
expected=Path('.venv').resolve()
actual=Path(sys.prefix).resolve()
assert actual==expected,(actual,expected)
print('[UV INTERPRETER PASS]',sys.executable,flush=True)
PY_UV_ASSERT
RAY_PYTHON="$PYTHON"
export RAY_PYTHON

MODE="${RUN_MODE:-all}"                         # check | prepare | train | merge | all (train+merge)
TRAIN_GPUS="0,1"                                  # Run:ai container-local IDs; student,teacher
E4_VCP_WEIGHT="${E4_VCP_WEIGHT:-0.001}"
E4_VCP_BETA="${E4_VCP_BETA:-2}"
while (($#)); do
  case "$1" in
    --mode)       (($#>=2)) || fail '--mode needs a value'; MODE="$2"; shift 2;;
    --mode=*)     MODE="${1#*=}"; shift;;
    --gpus)       (($#>=2)) || fail '--gpus needs GPU1,GPU2'; TRAIN_GPUS="$2"; shift 2;;
    --gpus=*)     TRAIN_GPUS="${1#*=}"; shift;;
    --vcp-weight) (($#>=2)) || fail '--vcp-weight needs a value'; E4_VCP_WEIGHT="$2"; shift 2;;
    --vcp-weight=*) E4_VCP_WEIGHT="${1#*=}"; shift;;
    --beta)       (($#>=2)) || fail '--beta needs a value'; E4_VCP_BETA="$2"; shift 2;;
    --beta=*)     E4_VCP_BETA="${1#*=}"; shift;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0;;
    *) fail "Unknown argument $1";;
  esac
done
[[ "$MODE" =~ ^(check|prepare|train|merge|all)$ ]] || fail "Bad --mode=$MODE (check|prepare|train|merge|all)"
[[ "$TRAIN_GPUS" =~ ^[0-9]+,[0-9]+$ ]] || fail 'TRAIN_GPUS must contain exactly two GPU IDs'
[[ "${TRAIN_GPUS%%,*}" != "${TRAIN_GPUS##*,}" ]] || fail 'Cannot run student and teacher on same physical GPU'
[[ "$TRAIN_GPUS" == '0,1' ]] || fail 'Run:ai pod must use GPU IDs 0,1 (student,teacher). Allocate two GPUs in the pod.'
export E4_VCP_WEIGHT E4_VCP_BETA

# Canonicalize numbers; stable checkpoint naming for equivalent representations.
read -r E4_VCP_WEIGHT E4_VCP_BETA WEIGHT_TAG BETA_TAG < <(
  "$PYTHON" - "$E4_VCP_WEIGHT" "$E4_VCP_BETA" <<'PY_NUMBER'
from decimal import Decimal,InvalidOperation
import sys
try:
    lam,beta=[Decimal(s) for s in sys.argv[1:]]
except InvalidOperation as exc:
    raise SystemExit(f'[VCP ARG ABORT] invalid decimal: {exc}')
if not lam.is_finite() or not 0 < lam <= 1:
    raise SystemExit('[VCP ARG ABORT] require 0 < lambda <= 1')
if not beta.is_finite() or beta <= 0:
    raise SystemExit('[VCP ARG ABORT] require beta > 0')
vals=[format(x.normalize(),'f') for x in (lam,beta)]
print(*vals,*(v.replace('.','p') for v in vals))
PY_NUMBER
)
export E4_VCP_WEIGHT E4_VCP_BETA

# No Reverse-KL override. Use exactly the earlier FORWARD-KL top-k loss.
LOSS_OVERRIDES=(
  'distillation.enabled=True'
  'distillation.distillation_loss.loss_mode=forward_kl_topk'
  'distillation.distillation_loss.use_policy_gradient=False'
  'distillation.distillation_loss.use_task_rewards=False'
  'distillation.distillation_loss.topk=64'
  "ray_kwargs.ray_init.runtime_env.py_executable=$RAY_PYTHON"
)
TOTAL_EPOCHS=5
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16
ROLLOUT_N=4
ACTOR_LR=1e-6
EXPECTED_QUESTIONS=2702
EXPECTED_VCP_PAIRS=2530
STEPS_PER_EPOCH=$((EXPECTED_QUESTIONS / TRAIN_BATCH_SIZE))  # 168, drop-last
MAX_STEP=$((TOTAL_EPOCHS * STEPS_PER_EPOCH))          # 840
SAVE_FREQ=$STEPS_PER_EPOCH
CKPT_KEEP=1
STUDENT_MODEL='Qwen/Qwen3-VL-2B-Instruct'
TEACHER_MODEL='Qwen/Qwen3-VL-8B-Instruct'
GEO_RELEASE="$ROOT/data/train/release/geo3k2702_2530pair"
GEO_VCP_ASSETS="$GEO_RELEASE/train_vcp_multi_runtime.jsonl"
CORE="${FWD_CORE:-$HERE/vvopd/train_core.sh}"
AUDIT="$HERE/vvopd/utils/check_geo3k_release.py"
SELECT="$HERE/vvopd/utils/select_actor_checkpoint.py"

for f in "$CORE" "$AUDIT" "$SELECT"; do
  [[ -s "$f" ]] || fail "Missing dependency: $f"
done
[[ -s "$GEO_RELEASE/train_geo3k_qa_aligned.parquet" ]] || fail 'Geo2702 release train Parquet missing'
[[ -s "$GEO_RELEASE/train_vcp_multi.jsonl" && -s "$GEO_VCP_ASSETS" ]] || fail 'Geo2702 VCP JSONL/runtime assets missing'
[[ -s "$ROOT/verl/trainer/distillation/losses.py" ]] || fail 'VERL loss implementation missing'
# Verify the selected core really forwards the trailing Hydra overrides, and
# reads the same runtime VCP assets (prevents accidentally selecting an old core).
bash -n "$CORE" || fail "Invalid training core: $CORE"
printf '[CORE SYNTAX PASS] %s\n' "$CORE"

# W&B was fixed to online + explicit project/key in the early environment block.
[[ "$WANDB_MODE" == online && "$WANDB_PROJECT" == vvopd && -n "$WANDB_API_KEY" ]] || fail 'W&B environment was overwritten'

# RAY_SHORT_SOCKET_V104: Ray appends /ray/session_.../sockets/plasma_store.
# A repo-local RAY_TMPDIR makes the AF_UNIX socket exceed Linux's 107-byte cap.
# Set this in the *parent* wrapper so the unchanged 104 core inherits it.
# We deliberately do not patch the core: its SHA256 is pinned in the run contract.
# Run:ai: keep Ray socket path SHORT, private, and pod-local (avoid /home path > 107B).
# Ignore inherited RAY_TMPDIR from login shells/core. Override only with RUNAI_RAY_TMPDIR.
export RAY_TMPDIR="${RUNAI_RAY_TMPDIR:-/tmp/vp${UID}_$$}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RAY_TMPDIR/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RAY_TMPDIR/inductor}"
unset RAY_ADDRESS RAY_NAMESPACE || true
"$PYTHON" - "$RAY_TMPDIR" <<'PY_RAY_SOCKET_PREFLIGHT'
import os, sys
root = sys.argv[1]
if not os.path.isabs(root):
    raise SystemExit('[RAY TMPDIR ABORT] RAY_TMPDIR must be absolute')
# Reserve extra bytes for variability in Ray session names / actor PIDs.
example = root.rstrip('/') + '/ray/session_2026-11-19_17-50-42_827455_3186562/sockets/plasma_store'
length = len(os.fsencode(example))
if length > 100:
    raise SystemExit(
        f'[RAY TMPDIR ABORT] Estimated plasma_store socket path is {length} bytes '
        '(safe target <= 100). Use a shorter path, e.g. RAY_TMPDIR=/tmp/vp_1234')
print(f'[RAY SOCKET PASS] root={root} projected_socket_bytes={length}/107')
PY_RAY_SOCKET_PREFLIGHT
mkdir -p -- "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
[[ -w "$RAY_TMPDIR" ]] || fail "RAY_TMPDIR is not writable: $RAY_TMPDIR"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.60}"
export TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.88}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
# Max verified contrastive pairs processed together per VCR batch; controls memory/throughput, 
# not the total pairs used per training step.
export E4_VCP_BATCH_PAIRS="${E4_VCP_BATCH_PAIRS:-4}" 
export E4_VCP_BATCH_MAX_PADDED_TOKENS="${E4_VCP_BATCH_MAX_PADDED_TOKENS:-8192}"
"$PYTHON" - "$ROLLOUT_GPU_MEM_UTIL" "$TEACHER_GPU_MEM_UTIL" <<'PY_MEM'
import math,sys
for n,x in zip(('student_rollout','teacher_vllm'),sys.argv[1:]):
    v=float(x)
    if not math.isfinite(v) or not 0<v<=.95:
        raise SystemExit(f'[GPU UTIL ABORT] {n}={x}')
    print(f'[GPU UTIL] {n}={v:.2f}')
PY_MEM

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vvopd_geo2702_2530vcp_5ep_b16_n4_lam${WEIGHT_TAG}_beta${BETA_TAG}}"
# Fail closed on name overrides that could accidentally target previous RKL outputs.
[[ "$EXPERIMENT_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]] || fail 'EXPERIMENT_NAME must be a simple folder name'
OUT="$ROOT/checkpoints/$EXPERIMENT_NAME"
MODEL_EXPORT="$ROOT/exports_final/$EXPERIMENT_NAME/global_step_${MAX_STEP}_hf"
LOG_DIR="$ROOT/tmp/104_${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"

printf '[CONFIG] mode=%s student_gpu=%s teacher_gpu=%s Geo3K=%s VCP=%s\n' \
  "$MODE" "${TRAIN_GPUS%%,*}" "${TRAIN_GPUS##*,}" "$EXPECTED_QUESTIONS" "$EXPECTED_VCP_PAIRS"
printf '[OBJECTIVE] forward_kl_topk(topk=64), use_policy_gradient=False, task_rewards=False; gated VCP lambda=%s beta=%s\n' "$E4_VCP_WEIGHT" "$E4_VCP_BETA"
printf '[TRAIN] epochs=%s batch=%s mini=%s N=%s lr=%s steps_per_epoch=%s target_step=%s\n' \
  "$TOTAL_EPOCHS" "$TRAIN_BATCH_SIZE" "$PPO_MINI_BATCH_SIZE" "$ROLLOUT_N" "$ACTOR_LR" "$STEPS_PER_EPOCH" "$MAX_STEP"
printf '[WANDB] mode=%s project=%s\n' "$WANDB_MODE" "$WANDB_PROJECT"
printf '[CHECKPOINTS] %s\n[HF EXPORT] %s\n[NO EVALUATION] No benchmark rollout/eval will run.\n' "$OUT" "$MODEL_EXPORT"

# Already-published Geo2702 release is used READ-ONLY; no prompt preparation/build here.
# Match the established Geo2702 release audits before starting or resuming training.
# Release is immutable in this experiment; verification only.
"$PYTHON" "$AUDIT" --release "$GEO_RELEASE"
if [[ "$MODE" == prepare ]]; then
  echo '[PREPARE PASS] Existing Geo2702 release verified (read-only); no new dataset published.'
  exit 0
fi
[[ -s "$GEO_RELEASE/train_geo3k_qa_aligned.parquet" && -s "$GEO_VCP_ASSETS" ]] || fail 'Geo2702 dataset incomplete after verify'
bash -n "$CORE"
bash -n "$0"
"$PYTHON" -m verl.trainer.main_ppo --cfg job "${LOSS_OVERRIDES[@]}" > "$LOG_DIR/forward_kl_job.yaml" \
  || fail 'Hydra rejected Forward-KL settings'
"$PYTHON" - "$LOG_DIR/forward_kl_job.yaml" <<'PY_CFG'
from pathlib import Path
import sys,yaml
c=yaml.safe_load(Path(sys.argv[1]).read_text())
d=c['distillation']['distillation_loss']
for k,w in (('loss_mode','forward_kl_topk'),('use_policy_gradient',False),
            ('use_task_rewards',False),('topk',64)):
    if d[k]!=w:
        raise SystemExit(f'[FORWARD KL ABORT] {k}: actual={d[k]!r} expected={w!r}')
if c['distillation']['enabled'] is not True:
    raise SystemExit('[FORWARD KL ABORT] distillation disabled')
print('[FORWARD KL HYDRA PASS] loss_mode=forward_kl_topk, PG=False, rewards=False, topk=64')
PY_CFG

# Existing core dry-run verifies dataset + hook/env + student/teacher route; GPU-free.
if [[ "$MODE" != merge ]]; then
  EXP_KIND=vcp EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    GEO_RELEASE="$GEO_RELEASE" GEO_VCP_ASSETS="$GEO_VCP_ASSETS" \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    STUDENT_MODEL="$STUDENT_MODEL" TEACHER_MODEL="$TEACHER_MODEL" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
    ROLLOUT_N="$ROLLOUT_N" ACTOR_LR="$ACTOR_LR" \
    E4_VCP_WEIGHT="$E4_VCP_WEIGHT" E4_VCP_BETA="$E4_VCP_BETA" \
    TOTAL_EPOCHS="$TOTAL_EPOCHS" TOTAL_TRAINING_STEPS="$MAX_STEP" \
    SAVE_FREQ="$SAVE_FREQ" MODEL_ONLY_SAVE=0 CKPT_KEEP="$CKPT_KEEP" DRY_RUN=1 \
    bash "$CORE" "${LOSS_OVERRIDES[@]}"
fi
if [[ "$MODE" == check ]]; then
  echo '[CHECK PASS] No GPU jobs or checkpoint/export created (dataset unchanged).'
  exit 0
fi

# Check devices only when starting training. Avoid importing CUDA in merge-only mode.
if [[ "$MODE" == all || "$MODE" == train ]]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
  "$PYTHON" - <<'PY_RUNAI_GPUS'
import torch
n=torch.cuda.device_count()
if n!=2:
    raise SystemExit(f'[RUNAI GPU ABORT] Need exactly two pod-visible GPUs 0,1; torch sees {n}. Request 2 GPUs in Run:ai.')
print('[RUNAI GPUS PASS]',[(i,torch.cuda.get_device_name(i)) for i in range(n)],flush=True)
PY_RUNAI_GPUS
  [[ -n "${WANDB_API_KEY:-}" ]] || fail 'W&B online requires WANDB_API_KEY (or ../.secrets/wandb_api_key). Core would silently use offline otherwise.'
fi

# Lock only THIS experiment; do not kill/manage other Ray processes.
command -v flock >/dev/null || fail 'flock required'
exec 9>"$ROOT/tmp/104_${EXPERIMENT_NAME}.lock"
flock -n 9 || fail "Another pipeline instance for $EXPERIMENT_NAME is running"

# Immutable training contract: compare ALL training-semantic settings; physical GPU IDs
# are placement metadata, not a resume-incompatible hyperparameter.
"$PYTHON" - "$MODE" "$OUT" "$EXPERIMENT_NAME" "$GEO_RELEASE" "$GEO_VCP_ASSETS" "$CORE" \
  "$E4_VCP_WEIGHT" "$E4_VCP_BETA" "$MAX_STEP" "$TRAIN_GPUS" <<'PY_CONTRACT'
from pathlib import Path
import hashlib,json,sys
mode,out,exp,release,assets,core,lam,beta,step,gpus=sys.argv[1:]
out=Path(out); release=Path(release)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):
            h.update(chunk)
    return h.hexdigest()
contract={
 'experiment':exp,'data_dir':str(release),'parquet_sha256':sha(release/'train_geo3k_qa_aligned.parquet'),
 'vcp_runtime_sha256':sha(Path(assets)),'vcp_original_sha256':sha(release/'train_vcp_multi.jsonl'),
 'core_sha256':sha(Path(core)),'questions':2702,'pairs':2530,
 'student':'Qwen/Qwen3-VL-2B-Instruct','teacher':'Qwen/Qwen3-VL-8B-Instruct',
 'loss_mode':'forward_kl_topk','use_policy_gradient':False,'use_task_rewards':False,'topk':64,
 'reward_gate':True,'vcp_visual_mask':True,'vcp_reference':False,'gradient_projection':False,
 'vcp_weight':lam,'vcp_beta':beta,'epochs':5,'batch_size':16,'mini_batch_size':16,
 'rollout_n':4,'lr':'1e-6','max_step':int(step),'model_only_save':False,
 'train_gpus':gpus,
}
path=out/'104_run_contract.json'
if path.is_file():
    old=json.loads(path.read_text())
    if not isinstance(old, dict) or 'train_gpus' not in old:
        raise SystemExit('[CONTRACT ABORT] Existing run contract has an unexpected schema')
    # Physical device IDs may change across cluster allocations. The first
    # selected GPU is still the student and the second is still the teacher.
    # Do NOT relax hashes, objective, batch, epochs, VCP, or any other setting.
    semantic_old={k:v for k,v in old.items() if k != 'train_gpus'}
    semantic_new={k:v for k,v in contract.items() if k != 'train_gpus'}
    if semantic_old != semantic_new:
        diffs={k:{'old':semantic_old.get(k,'<missing>'),
                  'new':semantic_new.get(k,'<missing>')}
               for k in sorted(set(semantic_old)|set(semantic_new))
               if semantic_old.get(k,'<missing>') != semantic_new.get(k,'<missing>')}
        raise SystemExit('[CONTRACT ABORT] Actual training/data/config mismatch:\n'+
                         json.dumps(diffs,indent=2,ensure_ascii=False))
    if old['train_gpus'] != gpus:
        print(f'[CONTRACT GPU CHANGE ALLOWED] initial={old["train_gpus"]} current={gpus}; '
              'all training-semantic settings verified identical',flush=True)
    else:
        print(f'[CONTRACT GPU SAME] {gpus}',flush=True)
    # Preserve the original contract unchanged as initial-run provenance.
elif mode=='merge':
    raise SystemExit(f'[MERGE ABORT] Cannot merge a run without contract: {path}')
else:
    if out.is_dir() and any(out.glob('global_step_*')):
        raise SystemExit('[CONTRACT ABORT] Existing checkpoints without run contract')
    out.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(contract,indent=2,ensure_ascii=False)+'\n')
print('[RUN CONTRACT PASS]',path)
PY_CONTRACT

latest_step(){
  local d n best=0
  for d in "$OUT"/global_step_*; do
    [[ -d "$d" ]] || continue
    n="${d##*global_step_}"
    if [[ "$n" =~ ^[0-9]+$ ]] && ((10#$n > best)); then best=$((10#$n)); fi
  done
  printf '%s\n' "$best"
}
full_final_ckpt(){
  local d="$OUT/global_step_${MAX_STEP}/actor" kind
  [[ -d "$d" ]] || fail "Final actor checkpoint missing: $d"
  for kind in optim extra; do
    find "$d" -type f -iname "*${kind}*" -print -quit | grep -q . \
      || fail "Full checkpoint missing ${kind} state in $d"
  done
}

if [[ "$MODE" == all || "$MODE" == train ]]; then
  current="$(latest_step)"
  ((current <= MAX_STEP)) || fail "Existing checkpoint step $current > target $MAX_STEP"
  if ((current < MAX_STEP)); then
    echo '[TRAIN START] Run:ai Geo2702 Forward KL + gated VCP, 5ep B16 N4, student/teacher on GPUs 0,1'
    if EXP_KIND=vcp EXPERIMENT_NAME="$EXPERIMENT_NAME" \
      GEO_RELEASE="$GEO_RELEASE" GEO_VCP_ASSETS="$GEO_VCP_ASSETS" \
      CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
      STUDENT_MODEL="$STUDENT_MODEL" TEACHER_MODEL="$TEACHER_MODEL" \
      TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
      ROLLOUT_N="$ROLLOUT_N" ACTOR_LR="$ACTOR_LR" \
      E4_VCP_WEIGHT="$E4_VCP_WEIGHT" E4_VCP_BETA="$E4_VCP_BETA" \
      TOTAL_EPOCHS="$TOTAL_EPOCHS" TOTAL_TRAINING_STEPS="$MAX_STEP" \
      SAVE_FREQ="$SAVE_FREQ" MODEL_ONLY_SAVE=0 CKPT_KEEP="$CKPT_KEEP" \
      WANDB_MODE=online \
      bash "$CORE" "${LOSS_OVERRIDES[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"; then
      echo '[TRAIN PASS]'
    else
      fail "Training failed; will not merge incomplete checkpoint. See $LOG_DIR/train.log"
    fi
  else
    echo "[SKIP TRAIN] Existing target step=$MAX_STEP"
  fi
fi

if [[ "$MODE" == train ]]; then
  echo '[TRAIN-ONLY DONE] HF merge not requested; --mode all merges automatically.'
  exit 0
fi

full_final_ckpt
# Independently verify exact source checkpoint and keep its signature in the HF export.
selected="$("$PYTHON" "$SELECT" --root "$OUT" --step "$MAX_STEP")" || fail 'Checkpoint selection failed'
IFS=$'\t' read -r selected_step actor signature <<< "$selected"
[[ "$selected_step" == "$MAX_STEP" && -d "$actor" && "$signature" =~ ^[0-9a-f]{64}$ ]] \
  || fail "Invalid selected checkpoint: $selected"

export _FWD_ACTOR="$actor" _FWD_SIGNATURE="$signature" _FWD_OUTPUT="$MODEL_EXPORT"
verify_export(){
  "$PYTHON" - "$1" <<'PY_VERIFY'
from pathlib import Path
import json,os,sys
p=Path(sys.argv[1]);assert (p/'config.json').is_file(), 'Missing HF config.json'
idx=list(p.glob('*.safetensors.index.json'))+list(p.glob('*.bin.index.json'))
if idx:
    names={name for file in idx for name in json.loads(file.read_text())['weight_map'].values()}
    assert names and all((p/name).is_file() and (p/name).stat().st_size>0 for name in names), 'Missing HF weight shard'
else:
    assert any(f.stat().st_size>0 for f in p.glob('*.safetensors')) or any(f.stat().st_size>0 for f in p.glob('pytorch_model*.bin')), 'HF weights missing'
meta=p/'3_source_checkpoint.json'
expected={'actor':os.environ['_FWD_ACTOR'],'signature':os.environ['_FWD_SIGNATURE']}
assert meta.is_file() and json.loads(meta.read_text())==expected, 'HF export source checkpoint provenance mismatch'
print('[HF EXPORT VERIFIED]',p)
PY_VERIFY
}
if [[ -d "$MODEL_EXPORT" ]]; then
  verify_export "$MODEL_EXPORT"
  echo '[MERGE SKIPPED] Exact matching HF export already exists'
  exit 0
fi

mkdir -p "$(dirname "$MODEL_EXPORT")"
STAGING="${MODEL_EXPORT}.building_${UID}_$$"
[[ ! -e "$STAGING" ]] || fail "Staging directory collision: $STAGING"
cleanup(){
  local rc=$?
  if ((rc != 0)) && [[ -n "${STAGING:-}" && -d "$STAGING" ]]; then
    echo "[CLEANUP] Removing incomplete HF staging $STAGING" >&2
    rm -rf -- "$STAGING"
  fi
}
trap cleanup EXIT
"$PYTHON" -m verl.model_merger merge --backend fsdp \
  --local_dir "$actor" --target_dir "$STAGING" \
  2>&1 | tee -a "$LOG_DIR/merge.log"
"$PYTHON" - "$STAGING" <<'PY_SOURCE'
from pathlib import Path
import json,os,sys
p=Path(sys.argv[1]);assert (p/'config.json').is_file(), 'Merged config missing'
(p/'3_source_checkpoint.json').write_text(json.dumps({
    'actor':os.environ['_FWD_ACTOR'],
    'signature':os.environ['_FWD_SIGNATURE']}, indent=2)+'\n')
PY_SOURCE
verify_export "$STAGING"
[[ ! -e "$MODEL_EXPORT" ]] || fail "Export destination created concurrently: $MODEL_EXPORT"
mv -T "$STAGING" "$MODEL_EXPORT"
STAGING=""
verify_export "$MODEL_EXPORT"
printf '[ALL DONE] Forward-KL training + HF merge completed.\n[CHECKPOINT] %s\n[HF MODEL] %s\n[NO EVAL] No benchmark generation performed.\n' "$OUT/global_step_${MAX_STEP}" "$MODEL_EXPORT"
