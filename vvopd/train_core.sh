#!/usr/bin/env bash
set -euo pipefail
# Qwen3-VL-8B teacher -> 2B student: Forward-KL topK64 + reward-gated VCP. Run:ai UV-only.
# Dedicated GPU pools: 1 student GPU + 1 teacher GPU.
# Geo3K2702 pure step-boxed Run:ai UV release ONLY; original 2530 verified pairs; no ViRL.
# One VERL PPOTrainer, forward_kl_topk + Reward-Gated PLAIN VCP with ZERO reference margin, no projection.
# Standalone plain-VCP variant: zero teacher reference; no projection; full checkpoint + resume.
# Usage: invoked by repository-root run.sh

fail() { echo "ERROR: $*" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
PROJECT_ROOT="$(dirname "$REPO_DIR")"
[[ -f "$REPO_DIR/verl/trainer/main_ppo.py" ]] || fail "Not a VERL checkout: $REPO_DIR"
cd "$REPO_DIR"
REWARD_FILE="$REPO_DIR/vvopd/neutral_reward.py"
[[ -f "$REWARD_FILE" ]] || fail "Missing Geo3K neutral OPD reward: $REWARD_FILE"
echo "[GEO3K OPD REWARD] $REWARD_FILE (neutral task rewards; Forward-KL selected by Hydra overrides)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || fail 'CUDA_VISIBLE_DEVICES must contain exactly 2 physical GPU IDs, e.g. 0,1'
[[ "${CUDA_VISIBLE_DEVICES%,*}" != "${CUDA_VISIBLE_DEVICES#*,}" ]] || fail 'GPU IDs must be distinct'
# This run uses DEDICATED teacher pool, NOT the old one-GPU teacher-colocation patch.
export VERL_COLOCATE_TEACHER=0
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-vvopd}"
if [[ -z "${WANDB_API_KEY:-}" && -r "$PROJECT_ROOT/.secrets/wandb_api_key" ]]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$PROJECT_ROOT/.secrets/wandb_api_key")"
fi
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "disabled" ]]; then export WANDB_MODE=offline; fi
export WANDB_DIR="${WANDB_DIR:-$REPO_DIR/tmp/wandb_vvopd}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_DIR/tmp/triton_vvopd_$USER}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$REPO_DIR/tmp/torchinductor_vvopd_$USER}"
export RAY_TMPDIR="${RAY_TMPDIR:-${REPO_DIR}/tmp/ray_vvopd_$USER}"
mkdir -p "$WANDB_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$RAY_TMPDIR"

# UV contract: DO NOT silently use Conda/system Python. 104 wrapper sets PYTHON to
# the absolute interpreter created by `uv run --frozen --all-packages --extra vllm --extra fsdp`.
[[ -n "${PYTHON:-}" && -x "$PYTHON" ]] || fail 'UV Python not supplied; launch using the 104 Run:ai UV wrapper'
[[ "$PYTHON" == "$REPO_DIR/.venv/bin/python" || "$PYTHON" == "$REPO_DIR/.venv/bin/python3" ]] ||     fail "Not the project UV interpreter: $PYTHON"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Baseline: pretrained original teacher, no GRPO checkpoint merge.
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
[[ "$STUDENT_MODEL" == "Qwen/Qwen3-VL-2B-Instruct" ]] || fail "Must start from untrained 2B-Instruct, not Original OPD checkpoint: $STUDENT_MODEL"

# Read Geo3K release DIRECTLY; never copy, rewrite, or merge training records.
GEO_RELEASE="${GEO_RELEASE:-data/train/release/geo3k2702_2530pair}"
if [[ "$GEO_RELEASE" != /* ]]; then GEO_RELEASE="$REPO_DIR/$GEO_RELEASE"; fi
GEO_RELEASE="$(realpath "$GEO_RELEASE")" || fail 'Geo3K release missing'
TRAIN_FILE="$GEO_RELEASE/train_geo3k_qa_aligned.parquet"
# No external validation set or val_placeholder is required for this train-only run.
EXPECTED_TRAIN=2702
ASSETS="${GEO_VCP_ASSETS:-$GEO_RELEASE/train_vcp_multi_runtime.jsonl}"
GEO_EXP6="$GEO_RELEASE/train_vcp_multi.jsonl" # Original verified pair records remain immutable.
[[ -s "$TRAIN_FILE" && -s "$ASSETS" && -s "$GEO_EXP6" ]] || fail "Geo3K release training or VCP assets missing: $GEO_RELEASE"
# Validate the existing Geo2702 release without mutation.
"$PYTHON" "$SCRIPT_DIR/utils/check_geo3k_release.py" --release "$GEO_RELEASE"
# The actively loaded VCP hook: V1; not the separate origopd multi3 variant.
HOOK_COPY="$SCRIPT_DIR/contrastive_distillation.py"
[[ -s "$HOOK_COPY" ]] || fail "Missing contrastive distillation module: $HOOK_COPY"
# Validate actual Python interfaces rather than historical source comments.
"$PYTHON" - "$REPO_DIR" "$HOOK_COPY" "$REWARD_FILE" <<'PY_INTERFACE'
import importlib
import inspect
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
checks = {
    "vvopd.contrastive_distillation": (Path(sys.argv[2]), {"apply_aux_gradients": 1}),
    "vvopd.reward_gate": (root / "vvopd/reward_gate.py", {"attach": 2}),
    "vvopd.edit_token_mask": (root / "vvopd/edit_token_mask.py", {"paired_masks": 4}),
    "vvopd.qwen3vl_input_alignment": (root / "vvopd/qwen3vl_input_alignment.py", {"align_appended_text_step": 3}),
    "vvopd.neutral_reward": (Path(sys.argv[3]), {"compute_score": None}),
}
for name, (expected_path, functions) in checks.items():
    try:
        module = importlib.import_module(name)
        actual_path = Path(module.__file__).resolve()
        if actual_path != expected_path.resolve():
            raise RuntimeError(f"loaded {actual_path}, expected {expected_path.resolve()}")
        for function, nargs in functions.items():
            fn = getattr(module, function, None)
            if not callable(fn):
                raise TypeError(f"missing callable {function}")
            if nargs is not None:
                inspect.signature(fn).bind(*([None] * nargs))
        print(f"[INTERFACE PASS] {name}: {', '.join(functions)}")
    except Exception as exc:
        raise SystemExit(f"[INTERFACE ERROR] {name}: {type(exc).__name__}: {exc}") from exc
for rel in ("verl/workers/engine_workers.py", "verl/workers/engine/fsdp/transformer_impl.py"):
    path = root / rel
    try:
        compile(path.read_bytes(), str(path), "exec")
    except Exception as exc:
        raise SystemExit(f"[ENGINE SOURCE ERROR] {path}: {exc}") from exc
print("[ENGINE SOURCE PASS] Worker and FSDP engine compile; runtime integration requires a training smoke test")
PY_INTERFACE
export E4_VCP_ENABLED=1 E4_VCP_QUESTION_ALIGNED=1
export E4_VCP_EMA_ENABLED=0
export E4_REWARDGATE_ENABLED=1
export E4_REWARDGATE_SOURCE="$TRAIN_FILE"
export E4_REWARDGATE_TOKENIZER="${E4_REWARDGATE_TOKENIZER:-$STUDENT_MODEL}"

export E4_VCP_HOOK_FILE="$HOOK_COPY" E4_VCP_FAST_HOOK_FILE="$HOOK_COPY"
export E4_VCP_ASSETS="$ASSETS" E4_VCP_USE_DISK_CACHE=0
export E4_VCP_BACKWARD_MODE="${E4_VCP_BACKWARD_MODE:-batched}"
export E4_VCP_BATCH_PAIRS="${E4_VCP_BATCH_PAIRS:-4}"
export E4_VCP_BATCH_MAX_PADDED_TOKENS="${E4_VCP_BATCH_MAX_PADDED_TOKENS:-8192}"
export E4_VCP_BATCH_VERIFY_FIRST="${E4_VCP_BATCH_VERIFY_FIRST:-1}"
export E4_VCP_WEIGHT="${E4_VCP_WEIGHT:-0.001}"
export E4_VCP_EMA_ENABLED=0
# Statically weighted NO projection. No EMA state, no Step38 restore.
[[ "$E4_VCP_EMA_ENABLED" == 0 ]] || fail 'ExpB NoProj must disable EMA'
"$PYTHON" - "$E4_VCP_WEIGHT" <<'PY_WEIGHT'
import math,sys
x=float(sys.argv[1]);assert math.isfinite(x) and 0<x<=1, x
print('[VCP STATIC WEIGHT]',x)
PY_WEIGHT
export E4_VCP_BETA="${E4_VCP_BETA:-2.0}"
# Absolutely no Sparse4 and no post-selection pair filter.
# One verified pair per distinct matched question in each mini-batch;
# if a question has multiple pairs, the QA-fast hook cycles them per update.
export E4_VCP_MAX_MATCHED_PER_UPDATE=0
export E4_VCP_MAX_STEP_TOKENS="${E4_VCP_MAX_STEP_TOKENS:-2048}"
export E4_VCP_MAX_SEQ_TOKENS="${E4_VCP_MAX_SEQ_TOKENS:-8192}"
# The V4 direct Hook inherits the V3 all-diff-token mask; it separates it from
# nonminimal-edit quality gating. Mask ON; quality filter OFF.
export E4_VCP_VISUAL_MASK=1
export E4_VCP_FILTER_NONMINIMAL=0
export E4_VCP_PRIORITY_ENABLED=0 E4_KLFIRST_ENABLED=0  # NO gradient projection
export E4_GRAD_DIAG_ENABLED=0
# Deliberately DO NOT set/require reference cache or scoring settings.
# Teacher 8B remains solely the original OPD KL target.
unset E4_VCP_REF_FILE E4_VCP_REF_MODE E4_VCP_REF_ALPHA E4_VCP_REF_BASE_MARGIN E4_VCP_REF_MAX_ERROR E4_VCP_REF_SOURCE_HOOK E4_VCP_REF_TEACHER_MODEL E4_VCP_REF_STUDENT_PROCESSOR || true
export VERL_COLOCATE_TEACHER=0

# Geo3K-only experiment modes: core reads the release 2702-row Parquet.
# In KL-only mode disable *both* VCP and reward gating BEFORE constructing VERL/Ray workers.
case "${EXP_KIND:-vcp}" in
  vcp)
    [[ "${E4_VCP_ENABLED}" == 1 && "${E4_REWARDGATE_ENABLED}" == 1 ]] || fail 'VCP mode lost RewardGate/VCP flags'
    echo '[OBJECTIVE AUDIT] REWARD-GATED KL + STATIC PLAIN VCP; NO PROJECTION; NO TEACHER REFERENCE'
    ;;
  kl)
    export E4_VCP_ENABLED=0 E4_VCP_QUESTION_ALIGNED=0 E4_REWARDGATE_ENABLED=0
    export E4_VCP_PRIORITY_ENABLED=0 E4_KLFIRST_ENABLED=0 E4_GRAD_DIAG_ENABLED=0
    [[ "${E4_VCP_ENABLED}" == 0 && "${E4_REWARDGATE_ENABLED}" == 0 ]] || fail 'KL-only flags not disabled'
    grep -q 'E4_VCP_ENABLED' "$REPO_DIR/verl/workers/engine/fsdp/transformer_impl.py" || fail 'Cannot verify that VERL engine honors E4_VCP_ENABLED flag; refuse KL baseline'
    echo '[OBJECTIVE AUDIT] KL-ONLY mode not used by this experiment'
    ;;
  *) fail "EXP_KIND must be vcp or kl; got ${EXP_KIND}" ;;
esac

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.60}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.88}"
DISTILLATION_TOPK="${DISTILLATION_TOPK:-64}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-0}" # e.g. 10 for smoke test
SAVE_FREQ="${SAVE_FREQ:-42}"
TEST_FREQ="${TEST_FREQ:--1}"  # -1 disables scheduled AND last-step validation in veRL.
[[ "$TEST_FREQ" == -1 ]] || fail "Validation is disabled in this train-only run; require TEST_FREQ=-1"
MODEL_ONLY_SAVE="${MODEL_ONLY_SAVE:-0}"
CKPT_KEEP="${CKPT_KEEP:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
DRY_RUN="${DRY_RUN:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vvopd_geo2702_2530vcp_stepboxed_5ep_n4}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/$EXPERIMENT_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/tmp/3_logs_$EXPERIMENT_NAME}"

[[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$PPO_MINI_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$ROLLOUT_N" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid batch sizes'
(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 )) || fail 'TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE'
(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 <= MAX_MODEL_LEN )) || fail 'MAX_MODEL_LEN too short for selected prompt/response budget'
[[ "$TOTAL_TRAINING_STEPS" =~ ^[0-9]+$ ]] || fail 'Invalid TOTAL_TRAINING_STEPS'
[[ "$MODEL_ONLY_SAVE" == 0 || "$MODEL_ONLY_SAVE" == 1 ]] || fail 'MODEL_ONLY_SAVE must be 0 or 1'
for numeric in SAVE_FREQ TOTAL_EPOCHS MIN_FREE_GB; do
    [[ "${!numeric}" =~ ^[1-9][0-9]*$ ]] || fail "Invalid $numeric=${!numeric}"
done
# Existing release is read-only. Validate only the training Parquet.
[[ -s "$TRAIN_FILE" ]] || fail 'Geo3K release training parquet missing'
"$PYTHON" - "$TRAIN_FILE" "$EXPECTED_TRAIN" <<'PY_AUDIT'
import sys
import pyarrow.parquet as pq
train = pq.ParquetFile(sys.argv[1])
expected_train = int(sys.argv[2])
assert train.metadata.num_rows == expected_train, (train.metadata.num_rows, expected_train)
assert {'prompt', 'images', 'extra_info', 'reward_model'}.issubset(train.schema_arrow.names)
print(f'[DATA AUDIT PASS] train={train.metadata.num_rows}; no external validation file')
PY_AUDIT
[[ "$CKPT_KEEP" =~ ^[1-9][0-9]*$ ]] || fail 'CKPT_KEEP must be >=1'

cd "$REPO_DIR"

# Accept Hugging Face model IDs or existing local model directories.
if [[ "$TEACHER_MODEL" == /* || "$TEACHER_MODEL" == ./* ]]; then
  [[ -s "$TEACHER_MODEL/config.json" ]] || fail "Local teacher config missing: $TEACHER_MODEL"
fi

printf '\n[ORIGINAL 8B TEACHER -> 2B | SINGLE TEACHER OPD BASELINE]\n'
printf 'teacher=%s\nstudent=%s\n' "$TEACHER_MODEL" "$STUDENT_MODEL"
printf 'visible GPUs=%s (student pool=1, teacher pool=1, teacher TP=1)\n' "$CUDA_VISIBLE_DEVICES"
printf 'train=%s (Geo3K2702 ONLY, 2530 VCP pairs); external val file=NONE\n' "$TRAIN_FILE"
printf 'Geo3K release verified VCP=%s\n' "$GEO_EXP6"
printf 'B=%s mini=%s N=%s lr=%s epochs=%s pilot_steps=%s\n' "$TRAIN_BATCH_SIZE" "$PPO_MINI_BATCH_SIZE" "$ROLLOUT_N" "$ACTOR_LR" "$TOTAL_EPOCHS" "$TOTAL_TRAINING_STEPS"
printf 'core_loss=forward_kl_topk (confirmed by parent Hydra overrides) | mode=%s | VCP_flag=%s | rewardgate=%s | topk=%s model_only=%s\n' "${EXP_KIND:-vcp}" "${E4_VCP_ENABLED}" "${E4_REWARDGATE_ENABLED}" "$DISTILLATION_TOPK" "$MODEL_ONLY_SAVE"
printf 'experiment=%s output=%s\n' "$EXPERIMENT_NAME" "$OUTPUT_DIR"
if [[ "$EXPERIMENT_NAME" == *visualmask_fullresume_v3* || "$EXPERIMENT_NAME" == *klfirst_allpairs_fullresume_v2* ]]; then
  fail 'Do not reuse old V2/V3 checkpoint experiment names'
fi
# No Teacher Reference cache is read, scored or verified in plain-VCP mode.
echo "[VCP RUNTIME ASSETS] $ASSETS"
echo '[PLAIN VCP REF AUDIT] teacher_reference=OFF | margin=0 | no cache required'

if [[ "$DRY_RUN" != 1 ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi missing; request 2 GPUs first"
fi
# Verify required keys exist in THIS local checkout; do not hide incompatible configs.
CONF="$("$PYTHON" -m verl.trainer.main_ppo --cfg job)" || fail 'Could not read VERL Hydra config (--cfg job)'
if ! printf '%s\n' "$CONF" | "$PYTHON" -c '
import sys,yaml
# The YAML config may contain extra Hydra messages? --cfg job outputs just YAML.
cfg=yaml.safe_load(sys.stdin.read())
d=cfg.get("distillation",{})
assert "enabled" in d and "teacher_models" in d and "distillation_loss" in d, "OPD config missing"
assert "teacher_model" in d["teacher_models"], "Default teacher_model not found"
assert "total_training_steps" in cfg.get("trainer",{}), "trainer.total_training_steps missing"
' ; then fail 'This checkout does not expose the expected OPD Hydra keys'; fi

# Auto-resume requires model + optimizer + extra, not model-only checkpoints.
printf '%s\n' "$CONF" | "$PYTHON" -c '
import sys,yaml
cfg=yaml.safe_load(sys.stdin.read())
c=cfg["actor_rollout_ref"]["actor"].get("checkpoint",{})
assert "save_contents" in c and "load_contents" in c, "Full resume unsupported"
assert "resume_mode" in cfg["trainer"]
' || fail 'The local VERL checkout does not support full-checkpoint resume.'
if [[ "$MODEL_ONLY_SAVE" == 1 ]];then
  CKPT_ARGS=('actor_rollout_ref.actor.checkpoint.save_contents=[model]' 'actor_rollout_ref.actor.checkpoint.load_contents=[model]')
  RESUME_MODE=disable
else
  CKPT_ARGS=('actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' 'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]')
  RESUME_MODE=auto
fi
if [[ "$TOTAL_TRAINING_STEPS" -gt 0 ]];then
  CKPT_ARGS+=( "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" )
fi

if [[ "$DRY_RUN" != 1 ]]; then
if [[ "$MODEL_ONLY_SAVE" == 1 ]] && compgen -G "$OUTPUT_DIR/global_step_*" > /dev/null;then
  fail 'Model-only experiment has previous checkpoints; use a new experiment name (no optimizer resume)'
fi
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
FREE_KIB="$(df -Pk "$OUTPUT_DIR" | awk 'END{print $4}')"
if (( FREE_KIB < MIN_FREE_GB*1024*1024 ));then
    df -h "$OUTPUT_DIR" >&2
    fail "Insufficient free disk for student checkpoint (${MIN_FREE_GB} GiB minimum)."
fi
if [[ "$MODEL_ONLY_SAVE" == 0 && "$FREE_KIB" -lt "$((60*1024*1024))" ]]; then
   fail 'Full optimizer checkpoints for 2B can require tens of GiB; change destination or use MODEL_ONLY_SAVE=1.'
fi
RUN_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$RUN_LOG") 2>&1
fi

# Remove inherited CVGA settings: this is an independent single-teacher baseline.
for cvga_var in "${!CVGA_AP_@}";do unset "$cvga_var";done
unset CVGA_CF_MODE CVGA_CF_ARBITRATION_MODE || true

# In this veRL PPO fork, val_dataset is initialized even when evaluation is disabled.
# Point that inert loader to one sample from the existing train Parquet (no extra file).
# TEST_FREQ=-1 prevents all validation runs, including at the final step.
PPO_ARGS=(
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['$TRAIN_FILE']" \
  data.val_files="['$TRAIN_FILE']" \
  data.val_max_samples=1 \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.image_key=images \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  reward.reward_model.enable=False \
  reward.custom_reward_function.path="$REWARD_FILE" \
  reward.custom_reward_function.name=compute_score \
  trainer.balance_batch=True \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.max_actor_ckpt_to_keep="$CKPT_KEEP" \
  distillation.enabled=True \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.data_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEM_UTIL" \
  distillation.teacher_models.teacher_model.inference.max_model_len="$MAX_MODEL_LEN" \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk="$DISTILLATION_TOPK" \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  "${CKPT_ARGS[@]}" \
  "$@"
)

if [[ "$DRY_RUN" == 1 ]]; then
  "$PYTHON" -m verl.trainer.main_ppo --cfg job "${PPO_ARGS[@]}" | "$PYTHON" -c '
import sys, yaml
from pathlib import Path
c = yaml.safe_load(sys.stdin.read())
d = c["distillation"]["distillation_loss"]
checks = {
    "distillation.enabled": (c["distillation"]["enabled"], True),
    "loss_mode": (d["loss_mode"], "forward_kl_topk"),
    "use_policy_gradient": (d["use_policy_gradient"], False),
    "use_task_rewards": (d["use_task_rewards"], False),
    "topk": (d["topk"], int(sys.argv[1])),
    "train_files": ([str(Path(p).resolve()) for p in c["data"]["train_files"]], [str(Path(sys.argv[2]).resolve())]),
    "train_batch_size": (c["data"]["train_batch_size"], int(sys.argv[3])),
    "ppo_mini_batch_size": (c["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"], int(sys.argv[4])),
    "rollout.n": (c["actor_rollout_ref"]["rollout"]["n"], int(sys.argv[5])),
    "trainer.val_before_train": (c["trainer"]["val_before_train"], False),
    "trainer.test_freq": (c["trainer"]["test_freq"], -1),
    "trainer.total_epochs": (c["trainer"]["total_epochs"], int(sys.argv[6])),
}
if int(sys.argv[7]) > 0:
    checks["trainer.total_training_steps"] = (c["trainer"]["total_training_steps"], int(sys.argv[7]))
for key, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"[CONFIG ERROR] {key}: actual={actual!r}; expected={expected!r}")
print("[EFFECTIVE CONFIG PASS] Full training arguments validated")
' "$DISTILLATION_TOPK" "$TRAIN_FILE" "$TRAIN_BATCH_SIZE" "$PPO_MINI_BATCH_SIZE" "$ROLLOUT_N" "$TOTAL_EPOCHS" "$TOTAL_TRAINING_STEPS" || fail 'Full training configuration rejected'
  echo '[DRY_RUN PASS] no GPU training; no external validation file'
  exit 0
fi
"$PYTHON" -m verl.trainer.main_ppo "${PPO_ARGS[@]}"
