#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

timestamp=$(date +"%Y%m%d_%H%M%S")
export MASTER_PORT="${MASTER_PORT:-$(shuf -n 1 -i 20000-60000)}"
echo "Current MASTER_PORT: ${MASTER_PORT}"

MODEL_PATH="${MODEL_PATH:-/path/to/stage1-merged-or-base-checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/training/stage2_grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${timestamp}_grpo}"

# Dataset paths consumed by training/grpo_utils/data_format.py.
export GRPO_DATASET_PATH="${GRPO_DATASET_PATH:-/path/to/grpo_dataset.jsonl}"
export GRPO_COMPOSITION_DATASET_PATH="${GRPO_COMPOSITION_DATASET_PATH:-${GRPO_DATASET_PATH}}"
export GRPO_POSE_DATASET_PATH="${GRPO_POSE_DATASET_PATH:-${GRPO_DATASET_PATH}}"
export SALIENCY_PRECOMPUTE_JSONL="${SALIENCY_PRECOMPUTE_JSONL:-}"
export RATIO_FOLLOWING_IMAGE_PREFIX="${RATIO_FOLLOWING_IMAGE_PREFIX:-}"

DATASET_NAME="${DATASET_NAME:-rl}"
REWARD_FUNCS="${REWARD_FUNCS:-ratio_orm pose_visibility_orm saliency_orm}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 1.0 1.0}"

EPOCHS="${EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TRAIN_TYPE="${TRAIN_TYPE:-lora}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"

if [[ ! -e "${GRPO_DATASET_PATH}" ]]; then
  echo "Missing GRPO dataset: ${GRPO_DATASET_PATH}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

echo "Stage-2 GRPO dataset: ${GRPO_DATASET_PATH}"
echo "Stage-2 output dir: ${OUTPUT_DIR}"
echo "Reward functions: ${REWARD_FUNCS}"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --external_plugins "${SCRIPT_DIR}/grpo_utils/data_format.py" \
                     "${SCRIPT_DIR}/grpo_utils/reward_func.py" \
  --reward_funcs ${REWARD_FUNCS} \
  --reward_weights ${REWARD_WEIGHTS} \
  --epsilon_high 0.28 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host "${VLLM_SERVER_HOST}" \
  --vllm_server_port "${VLLM_SERVER_PORT}" \
  --train_type "${TRAIN_TYPE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --dataset "${DATASET_NAME}" \
  --load_from_cache_file false \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --split_dataset_ratio 0 \
  --learning_rate "${LEARNING_RATE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-1000}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-40}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.01 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --deepspeed zero2 \
  --log_completions true \
  --num_iterations 1 \
  --async_generate false \
  --beta "${BETA:-0.001}" \
  --max_pixels "${MAX_PIXELS}" \
  --report_to "${REPORT_TO:-tensorboard}"
