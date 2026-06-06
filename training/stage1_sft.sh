#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

timestamp=$(date +"%Y%m%d_%H%M%S")
export MASTER_PORT="${MASTER_PORT:-$(shuf -n 1 -i 20000-60000)}"
echo "Current MASTER_PORT: ${MASTER_PORT}"

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}"
SFT_DATASET="${SFT_DATASET:-/path/to/sft_train.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/training/stage1_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${timestamp}_lora}"

EPOCHS="${EPOCHS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_TYPE="${TRAIN_TYPE:-lora}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_PIXELS="${MAX_PIXELS:-2073600}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"

if [[ ! -e "${SFT_DATASET}" ]]; then
  echo "Missing SFT dataset: ${SFT_DATASET}" >&2
  exit 1
fi

runtime_cache_dir="${TMPDIR:-/tmp}/shuttermuse_swift_cache/${USER:-root}/sft_${timestamp}_${MASTER_PORT}"
mkdir -p "${runtime_cache_dir}/modelscope" "${runtime_cache_dir}/hf_datasets" \
         "${runtime_cache_dir}/packing" "${runtime_cache_dir}/tmp" "${OUTPUT_DIR}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${runtime_cache_dir}/modelscope}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${runtime_cache_dir}/hf_datasets}"
export PACKING_CACHE="${PACKING_CACHE:-${runtime_cache_dir}/packing}"
export TMPDIR="${runtime_cache_dir}/tmp"

echo "Stage-1 SFT dataset: ${SFT_DATASET}"
echo "Stage-1 output dir: ${OUTPUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}" \
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_PATH}" \
  --dataset "${SFT_DATASET}" \
  --load_from_cache_file false \
  --train_type "${TRAIN_TYPE}" \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --attn_impl flash_attn \
  --padding_free true \
  --learning_rate "${LEARNING_RATE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --packing true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.05 \
  --deepspeed zero2 \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --max_pixels "${MAX_PIXELS}"
