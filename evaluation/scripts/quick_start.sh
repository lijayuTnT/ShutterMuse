#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVALUATION_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${EVALUATION_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/quick_start}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspacedir/lijiayu/checkpoints/Merged_lora/Qwen3-VL-8B/20260518_151450_lora_lr_1e-4_rank_32_alpha_32_bs_64_e_10_compostion_100K_pose_30K_without_kd_4500}"
LORA_PATH="${LORA_PATH:-/mnt/workspacedir/lijiayu/checkpoints/grpo/Qwen3-VL-8B/20260521_010510_grpo_drops/v0-20260521-010620/checkpoint-6000}"
PHOTOGRAPHER_PROMPT="${PHOTOGRAPHER_PROMPT:-}"
SUBJECT_PROMPT="${SUBJECT_PROMPT:-}"
SIDE=""
IMAGE_PATH=""

usage() {
    cat <<'EOF'
Usage: evaluation/scripts/quick_start.sh --side <photographer|subject> --image <image_path> [options]

Options:
  --side                 Which single-image inference to run: photographer or subject.
  --image                Input image path.
  --output-dir           Output directory. Defaults to outputs/quick_start under the project.
  --model-path           Base/merged model path. Defaults to MODEL_PATH env or the repo default.
  --lora-path            LoRA checkpoint path. Defaults to LORA_PATH env or the repo default.
  --photographer-prompt  Override photographer-side prompt. If unset, infer_single_qwen_lora.py uses its default.
  --subject-prompt       Override subject-side prompt. If unset, infer_single_qwen_lora.py uses its default.
  -h, --help             Show this help.

Environment variables:
  PYTHON_BIN, MODEL_PATH, LORA_PATH, OUTPUT_DIR, PHOTOGRAPHER_PROMPT, SUBJECT_PROMPT
  QUICK_START_GPUS, QUICK_START_GPUS_PER_PROCESS, QUICK_START_DTYPE, QUICK_START_MAX_NEW_TOKENS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --side)
            SIDE="$2"
            shift 2
            ;;
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --photographer-prompt)
            PHOTOGRAPHER_PROMPT="$2"
            shift 2
            ;;
        --subject-prompt)
            SUBJECT_PROMPT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing file: $1" >&2
        exit 1
    fi
}

run_single_image() {
    local side="$1"
    local side_output_dir="${OUTPUT_DIR}/${side}-side"
    mkdir -p "${side_output_dir}"
    local prompt="${PHOTOGRAPHER_PROMPT}"
    local instruction_args=()
    local max_new_tokens="${QUICK_START_MAX_NEW_TOKENS:-512}"
    if [[ "${side}" == "subject" ]]; then
        prompt="${SUBJECT_PROMPT}"
        max_new_tokens="${QUICK_START_MAX_NEW_TOKENS:-10240}"
    fi
    if [[ -n "${prompt}" ]]; then
        instruction_args=(--instruction "${prompt}")
    fi
    echo "[quick-start][${side}] image=${IMAGE_PATH} output=${side_output_dir}"
    "${PYTHON_BIN}" "${EVALUATION_DIR}/photographer-side/infer_single_qwen_lora.py" \
        --side "${side}" \
        --model_path "${MODEL_PATH}" \
        --lora_path "${LORA_PATH}" \
        --image "${IMAGE_PATH}" \
        "${instruction_args[@]}" \
        --output_dir "${side_output_dir}" \
        --max_new_tokens "${max_new_tokens}" \
        --device "${QUICK_START_DEVICE:-cuda}"
}

if [[ -z "${SIDE}" || -z "${IMAGE_PATH}" ]]; then
    usage >&2
    exit 2
fi
require_file "${IMAGE_PATH}"

case "${SIDE}" in
    photographer|photographer-side)
        run_single_image photographer
        ;;
    subject|subject-side)
        run_single_image subject
        ;;
    *)
        echo "Unknown side: ${SIDE}" >&2
        usage >&2
        exit 2
        ;;
esac

echo "Quick Start outputs saved to: ${OUTPUT_DIR}"
