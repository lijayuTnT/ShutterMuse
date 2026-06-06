#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVALUATION_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${EVALUATION_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/evaluation}"

PHOTOGRAPHER_EVAL_DIR="${EVALUATION_DIR}/photographer-side"
PHOTOGRAPHER_BENCHMARK_DIR="${PROJECT_ROOT}/Benchmark/photographer-side/composition_benchmark"
PHOTOGRAPHER_DATA_PATH="${PHOTOGRAPHER_BENCHMARK_DIR}/meta_new.json"
PHOTOGRAPHER_IMAGE_ROOT="${PHOTOGRAPHER_BENCHMARK_DIR}/original_composition"
PHOTOGRAPHER_PROMPT="${PHOTOGRAPHER_PROMPT:-Recommend a composition.}"
PHOTOGRAPHER_MODEL_PATH="${PHOTOGRAPHER_MODEL_PATH:-/mnt/workspacedir/lijiayu/checkpoints/Merged_lora/Qwen3-VL-8B/20260518_151450_lora_lr_1e-4_rank_32_alpha_32_bs_64_e_10_compostion_100K_pose_30K_without_kd_4500}"
PHOTOGRAPHER_LORA_TEMPLATE="${PHOTOGRAPHER_LORA_TEMPLATE:-/mnt/workspacedir/lijiayu/checkpoints/grpo/Qwen3-VL-8B/20260521_010510_grpo_drops/v0-20260521-010620/checkpoint-6000}"
PHOTOGRAPHER_STEPS="${PHOTOGRAPHER_STEPS:-6000}"
PHOTOGRAPHER_GPUS="${PHOTOGRAPHER_GPUS:-0,1,2,3,4,5,6,7}"
PHOTOGRAPHER_GPUS_PER_PROCESS="${PHOTOGRAPHER_GPUS_PER_PROCESS:-1}"
PHOTOGRAPHER_BASELINE_MODELS="${PHOTOGRAPHER_BASELINE_MODELS:-gemini-3-pro-native}"
PHOTOGRAPHER_BASELINE_MODEL_PATH="${PHOTOGRAPHER_BASELINE_MODEL_PATH:-/mnt/workspacedir/fangyixiao/models/huggingface/Venus-Q-Stage2}"

SUBJECT_EVAL_DIR="${EVALUATION_DIR}/subject-side"
SUBJECT_BENCHMARK_DIR="${PROJECT_ROOT}/Benchmark/subject-side"
SUBJECT_IMAGE_DIR="${SUBJECT_BENCHMARK_DIR}/paper-benchmark"
SUBJECT_GT_DIR="${SUBJECT_BENCHMARK_DIR}/paper-benchmark-gt"
SUBJECT_PROMPT="${SUBJECT_PROMPT:-You are a portrait photography pose analysis expert. Based on the image, recommend a human pose and provide the relative coordinates of 17 human keypoints and whether each keypoint is visible in the image in JSON format. The 17 keypoints are, in order: nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle.}"
SUBJECT_MODEL_PATH="${SUBJECT_MODEL_PATH:-/mnt/workspacedir/lijiayu/checkpoints/Merged_lora/Qwen3-VL-8B/20260518_151450_lora_lr_1e-4_rank_32_alpha_32_bs_64_e_10_compostion_100K_pose_30K_without_kd_4500}"
SUBJECT_LORA_TEMPLATE="${SUBJECT_LORA_TEMPLATE:-/mnt/workspacedir/lijiayu/checkpoints/grpo/Qwen3-VL-8B/20260521_010510_grpo_drops/v0-20260521-010620/checkpoint-6000}"
SUBJECT_STEPS="${SUBJECT_STEPS:-6000}"
SUBJECT_JUDGERS="${SUBJECT_JUDGERS:-gemini}"
SUBJECT_GPUS="${SUBJECT_GPUS:-0,1,2,3,4,5,6,7}"
SUBJECT_GPUS_PER_PROCESS="${SUBJECT_GPUS_PER_PROCESS:-1}"
SUBJECT_MAX_NEW_TOKENS="${SUBJECT_MAX_NEW_TOKENS:-10240}"
SUBJECT_BASELINE_BACKENDS="${SUBJECT_BASELINE_BACKENDS:-gemini gpt}"
SUBJECT_BASELINE_LIMIT="${SUBJECT_BASELINE_LIMIT:-20}"
SUBJECT_BASELINE_MAX_WORKERS="${SUBJECT_BASELINE_MAX_WORKERS:-10}"
SUBJECT_BASELINE_MAX_RETRIES="${SUBJECT_BASELINE_MAX_RETRIES:-5}"
SUBJECT_YOLO_CHECKPOINT="${SUBJECT_YOLO_CHECKPOINT:-/mnt/workspacedir/lijiayu/yolo26x-pose.pt}"
SUBJECT_YOLO_GPUS="${SUBJECT_YOLO_GPUS:-0,1,2,3}"
SUBJECT_YOLO_BATCH_SIZE="${SUBJECT_YOLO_BATCH_SIZE:-16}"
SUBJECT_YOLO_NUM_WORKERS="${SUBJECT_YOLO_NUM_WORKERS:-4}"

usage() {
    cat <<'EOF'
Usage: evaluation/scripts/run_unified_evaluation.sh <target>

Targets:
  photographer-model      Run photographer-side model/lora benchmark.
  photographer-baseline   Run photographer-side API/local baseline benchmark.
  subject                 Run subject-side pose benchmark.
  subject-baseline        Run subject-side image-edit baseline benchmark.
  all                     Run photographer-model, photographer-baseline, subject, and subject-baseline.

Common environment variables:
  PYTHON_BIN, OUTPUT_ROOT, GEMINI_API_KEY, QWEN_API_KEY, GPT_API_KEY
  PHOTOGRAPHER_STEPS, PHOTOGRAPHER_BASELINE_MODELS, SUBJECT_STEPS, SUBJECT_JUDGERS
  PHOTOGRAPHER_MODEL_PATH, PHOTOGRAPHER_LORA_TEMPLATE
  SUBJECT_MODEL_PATH, SUBJECT_LORA_TEMPLATE, SUBJECT_YOLO_CHECKPOINT
EOF
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing file: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Missing directory: $1" >&2
        exit 1
    fi
}

run_photographer_vlm_scoring() {
    local output_dir="$1"
    local run_policy="${2:-resume}"
    local run_flag="--resume"
    if [[ "${run_policy}" == "overwrite" ]]; then
        run_flag="--overwrite"
    fi
    "${PYTHON_BIN}" "${PHOTOGRAPHER_EVAL_DIR}/evaluate_vlm_benchmark.py" \
        --backend gemini \
        --annotation-json "${PHOTOGRAPHER_DATA_PATH}" \
        --eval-json "${output_dir}/eval_records.json" \
        --image-root "${PHOTOGRAPHER_IMAGE_ROOT}" \
        --output-jsonl "${output_dir}/vlm_scores_details.jsonl" \
        --summary-json "${output_dir}/vlm_scores_summary.json" \
        --vis-dir "${output_dir}/vis_vlm_scores" \
        --max-workers "${VLM_MAX_WORKERS:-15}" \
        --max-retries "${VLM_MAX_RETRIES:-3}" \
        --gemini-api-key "${GEMINI_API_KEY:-}" \
        --qwen-api-key "${QWEN_API_KEY:-}" \
        "${run_flag}" \
        --max-tokens "${VLM_MAX_TOKENS:-5120}"
}

run_photographer_model() {
    require_file "${PHOTOGRAPHER_DATA_PATH}"
    require_dir "${PHOTOGRAPHER_IMAGE_ROOT}"
    read -r -a steps <<< "${PHOTOGRAPHER_STEPS}"
    for step in "${steps[@]}"; do
        local lora_path="${PHOTOGRAPHER_LORA_TEMPLATE//\{step\}/${step}}"
        local exp_name
        exp_name=$(basename "$(dirname "$(dirname "${lora_path}")")")
        local output_dir="${OUTPUT_ROOT}/photographer-side/model/${exp_name}_step_${step}"
        echo "[photographer-model] step=${step} output=${output_dir}"
        "${PYTHON_BIN}" "${PHOTOGRAPHER_EVAL_DIR}/evaluate_benchmark.py" \
            --data_format eval \
            --annotation_json "${PHOTOGRAPHER_DATA_PATH}" \
            --image_root "${PHOTOGRAPHER_IMAGE_ROOT}" \
            --model_path "${PHOTOGRAPHER_MODEL_PATH}" \
            --output_dir "${output_dir}" \
            --prompt "${PHOTOGRAPHER_PROMPT}" \
            --result_json "${output_dir}/eval_records.json" \
            --gpus "${PHOTOGRAPHER_GPUS}" \
            --gpus_per_process "${PHOTOGRAPHER_GPUS_PER_PROCESS}" \
            --overwrite \
            --lora_path "${lora_path}"
        run_photographer_vlm_scoring "${output_dir}" resume
    done
}

run_photographer_baseline() {
    require_file "${PHOTOGRAPHER_DATA_PATH}"
    require_dir "${PHOTOGRAPHER_IMAGE_ROOT}"
    read -r -a model_types <<< "${PHOTOGRAPHER_BASELINE_MODELS}"
    for model_type in "${model_types[@]}"; do
        local output_dir="${OUTPUT_ROOT}/photographer-side/baseline/${model_type}"
        local gpt_model="${model_type/gpt/gpt-}"
        local gemini_model="${model_type}"
        if [[ "${model_type}" == "gemini-3-flash" ]]; then
            gemini_model="gemini-3-flash-image-native"
        fi
        echo "[photographer-baseline] model=${model_type} output=${output_dir}"
        "${PYTHON_BIN}" "${PHOTOGRAPHER_EVAL_DIR}/evaluate_benchmark.py" \
            --eval_mode baseline \
            --annotation_json "${PHOTOGRAPHER_DATA_PATH}" \
            --image_root "${PHOTOGRAPHER_IMAGE_ROOT}" \
            --model_path "${PHOTOGRAPHER_BASELINE_MODEL_PATH}" \
            --gpus "${PHOTOGRAPHER_GPUS}" \
            --gpus_per_process "${PHOTOGRAPHER_GPUS_PER_PROCESS}" \
            --baseline_name "${model_type}" \
            --output_dir "${output_dir}" \
            --result_json "${output_dir}/eval_records.json" \
            --api_max_workers "${API_MAX_WORKERS:-15}" \
            --api_max_retries "${API_MAX_RETRIES:-3}" \
            --gpt_model "${gpt_model}" \
            --gemini_model "${gemini_model}" \
            --overwrite \
            --gemini_api_key "${GEMINI_API_KEY:-}" \
            --gpt_api_key "${GPT_API_KEY:-}" \
            --qwen_api_key "${QWEN_API_KEY:-}"
        run_photographer_vlm_scoring "${output_dir}" overwrite
    done
}

run_subject() {
    require_dir "${SUBJECT_IMAGE_DIR}"
    require_dir "${SUBJECT_GT_DIR}"
    read -r -a steps <<< "${SUBJECT_STEPS}"
    read -r -a judgers <<< "${SUBJECT_JUDGERS}"
    for step in "${steps[@]}"; do
        local lora_path="${SUBJECT_LORA_TEMPLATE//\{step\}/${step}}"
        local run_id
        run_id=$(basename "$(dirname "$(dirname "${lora_path}")")")_${step}
        local output_dir="${OUTPUT_ROOT}/subject-side/Ours/${run_id}"
        echo "[subject] step=${step} output=${output_dir}"
        "${PYTHON_BIN}" "${SUBJECT_EVAL_DIR}/01_run_benchmark_gen.py" \
            --base_model_path "${SUBJECT_MODEL_PATH}" \
            --lora_path "${lora_path}" \
            --image_dir "${SUBJECT_IMAGE_DIR}" \
            --prompt "${SUBJECT_PROMPT}" \
            --output_dir "${output_dir}/benchmark_pred_json" \
            --dtype "${SUBJECT_DTYPE:-bf16}" \
            --gpus "${SUBJECT_GPUS}" \
            --gpus_per_process "${SUBJECT_GPUS_PER_PROCESS}" \
            --max_new_tokens "${SUBJECT_MAX_NEW_TOKENS}"
        "${PYTHON_BIN}" "${SUBJECT_EVAL_DIR}/03_qwen3vl_draw17.py" \
            --json-dir "${output_dir}/benchmark_pred_json" \
            --image-dir "${SUBJECT_IMAGE_DIR}" \
            --output-dir "${output_dir}/benchmark_pred_json_kptsV"
        for judger in "${judgers[@]}"; do
            "${PYTHON_BIN}" "${SUBJECT_EVAL_DIR}/04_VLM_scores.py" \
                --backend "${judger}" \
                --pred-dir "${output_dir}/benchmark_pred_json_kptsV" \
                --gt-dir "${SUBJECT_GT_DIR}" \
                --gemini-api-key "${GEMINI_API_KEY:-}" \
                --qwen-api-key "${QWEN_API_KEY:-}" \
                --max-workers "${SUBJECT_VLM_MAX_WORKERS:-10}" \
                --output-jsonl "${output_dir}/${judger}_scores.jsonl" \
                --vis-dir "${output_dir}/benchmark_pred_score_vis" \
                --overwrite
        done
    done
}

run_subject_baseline() {
    require_dir "${SUBJECT_IMAGE_DIR}"
    require_dir "${SUBJECT_GT_DIR}"
    if [[ -z "${SUBJECT_YOLO_CHECKPOINT}" || ! -f "${SUBJECT_YOLO_CHECKPOINT}" ]]; then
        echo "Missing YOLO checkpoint. Set SUBJECT_YOLO_CHECKPOINT to yolo26x-pose.pt" >&2
        exit 1
    fi
    read -r -a baselines <<< "${SUBJECT_BASELINE_BACKENDS}"
    for baseline in "${baselines[@]}"; do
        local output_dir="${OUTPUT_ROOT}/subject-side/baseline/${baseline}"
        local edit_meta_dir="${output_dir}/image_edit_meta"
        local edited_dir="${output_dir}/edited_images"
        local yolo_json_dir="${output_dir}/pose_yolo26x_17kpt"
        local draw_dir="${output_dir}/benchmark_pred_json_kptsV"
        echo "[subject-baseline] backend=${baseline} output=${output_dir}"
        "${PYTHON_BIN}" "${SUBJECT_EVAL_DIR}/05_run_baseline.py" \
            --backend "${baseline}" \
            --image_dir "${SUBJECT_IMAGE_DIR}" \
            --output_dir "${edit_meta_dir}" \
            --edited_dir "${edited_dir}" \
            --gemini_api_key "${GEMINI_API_KEY:-}" \
            --gpt_api_key "${GPT_API_KEY:-}" \
            --max_workers "${SUBJECT_BASELINE_MAX_WORKERS}" \
            --limit "${SUBJECT_BASELINE_LIMIT}" \
            --overwrite \
            --max_retries "${SUBJECT_BASELINE_MAX_RETRIES}"
        "${PYTHON_BIN}" "${SUBJECT_EVAL_DIR}/D02_yolo26E2E_infer.py" \
            --input-root "${edited_dir}" \
            --output-root "${yolo_json_dir}" \
            --pe-checkpoint "${SUBJECT_YOLO_CHECKPOINT}" \
            --batch-size "${SUBJECT_YOLO_BATCH_SIZE}" \
            --num-workers "${SUBJECT_YOLO_NUM_WORKERS}" \
            --gpu-ids "${SUBJECT_YOLO_GPUS}"
        "${PYTHON_BIN}" "${SUBJECT_EVAL_DIR}/03_qwen3vl_draw17.py" \
            --json-dir "${yolo_json_dir}" \
            --image-dir "${SUBJECT_IMAGE_DIR}" \
            --output-dir "${draw_dir}"
        "${PYTHON_BIN}" "${SUBJECT_EVAL_DIR}/04_VLM_scores.py" \
            --backend gemini \
            --pred-dir "${draw_dir}" \
            --gt-dir "${SUBJECT_GT_DIR}" \
            --gemini-api-key "${GEMINI_API_KEY:-}" \
            --qwen-api-key "${QWEN_API_KEY:-}" \
            --max-workers "${SUBJECT_VLM_MAX_WORKERS:-10}" \
            --output-jsonl "${output_dir}/gemini_scores.jsonl" \
            --vis-dir "${output_dir}/benchmark_pred_score_vis" \
            --limit 0 \
            --overwrite
    done
}

target="${1:-help}"
case "${target}" in
    photographer-model)
        run_photographer_model
        ;;
    photographer-baseline)
        run_photographer_baseline
        ;;
    subject)
        run_subject
        ;;
    subject-baseline)
        run_subject_baseline
        ;;
    all)
        run_photographer_model
        run_photographer_baseline
        run_subject
        run_subject_baseline
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown target: ${target}" >&2
        usage >&2
        exit 2
        ;;
esac
