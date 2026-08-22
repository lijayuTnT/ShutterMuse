<div align="center">
  <h2>
    <img src="./assets/logo.png" alt="ShutterMuse logo" width="72" align="center">
    ShutterMuse: Capture-Time Photography Guidance with MLLMs
  </h2>
  <p>
    <a href="https://arxiv.org/abs/2606.25763"><img src="https://img.shields.io/badge/arXiv-2606.25763-b31b1b.svg" alt="arXiv"/></a>
    <a href="https://lijayuTnT.github.io/ShutterMuse/"><img src="https://img.shields.io/badge/Project-ShutterMuse-blue.svg" alt="Project"/></a>
    <a href="https://huggingface.co/ShutterMuse/ShutterMuse"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow.svg" alt="Model"/></a>
    <a href="https://huggingface.co/datasets/ShutterMuse/CaptureGuide-Bench"><img src="https://img.shields.io/badge/Bench-HuggingFace-purple.svg" alt="Bench"/></a>
  </p>
</div>

<div align="center">
  <a href="./assets/teaser.png">
    <img src="./assets/teaser.png" alt="ShutterMuse teaser" width="800">
  </a>
</div>


**ShutterMuse** is a unified multimodal large language model for capture-time photography guidance. It supports:

- **Photographer-side guidance**: keep, refine, or reject the current framing, with a composition box when refinement is needed.
- **Subject-side guidance**: recommend scene-conditioned portrait poses with COCO-17 keypoints and visibility states.

## News

- **2026-06**: Code, quick start scripts, evaluation scripts, examples, CaptureGuide-Bench, and ShutterMuse model weights are released.

## CaptureGuide Dataset and Bench

CaptureGuide contains two task sides: photographer-side composition guidance and subject-side pose guidance. CaptureGuide-Dataset is used for model development, while CaptureGuide-Bench evaluates composition decision/refinement and pose recommendation quality.

<div align="center">
  <a href="./assets/data_distribution_01.png">
    <img src="./assets/data_distribution_01.png" alt="CaptureGuide dataset and benchmark distribution" width="800">
  </a>
</div>

<p align="center"><em>Distribution of CaptureGuide-Dataset and CaptureGuide-Bench.</em></p>

## Results

### Photographer-side Guidance

| Method | IoU (%) ↑ | BDE ↓ | R (%) ↑ | RSR (%) ↑ | KSR (%) ↑ | MLLM-Score ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Open-source General MLLMs** |  |  |  |  |  |  |
| InternVL3.5-8B | 42.86 | 0.127 | 8.61 | 0.00 | 20.00 | 0.15 |
| Kimi-K2.6 | 65.44 | 0.087 | 37.92 | 0.00 | 90.90 | 0.47 |
| Qwen3-VL-8B-Instruct | 55.18 | 0.105 | 18.40 | 0.00 | 36.36 | 0.25 |
| Qwen3-VL-32B-Instruct | 63.80 | 0.101 | 35.91 | 13.79 | **98.18** | 0.47 |
| Qwen3-VL-235B-A22B-Instruct | 61.84 | 0.093 | 33.53 | 20.69 | <u>94.55</u> | 0.48 |
| Qwen3.5-9B | 61.94 | 0.094 | 30.86 | 3.45 | 83.64 | 0.45 |
| Qwen3.6-27B | 53.93 | 0.090 | 33.23 | 48.28 | 72.72 | 0.47 |
| **Proprietary General MLLMs** |  |  |  |  |  |  |
| Gemini-3.0-Flash | 64.10 | 0.079 | 38.58 | 55.17 | 87.27 | 0.50 |
| Gemini-3.0-Pro | 63.62 | 0.070 | 47.48 | **82.76** | 89.09 | 0.54 |
| Gemini-3.1-Pro | 65.63 | <u>0.068</u> | 51.34 | <u>79.31</u> | 89.09 | 0.56 |
| Gemini-3.5-Flash | 66.95 | 0.076 | 41.54 | 48.28 | 67.27 | 0.50 |
| GPT-5.4 | 64.72 | 0.093 | 40.06 | 10.34 | 85.45 | 0.49 |
| GPT-5.5 | 65.44 | 0.091 | 41.84 | 10.34 | 81.82 | 0.48 |
| **Specialized Aesthetic Cropping Models** |  |  |  |  |  |  |
| CACNet | 68.29 | 0.080 | 54.08 | 0.00 | 0.00 | 0.52 |
| UNIC | 62.46 | 0.081 | 31.12 | 0.00 | 0.00 | 0.29 |
| InstructCrop | <u>69.53</u> | 0.072 | 56.97 | 0.00 | 0.00 | 0.43 |
| Venus | 69.43 | 0.076 | <u>57.27</u> | 0.00 | 3.64 | <u>0.57</u> |
| **ShutterMuse (Ours)** | **74.65** | **0.051** | **67.06** | **82.76** | 74.55 | **0.64** |

### FLMS Benchmark

| Method | IoU (%) ↑ | BDE ↓ |
| --- | ---: | ---: |
| Gemini-3.0-Pro | 77.13 | 0.049 |
| Gemini-3.5-Flash | 77.69 | 0.048 |
| GPT-5.5 | 64.99 | 0.049 |
| InstructCrop | 80.98 | 0.043 |
| CACNet | 84.04 | 0.037 |
| Venus | <u>86.63</u> | <u>0.030</u> |
| **ShutterMuse (Ours)** | **87.61** | **0.027** |

### Subject-side Guidance


| Method          | Plausibility ↑ | Interaction ↑ | Aesthetics ↑ | Mean ↑ | Time ↓   | Tokens ↓ |
| --------------- | -------------- | ------------- | ------------ | ------ | -------- | -------- |
| Nano-Banana-Pro | 0.63           | 0.35          | 0.17         | 0.39   | 55.16    | 1370     |
| GPT-Image-2     | 0.59           | 0.29          | 0.15         | 0.35   | 102.61   | 1427     |
| **ShutterMuse** | 0.58           | 0.27          | 0.14         | 0.34   | **4.96** | **412**  |


## Installation

```bash
git clone https://github.com/lijayuTnT/ShutterMuse.git
cd ShutterMuse
conda create -n shuttermuse python=3.10 -y
conda activate shuttermuse
pip install -r requirements.txt
```

ShutterMuse model weights are released on [Hugging Face](https://huggingface.co/ShutterMuse/ShutterMuse). Please prepare the base or merged Qwen-VL checkpoint and the ShutterMuse LoRA/checkpoint according to the model card.

## Quick Start

Set checkpoint paths:

```bash
export MODEL_PATH=/path/to/base-or-merged-qwen-vl-checkpoint
export LORA_PATH=/path/to/shuttermuse-lora  # leave empty for a fully merged checkpoint
export OUTPUT_DIR=outputs/quick_start
```

Photographer-side composition guidance:

```bash
bash evaluation/scripts/quick_start.sh \
  --side photographer \
  --image test/401128801616615964.webp \
  --model-path "$MODEL_PATH" \
  --lora-path "$LORA_PATH" \
  --output-dir "$OUTPUT_DIR"
```

Subject-side pose guidance:

```bash
bash evaluation/scripts/quick_start.sh \
  --side subject \
  --image /path/to/scene.jpg \
  --model-path "$MODEL_PATH" \
  --lora-path "$LORA_PATH" \
  --output-dir "$OUTPUT_DIR"
```

Outputs include a JSON prediction and a `.webp` visualization. Run `bash evaluation/scripts/quick_start.sh --help` for all options.

## Training

ShutterMuse training follows two stages. The released scripts are lightweight launch templates; set local model, data, and GPU paths before running.

Stage 1: supervised fine-tuning (SFT) with ModelScope Swift:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export SFT_DATASET=/path/to/sft_train.jsonl
export OUTPUT_ROOT=outputs/training/stage1_sft
bash training/stage1_sft.sh
```

Stage 2: GRPO fine-tuning from the stage-1 checkpoint:

```bash
export MODEL_PATH=/path/to/stage1-merged-or-base-checkpoint
export GRPO_DATASET_PATH=/path/to/grpo_dataset.jsonl
export OUTPUT_ROOT=outputs/training/stage2_grpo
bash training/stage2_grpo.sh
```

Optional saliency rewards can use a precomputed BiRefNet file:

```bash
python training/grpo_utils/precompute_birefnet_saliency.py \
  --dataset "$GRPO_DATASET_PATH" \
  --output /path/to/grpo_dataset_birefnet_saliency.jsonl
export SALIENCY_PRECOMPUTE_JSONL=/path/to/grpo_dataset_birefnet_saliency.jsonl
```

The GRPO script registers datasets with `training/grpo_utils/data_format.py` and rewards with `training/grpo_utils/reward_func.py` (`ratio_orm`, `iou_orm`, `pose_visibility_orm`, `saliency_orm`). Common overrides include `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `LEARNING_RATE`, `OUTPUT_DIR`, and `VLLM_SERVER_PORT`.

## Evaluation

Unified entry:

```bash
bash evaluation/scripts/run_unified_evaluation.sh photographer-model
bash evaluation/scripts/run_unified_evaluation.sh photographer-baseline
bash evaluation/scripts/run_unified_evaluation.sh subject
bash evaluation/scripts/run_unified_evaluation.sh subject-baseline
```

Common configuration:

```bash
export OUTPUT_ROOT=outputs/evaluation
export PHOTOGRAPHER_MODEL_PATH=/path/to/base-or-merged-qwen-vl-checkpoint
export PHOTOGRAPHER_LORA_TEMPLATE=/path/to/lora/checkpoint-{step}
export PHOTOGRAPHER_STEPS="6000"
export SUBJECT_MODEL_PATH=/path/to/base-or-merged-qwen-vl-checkpoint
export SUBJECT_LORA_TEMPLATE=/path/to/lora/checkpoint-{step}
export SUBJECT_STEPS="6000"
```

For VLM scoring or API baselines, set keys through environment variables:

```bash
export GEMINI_API_KEY="your_api_key"
export QWEN_API_KEY="your_api_key"
export GPT_API_KEY="your_api_key"
```

## Repository Structure

```text
ShutterMuse/
├── assets/          # README figures
├── evaluation/      # Inference and benchmark scripts
├── training/        # Two-stage SFT and GRPO training scripts
├── test/            # Small example images
├── README.md
└── requirements.txt
```

`Benchmark/` and `outputs/` are intentionally excluded from git. The released benchmark is available on [Hugging Face](https://huggingface.co/datasets/ShutterMuse/CaptureGuide-Bench).

## Data and Checkpoints


| Resource               | Status      | Link |
| ---------------------- | ----------- | ---- |
| CaptureGuide-Bench     | Released    | [Hugging Face](https://huggingface.co/datasets/ShutterMuse/CaptureGuide-Bench) |
| ShutterMuse checkpoint | Released    | [Hugging Face](https://huggingface.co/ShutterMuse/ShutterMuse) |


## Citation

```bibtex
@misc{li2026shuttermuse,
  title        = {ShutterMuse: Capture-Time Photography Guidance with MLLMs},
  author       = {Li, Jiayu and Fang, Yixiao and Hu, Tianyu and Cheng, Wei and Huang, Ping and Fan, Zheheng and Yu, Gang and Ma, Xingjun},
  year         = {2026},
  note         = {Preprint}
}
```

## License

TODO: Add license information before public release.
