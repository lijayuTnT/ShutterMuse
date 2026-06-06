<div align="center">
  <h2>ShutterMuse: Capture-Time Photography Guidance with MLLMs</h2>
  <!-- <p>
    Jiayu Li, Yixiao Fang, Tianyu Hu, Wei Cheng, Ping Huang,
    Zheheng Fan, Gang Yu, Xingjun Ma
  </p> -->
  <!-- <p><em>Project page, paper, checkpoints, and dataset links will be updated upon release.</em></p> -->
  <p>
    <a href="#citation"><img src="https://img.shields.io/badge/Paper-Citation-b31b1b.svg" alt="Paper"/></a>
    <a href="#captureguide-bench"><img src="https://img.shields.io/badge/CaptureGuide--Bench-Benchmark-green.svg" alt="CaptureGuide-Bench"/></a>
    <a href="#captureguide-dataset"><img src="https://img.shields.io/badge/CaptureGuide--Dataset-Dataset-green.svg" alt="CaptureGuide-Dataset"/></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-Inference-blue.svg" alt="Quick Start"/></a>
    <a href="#evaluation"><img src="https://img.shields.io/badge/Evaluation-Scripts-orange.svg" alt="Evaluation"/></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-TBD-lightgrey.svg" alt="License"/></a>
  </p>
</div>

<div align="center">
  <a href="./assets/teaser.png">
    <img src="./assets/teaser.png" alt="ShutterMuse teaser" width="800">
  </a>
</div>

> Official repository for **ShutterMuse**, a unified multimodal large language model for capture-time photography guidance.

ShutterMuse studies a practical photography setting where guidance is needed **before or during image capture**, rather than only after a photo has already been taken. Given an image and a user request, ShutterMuse can reason from both sides of the shooting process:

- **Photographer-side guidance**: decide whether the current framing should be **kept**, **refined**, or **rejected**, and output a valid composition box when refinement is needed.
- **Subject-side guidance**: recommend a scene-conditioned human pose with both natural-language guidance and structured COCO-17 keypoints.

The project introduces **CaptureGuide-Bench** for evaluation, **CaptureGuide-Dataset** for model development, and **ShutterMuse**, an MLLM trained with supervised fine-tuning and reinforcement fine-tuning.

## News

- **2026-06**: Repository initialized with CaptureGuide-Bench evaluation code, single-image quick start scripts, and example outputs.
- **TBD**: Release public links for CaptureGuide-Bench, CaptureGuide-Dataset, checkpoints, and model cards.

## Overview

Real-world photography requires decisions that are not fully covered by conventional aesthetic cropping benchmarks. A photographer may need to know whether an image is already good enough, whether a better frame can be obtained, or whether the current view should be abandoned. At the same time, the subject may need actionable pose suggestions that fit the scene.

ShutterMuse addresses this capture-time setting with two complementary capabilities:

### Photographer-side Guidance

Given an image and an optional user instruction such as a desired aspect ratio or shooting intention, the model outputs:

- `keep`: the current composition is already suitable;
- `refine`: the image can be improved by reframing, together with a normalized box `[x1, y1, x2, y2]`;
- `reject`: no aesthetically valid composition can be identified from the current image.

### Subject-side Guidance

Given a person-free scene image, the model recommends a pose that is:

- physically plausible;
- semantically aligned with scene layout and objects;
- visually appealing for photography.

The model returns textual pose instructions and structured pose annotations in the standard COCO-17 keypoint format.

## Key Contributions

- **CaptureGuide-Bench**: a benchmark for capture-time photography guidance, covering photographer-side composition decision/refinement and subject-side pose recommendation.
- **CaptureGuide-Dataset**: a large-scale dataset with approximately **130K** samples, including textual rationales, composition boxes, pose keypoints, and visibility states.
- **ShutterMuse**: a unified MLLM initialized from Qwen3-VL-8B and trained with supervised fine-tuning plus GRPO-based reinforcement fine-tuning.
- **Structured guidance**: JSON-style outputs make the model usable in downstream photography assistants, camera applications, and interactive editing tools.

## CaptureGuide-Dataset

CaptureGuide-Dataset contains two subsets corresponding to the two guidance tasks.

<div align="center">
  <a href="./assets/data_distribution_01.png">
    <img src="./assets/data_distribution_01.png" alt="CaptureGuide dataset and benchmark distribution" width="800">
  </a>
</div>

<p align="center"><em>Distribution of CaptureGuide-Dataset and CaptureGuide-Bench across task sides, scene categories, aspect ratios, and composition decisions.</em></p>

### Photographer-side Subset

The photographer-side data covers five representative photography scenarios:

- portrait;
- landscape;
- street snap;
- animal;
- still life.

Each sample is annotated with a three-way decision: `refine`, `keep`, or `reject`. For refinement samples, the annotation includes normalized composition boxes and textual rationales explaining the aesthetic decision.

The dataset is scaled with an **expert-seeded, MLLM-verified self-distillation pipeline (EMDP)**:

1. experts annotate a high-quality seed set;
2. an MLLM converts comments into structured rationales;
3. an initial composition model generates pseudo annotations;
4. an MLLM verifier filters rationale correctness and rationale-box consistency;
5. verified data is used for iterative retraining.

### Subject-side Subset

The subject-side data is built from portrait photographs and converted into triplets:

- a person-free scene image;
- target human pose keypoints;
- textual rationales explaining why the pose fits the scene.

The construction pipeline uses person removal, YOLO-based pose extraction, MLLM-assisted rationale generation, and expert verification. Each pose is represented using COCO-17 normalized keypoints and a visibility vector.

Visibility states:

- `1`: visible keypoint;
- `0`: invisible or occluded but inside the image;
- `-1`: outside the image frame.

## CaptureGuide-Bench

CaptureGuide-Bench evaluates capture-time photography guidance with held-out samples that are not used during SFT or RFT.

### Photographer-side Evaluation

Models are evaluated on three-way decision making and composition quality. Metrics include:

- **IoU**: maximum overlap between predicted and annotated boxes;
- **BDE**: boundary displacement error;
- **R**: refinement success rate with IoU greater than 0.7;
- **RSR**: reject success rate;
- **KSR**: keep success rate;
- **MLLM-Score**: task-aware composition and decision quality score.

### Subject-side Evaluation

Pose recommendation is evaluated with an MLLM judge after rendering predicted keypoints as a skeleton overlay. The evaluation focuses on:

- physical plausibility;
- scene interaction;
- pose aesthetics.

The benchmark also reports inference efficiency, including average inference time and generated token count.

## Method

ShutterMuse is a unified MLLM for both photographer-side and subject-side guidance.

### Model

- Backbone: Qwen3-VL-8B
- Stage 1: supervised fine-tuning on CaptureGuide-Dataset
- Stage 2: reinforcement fine-tuning with GRPO
- Output format: structured JSON responses

### Output Schema

#### Photographer-side Guidance

```json
{
  "task_type": "composition",
  "reason": "A concise explanation of the aesthetic decision.",
  "composition_xy": [0.12, 0.08, 0.92, 0.86]
}
```

`composition_xy` follows this convention:

- `[]`: reject;
- `[0, 0, 1, 1]`: keep;
- `[x1, y1, x2, y2]`: refine.

#### Subject-side Guidance

```json
{
  "task_type": "pose",
  "reason": "Natural-language pose recommendation grounded in the scene.",
  "keypoints_xyn": [[0.50, 0.18], [0.49, 0.16], ...],
  "visibility": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

`keypoints_xyn` should contain 17 normalized COCO keypoints. The shortened example above is only for illustration.

## Results

On CaptureGuide-Bench, ShutterMuse achieves the best overall photographer-side performance among evaluated baselines and provides competitive subject-side pose recommendation with substantially lower inference cost.

### Photographer-side Guidance


| Method          | IoU ↑     | BDE ↓     | R ↑       | RSR ↑     | KSR ↑ | MLLM-Score ↑ |
| --------------- | --------- | --------- | --------- | --------- | ----- | ------------ |
| Gemini-3.0-Pro  | 63.62     | 0.070     | 47.48     | 82.76     | 89.09 | 0.54         |
| GPT-5.5         | 65.44     | 0.091     | 41.84     | 10.34     | 81.82 | 0.48         |
| Venus           | 69.43     | 0.076     | 57.27     | 0.00      | 3.64  | 0.57         |
| **ShutterMuse** | **74.30** | **0.054** | **70.03** | **82.76** | 74.55 | **0.64**     |


### Subject-side Guidance


| Method          | Plausibility ↑ | Interaction ↑ | Aesthetics ↑ | Mean ↑ | Time ↓   | Tokens ↓ |
| --------------- | -------------- | ------------- | ------------ | ------ | -------- | -------- |
| Nano-Banana-Pro | 0.63           | 0.35          | 0.17         | 0.39   | 55.16    | 1370     |
| GPT-Image-2     | 0.59           | 0.29          | 0.15         | 0.35   | 102.61   | 1427     |
| **ShutterMuse** | 0.58           | 0.27          | 0.14         | 0.34   | **4.96** | **412**  |


## Repository Structure

```text
ShutterMuse/
├── Benchmark/
│   ├── photographer-side/        # CaptureGuide-Bench photographer-side data
│   └── subject-side/             # CaptureGuide-Bench subject-side data
├── evaluation/
│   ├── photographer-side/        # Composition/refinement evaluation scripts
│   ├── subject-side/             # Pose recommendation evaluation scripts
│   ├── scripts/                  # Unified quick start and benchmark runners
│   └── README.md
├── outputs/                      # Default output directory for quick start/evaluation runs
├── test/                         # Example images for smoke testing
└── README.md
```

Large model checkpoints are not stored in this repository. Please download or prepare the base Qwen-VL checkpoint and the ShutterMuse LoRA/checkpoint separately, then pass their paths to the scripts below.

## Installation

The dependency list in [requirements.txt](requirements.txt) was exported from the local `qwen3` environment used for development and testing. The original environment uses **Python 3.10** with PyTorch/CUDA packages for MLLM inference and evaluation.

We recommend creating a fresh environment with `conda` and installing the exported dependencies:

```bash
git clone https://github.com/<your-org-or-username>/ShutterMuse.git
cd ShutterMuse

conda create -n shuttermuse python=3.10 -y
conda activate shuttermuse

pip install -r requirements.txt
```

If `flash_attn` installation fails, please install a wheel that matches your local CUDA and PyTorch versions, then rerun the remaining requirements installation.

## Quick Start

The quickest way to test ShutterMuse on one image is `evaluation/scripts/quick_start.sh`. It wraps `evaluation/photographer-side/infer_single_qwen_lora.py` and supports both photographer-side composition guidance and subject-side pose recommendation.

Before running, set the checkpoint paths. The script contains development-machine defaults, so public users should override them explicitly:

```bash
export MODEL_PATH=/path/to/base-or-merged-qwen-vl-checkpoint
export LORA_PATH=/path/to/shuttermuse-lora-or-empty-string
export OUTPUT_DIR=outputs/quick_start
```

If you are using a fully merged checkpoint, leave `LORA_PATH` empty:

```bash
export LORA_PATH=
```

The default outputs are written to `outputs/quick_start/<side>-side/` and include:

- a `.json` file containing the raw model response, parsed prediction, prompt, and metadata;
- a `.webp` visualization with the predicted composition box or rendered COCO-17 skeleton.

Check all quick start options with:

```bash
bash evaluation/scripts/quick_start.sh --help
```

### Photographer-side Guidance

Run composition guidance on one image:

```bash
bash evaluation/scripts/quick_start.sh \
  --side photographer \
  --image test/401128801616615964.webp \
  --model-path "$MODEL_PATH" \
  --lora-path "$LORA_PATH" \
  --output-dir outputs/quick_start
```

By default, the script builds a Chinese composition prompt automatically. It resizes the image for inference, selects the closest aspect ratio from the supported ratios, and asks the model to return one bounding box in `(x1,y1),(x2,y2)` format. To provide your own prompt, use `--photographer-prompt`:

```bash
bash evaluation/scripts/quick_start.sh \
  --side photographer \
  --image /path/to/image.jpg \
  --photographer-prompt "请找出图片中构图最好的区域，请按照3:4的比例输出bounding box。"
```

The JSON output contains fields such as:

```json
{
  "side": "photographer",
  "instruction": "...",
  "output_text": "...",
  "pred_bbox": [120.0, 30.0, 860.0, 990.0],
  "visualization_path": "outputs/quick_start/photographer-side/<name>.webp"
}
```

### Subject-side Guidance

Run pose recommendation on one scene image:

```bash
bash evaluation/scripts/quick_start.sh \
  --side subject \
  --image test/1_2026-03-18_3：4_风景.jpg \
  --model-path "$MODEL_PATH" \
  --lora-path "$LORA_PATH" \
  --output-dir outputs/quick_start
```

The default subject-side prompt asks the model to recommend a portrait pose and return the relative coordinates and visibility of 17 COCO keypoints in JSON format. To override the prompt, use `--subject-prompt`.

The JSON output is compatible with the subject-side visualization and scoring scripts:

```json
{
  "instance_info": [
    {
      "keypoints_xyn": [[0.50, 0.18], [0.49, 0.16]],
      "visibility": [1, 1]
    }
  ],
  "meta": {
    "side": "subject",
    "instruction": "...",
    "visualization_path": "outputs/quick_start/subject-side/<name>.webp"
  }
}
```

You can also call the underlying Python script directly when integrating it into another pipeline:

```bash
python evaluation/photographer-side/infer_single_qwen_lora.py \
  --side photographer \
  --model_path "$MODEL_PATH" \
  --lora_path "$LORA_PATH" \
  --image /path/to/image.jpg \
  --output_dir outputs/quick_start/photographer-side
```

## Evaluation

Evaluation code is organized under [`evaluation/`](evaluation/README.md). The recommended entry point is `evaluation/scripts/run_unified_evaluation.sh`, which provides one target for each benchmark setting:

```bash
bash evaluation/scripts/run_unified_evaluation.sh help
```

Available targets:

| Target | What it runs | Main outputs |
| ------ | ------------ | ------------ |
| `photographer-model` | ShutterMuse/Qwen-VL + LoRA on the photographer-side benchmark, followed by VLM scoring. | `eval_records.json`, `meta.json`, visualizations, VLM score summaries |
| `photographer-baseline` | API or local baseline models on the photographer-side benchmark, followed by VLM scoring. | baseline prediction records and VLM score summaries |
| `subject` | ShutterMuse/Qwen-VL + LoRA subject-side generation, skeleton rendering, and VLM scoring. | predicted keypoint JSON files, skeleton images, score JSONL |
| `subject-baseline` | Image-editing baselines, YOLO pose extraction, skeleton rendering, and VLM scoring. | edited images, YOLO keypoints, skeleton images, score JSONL |
| `all` | Runs all targets above in sequence. | all benchmark outputs |

All outputs are written under `outputs/evaluation/` by default. Override this with `OUTPUT_ROOT`.

### Common Configuration

Most paths are controlled by environment variables so that credentials and local checkpoints do not need to be committed:

```bash
export PYTHON_BIN=python
export OUTPUT_ROOT=outputs/evaluation

export PHOTOGRAPHER_MODEL_PATH=/path/to/base-or-merged-qwen-vl-checkpoint
export PHOTOGRAPHER_LORA_TEMPLATE=/path/to/lora/checkpoint-6000
export PHOTOGRAPHER_STEPS="6000"
export PHOTOGRAPHER_GPUS="0,1,2,3,4,5,6,7"
export PHOTOGRAPHER_GPUS_PER_PROCESS=1

export SUBJECT_MODEL_PATH=/path/to/base-or-merged-qwen-vl-checkpoint
export SUBJECT_LORA_TEMPLATE=/path/to/lora/checkpoint-6000
export SUBJECT_STEPS="6000"
export SUBJECT_GPUS="0,1,2,3,4,5,6,7"
export SUBJECT_GPUS_PER_PROCESS=1
```

If your LoRA path contains a `{step}` placeholder, the runner replaces it with each value in `PHOTOGRAPHER_STEPS` or `SUBJECT_STEPS`. For example:

```bash
export PHOTOGRAPHER_LORA_TEMPLATE=/path/to/run/checkpoint-{step}
export PHOTOGRAPHER_STEPS="2000 4000 6000"
```

For VLM-based scoring and API baselines, set API keys through environment variables. Do not hard-code them in scripts:

```bash
export GEMINI_API_KEY="your_api_key"
export QWEN_API_KEY="your_api_key"
export GPT_API_KEY="your_api_key"
```

### Photographer-side Benchmark

The photographer-side benchmark uses:

- annotations: `Benchmark/photographer-side/composition_benchmark/meta_new.json`;
- images: `Benchmark/photographer-side/composition_benchmark/original_composition/`.

Run ShutterMuse on the photographer-side benchmark:

```bash
bash evaluation/scripts/run_unified_evaluation.sh photographer-model
```

The compatibility wrapper below runs the same target:

```bash
bash evaluation/photographer-side/scripts/run_benchmark.sh
```

Run photographer-side baselines:

```bash
export PHOTOGRAPHER_BASELINE_MODELS="gemini-3-pro-native"
bash evaluation/scripts/run_unified_evaluation.sh photographer-baseline
```

The model benchmark produces `eval_records.json` with parsed composition predictions, `meta.json` with aggregate metrics, and VLM-score files such as `vlm_scores_details.jsonl` and `vlm_scores_summary.json`.

### Subject-side Benchmark

The subject-side benchmark uses:

- input images: `Benchmark/subject-side/paper-benchmark/`;
- ground-truth visual references: `Benchmark/subject-side/paper-benchmark-gt/`.

Run ShutterMuse on the subject-side benchmark:

```bash
bash evaluation/scripts/run_unified_evaluation.sh subject
```

The compatibility wrapper below runs the same target:

```bash
bash evaluation/subject-side/scripts/benchmark_gen.sh
```

This target first writes model predictions to `benchmark_pred_json/`, then renders COCO-17 skeletons with `03_qwen3vl_draw17.py`, and finally scores the rendered predictions with `04_VLM_scores.py`.

Run subject-side baselines:

```bash
export SUBJECT_BASELINE_BACKENDS="gemini gpt"
export SUBJECT_YOLO_CHECKPOINT=/path/to/yolo26x-pose.pt
bash evaluation/scripts/run_unified_evaluation.sh subject-baseline
```

Subject baselines generate edited images, extract poses with YOLO, render skeleton visualizations, and run VLM scoring. Use `SUBJECT_BASELINE_LIMIT` for a small smoke test before launching a full benchmark.

### Running Individual Scripts

The unified runner is recommended, but each stage can be called manually when debugging:

```bash
# Photographer-side model benchmark
python evaluation/photographer-side/evaluate_benchmark.py --help

# Photographer-side VLM scoring
python evaluation/photographer-side/evaluate_vlm_benchmark.py --help

# Subject-side model generation
python evaluation/subject-side/01_run_benchmark_gen.py --help

# Draw COCO-17 keypoints as skeleton overlays
python evaluation/subject-side/03_qwen3vl_draw17.py --help

# Subject-side VLM scoring
python evaluation/subject-side/04_VLM_scores.py --help
```

## Data and Checkpoints

> TODO: Add release links and access instructions.


| Resource                   | Status      | Link |
| -------------------------- | ----------- | ---- |
| CaptureGuide-Bench         | Coming soon | TODO |
| CaptureGuide-Dataset       | Coming soon | TODO |
| ShutterMuse-SFT checkpoint | Coming soon | TODO |
| ShutterMuse-RFT checkpoint | Coming soon | TODO |


## Citation

If you find this project useful, please consider citing our work:

```bibtex
@inproceedings{li2027shuttermuse,
  title     = {ShutterMuse: Capture-Time Photography Guidance with MLLMs},
  author    = {Li, Jiayu and Fang, Yixiao and Hu, Tianyu and Cheng, Wei and Huang, Ping and Fan, Zheheng and Yu, Gang and Ma, Xingjun},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

## License

TODO: Add license information before public release.

## Acknowledgements

This project builds on recent progress in multimodal large language models, aesthetic cropping, pose estimation, and visual evaluation. We thank the annotators and photographers involved in constructing and verifying CaptureGuide-Dataset and CaptureGuide-Bench.
