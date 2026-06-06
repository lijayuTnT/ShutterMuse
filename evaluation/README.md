# Evaluation

This directory contains the evaluation code used by ShutterMuse.

## Structure

- `photographer-side/`: composition/cropping evaluation.
- `scripts/run_unified_evaluation.sh`: unified entry for photographer model, photographer baseline, subject-side model, and subject-side baseline evaluation.
- `photographer-side/scripts/run_benchmark.sh`: compatibility wrapper for `photographer-model`.
- `subject-side/`: pose recommendation evaluation.
- `subject-side/scripts/benchmark_gen.sh`: compatibility wrapper for `subject`.

## Notes

- Model checkpoints, datasets, and output paths are still configured in the shell scripts.
- API keys are read from environment variables instead of being stored in the repository.
- Set `GEMINI_API_KEY` and/or `QWEN_API_KEY` before running VLM-based scoring.


## Benchmark Data

- `../Benchmark/photographer-side/composition_benchmark/meta_new.json` and `original_composition/` are used by photographer-side evaluation.
- `../Benchmark/subject-side/paper-benchmark/` and `paper-benchmark-gt/` are used by subject-side evaluation.
- The copied benchmark data is minimal: only files referenced or matched by the current benchmark scripts are kept.


## Unified Runner

Run one target at a time:

```bash
evaluation/scripts/run_unified_evaluation.sh photographer-model
evaluation/scripts/run_unified_evaluation.sh photographer-baseline
evaluation/scripts/run_unified_evaluation.sh subject
evaluation/scripts/run_unified_evaluation.sh subject-baseline
```

Use environment variables such as `OUTPUT_ROOT`, `PYTHON_BIN`, `GEMINI_API_KEY`, `QWEN_API_KEY`, `GPT_API_KEY`, and `SUBJECT_YOLO_CHECKPOINT` to configure runtime paths and API credentials.


## Quick Start

Run single-image inference for either side:

```bash
evaluation/scripts/quick_start.sh --side photographer --image /path/to/image.jpg
evaluation/scripts/quick_start.sh --side subject --image /path/to/image.jpg
```

Use `--output-dir`, `--model-path`, and `--lora-path` to override defaults.
