#!/usr/bin/env python3
"""Baseline benchmark script: text generation for keypoints + visibility.

This script runs a Qwen-VL generation model on benchmark images, parses model text
output into JSON, and writes one prediction file per image.
Output format is compatible with pose_02_draw17.py:
{
  "instance_info": [
    {
      "visibility": [...],
      "keypoints_xyn": [[x, y], ...]
    }
  ],
  "meta": {...}
}
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

DEFAULT_PROMPT = (
    "You are a portrait photography pose analysis expert. Based on the image, recommend a human pose "
    "and provide the relative coordinates of 17 human keypoints and whether each keypoint is visible "
    "in the image in JSON format. The 17 keypoints are, in order: nose, left eye, right eye, "
    "left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, "
    "right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle."
)
MIXTASK_PROMPT = (
    "You are a portrait photography expert. Based on the image, first identify a suitable composition "
    "and then recommend a human pose. Provide the relative coordinates of 17 human keypoints and "
    "whether each keypoint is visible in the image in JSON format. The 17 keypoints are, in order: "
    "nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, "
    "right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, "
    "right ankle."
)
RATIO_PROMPT=(
    ""
)
SUBJECT_PROMPT=(
    ""
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend. Use 'vllm' for Qwen3-VL accelerated inference.",
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--annotation_json",
        type=str,
        default="",
        help="Optional train/eval json or jsonl. If provided, objects.bbox is used as a fallback bbox per image.",
    )
    parser.add_argument("--prompt", type=str, default="", help="Optional explicit prompt. Overrides --mixtask prompt selection.")
    parser.add_argument(
        "--mixtask",
        action="store_true",
        help="If set and --prompt is not provided, use the composition-and-pose prompt.",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--limit", type=int, default=-1, help="Maximum number of images to evaluate in this run.")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--filename_mode", type=str, default="relative", choices=["relative", "stem"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpus", type=str, default="")
    parser.add_argument("--gpus_per_process", type=int, default=1)
    parser.add_argument("--vllm_batch_size", type=int, default=8, help="Batch size per vLLM generate call.")
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=0,
        help="Tensor parallel size for vLLM. Default 0 means infer from --gpus_per_process or use 1.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--meta_output_path",
        type=str,
        default="",
        help="Output path for visualization meta.json. Default: <output_dir_parent>/benchmark_vis/meta.json",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="If set, append one depth image after RGB image for each sample.",
    )
    parser.add_argument(
        "--depth_image_root",
        type=str,
        default="",
        help="Depth image root. For 001.png, expected file is <depth_image_root>/001_depth.png",
    )
    parser.add_argument(
        "--depth_image_suffix",
        type=str,
        default="_depth.png",
        help="Depth filename suffix appended to RGB stem.",
    )
    return parser.parse_args()


def pick_dtype(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def make_output_name(image_path: Path, image_root: Path, mode: str) -> str:
    if mode == "relative":
        name = str(image_path.relative_to(image_root).with_suffix(""))
        name = name.replace("/", "__")
    else:
        suffix = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8]
        name = f"{image_path.stem}__{suffix}"
    return name


def extract_json_text(text: str) -> Optional[str]:
    # Find first plausible JSON object span.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def try_parse_obj(text: str) -> Optional[Dict]:
    s = text.strip()
    # 1) strict json
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) python literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) regex extract json object then parse
    js = extract_json_text(s)
    if js is not None:
        for fn in (json.loads, ast.literal_eval):
            try:
                obj = fn(js)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def _coerce_bbox_numbers(values: Sequence, image_size: Optional[tuple[int, int]] = None) -> Optional[List[float]]:
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except Exception:
            return None
    if len(nums) != 4:
        return None

    x1, y1, x2, y2 = nums
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_abs <= 1.0:
        norm = [x1, y1, x2, y2]
    elif max_abs <= 1000.0:
        # Qwen bbox-style outputs are often in the 0-1000 coordinate space.
        norm = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
    elif image_size is not None:
        w, h = image_size
        if w <= 0 or h <= 0:
            return None
        norm = [x1 / float(w), y1 / float(h), x2 / float(w), y2 / float(h)]
    else:
        return None

    return [max(0.0, min(1.0, float(v))) for v in norm]


def _parse_bbox_value(value, image_size: Optional[tuple[int, int]] = None) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        for keys in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
            if all(k in value for k in keys):
                return _coerce_bbox_numbers([value[k] for k in keys], image_size=image_size)
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return _parse_bbox_value(value[0], image_size=image_size)
        if len(value) == 2 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in value):
            return _coerce_bbox_numbers([value[0][0], value[0][1], value[1][0], value[1][1]], image_size=image_size)
        if len(value) >= 4:
            return _coerce_bbox_numbers(value[:4], image_size=image_size)
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s == "<bbox>":
            return None
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if len(nums) >= 4:
            return _coerce_bbox_numbers(nums[:4], image_size=image_size)
    return None


def _iter_records_from_annotation(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"--annotation_json not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("records", "data", "items", "annotations"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
            if isinstance(value, dict):
                return [x for x in value.values() if isinstance(x, dict)]
        return [data]
    return []


def _record_image_names(record: Dict) -> List[str]:
    names: List[str] = []
    images = record.get("images")
    if isinstance(images, list):
        names.extend(str(x) for x in images if x)
    for key in ("image", "image_path", "img_path", "path", "file_name", "filename"):
        value = record.get(key)
        if value:
            names.append(str(value))
    meta = record.get("meta")
    if isinstance(meta, dict):
        for key in ("image_path", "file_name", "filename"):
            value = meta.get(key)
            if value:
                names.append(str(value))
    return names


def _record_bbox(record: Dict) -> Optional[List[float]]:
    objects = record.get("objects")
    if isinstance(objects, dict):
        bbox = _parse_bbox_value(objects.get("bbox"))
        if bbox is not None:
            return bbox
    return extract_bbox_from_obj(record)


def load_annotation_bbox_map(annotation_json: str) -> Dict[str, List[float]]:
    if not annotation_json:
        return {}
    bbox_map: Dict[str, List[float]] = {}
    for record in _iter_records_from_annotation(Path(annotation_json)):
        bbox = _record_bbox(record)
        if bbox is None:
            continue
        for name in _record_image_names(record):
            p = Path(name)
            keys = {name, p.name, p.stem}
            for key in keys:
                if key:
                    bbox_map[key] = bbox
    return bbox_map


def lookup_annotation_bbox(bbox_map: Dict[str, List[float]], image_path: Path, out_name: str) -> Optional[List[float]]:
    if not bbox_map:
        return None
    candidates = [
        str(image_path),
        image_path.name,
        image_path.stem,
        out_name,
        f"{out_name}{image_path.suffix}",
    ]
    for key in candidates:
        bbox = bbox_map.get(key)
        if bbox is not None:
            return bbox
    return None


def extract_bbox_from_obj(obj: Dict, image_size: Optional[tuple[int, int]] = None) -> Optional[List[float]]:
    if "instance_info" in obj and isinstance(obj["instance_info"], list) and len(obj["instance_info"]) > 0:
        inst = obj["instance_info"][0]
    else:
        inst = obj
    if not isinstance(inst, dict):
        return None

    for key in ("bbox", "composition_bbox", "composition_xy", "box_xyxyn", "box"):
        bbox = _parse_bbox_value(inst.get(key), image_size=image_size)
        if bbox is not None:
            return bbox
    for key in ("bbox", "composition_bbox", "composition_xy", "box_xyxyn", "box"):
        bbox = _parse_bbox_value(obj.get(key), image_size=image_size)
        if bbox is not None:
            return bbox
    return None


def extract_trailing_bbox_from_text(text: str, image_size: Optional[tuple[int, int]] = None) -> Optional[List[float]]:
    js = extract_json_text(text)
    if not js:
        return None
    start = text.find(js)
    if start < 0:
        return None
    tail = text[start + len(js) :].strip()
    return _parse_bbox_value(tail, image_size=image_size)


def sanitize_instance_info(obj: Dict) -> Dict:
    """Normalize output to {instance_info:[{keypoints_xyn, visibility}]} format."""
    if "instance_info" in obj and isinstance(obj["instance_info"], list) and len(obj["instance_info"]) > 0:
        inst = obj["instance_info"][0]
    else:
        inst = obj

    keypoints = inst.get("keypoints_xyn", [])
    visibility = inst.get("visibility", [])

    # if keypoints flat list -> reshape
    if len(keypoints) > 0 and isinstance(keypoints[0], (int, float)):
        if len(keypoints) % 2 == 0:
            keypoints = [[float(keypoints[i]), float(keypoints[i + 1])] for i in range(0, len(keypoints), 2)]
        else:
            keypoints = []

    # keep only [x,y]
    cleaned_kpt = []
    for kp in keypoints:
        if isinstance(kp, (list, tuple)) and len(kp) >= 2:
            x = float(kp[0])
            y = float(kp[1])
            cleaned_kpt.append([x, y])

    if len(cleaned_kpt) == 0:
        raise ValueError("parsed JSON has no valid keypoints_xyn")

    if not isinstance(visibility, list) or len(visibility) != len(cleaned_kpt):
        # fallback: visible for all
        visibility = [1 for _ in cleaned_kpt]
    else:
        def _to_vis_3val(v):
            try:
                iv = int(float(v))
            except Exception:
                return 1
            if iv > 0:
                return 1
            if iv == 0:
                return 0
            return -1

        visibility = [_to_vis_3val(v) for v in visibility]

    # clip keypoints to [0,1] for downstream drawing safety
    clipped_kpt = [[max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))] for x, y in cleaned_kpt]

    return {
        "instance_info": [
            {
                "visibility": visibility,
                "keypoints_xyn": clipped_kpt,
            }
        ]
    }


def extract_reason_from_obj(obj: Dict) -> str:
    if "instance_info" in obj and isinstance(obj["instance_info"], list) and len(obj["instance_info"]) > 0:
        inst = obj["instance_info"][0]
    else:
        inst = obj
    reason = inst.get("reason", "")
    if reason is None:
        return ""
    return str(reason)


def format_meta_reason_value(reason: str) -> str:
    reason_escaped = reason.replace("\\", "\\\\").replace("'", "\\'")
    return "{\n  'reason': '" + reason_escaped + "'\n}"


def extract_reason_from_existing_output(output_json_path: Path) -> str:
    try:
        with output_json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        raw_text = obj.get("meta", {}).get("raw_text", "")
        if isinstance(raw_text, str) and raw_text.strip():
            parsed = try_parse_obj(raw_text)
            if isinstance(parsed, dict):
                return extract_reason_from_obj(parsed)
    except Exception:
        pass
    return ""


def resolve_meta_output_path(output_dir: Path, meta_output_path: str) -> Path:
    if meta_output_path:
        return Path(meta_output_path)
    return output_dir.parent / "benchmark_vis" / "meta.json"


def resolve_depth_image_path(image_path: Path, depth_image_root: Path, depth_image_suffix: str) -> Path:
    return depth_image_root / f"{image_path.stem}{depth_image_suffix}"


def resize_image_for_inference(image: Image.Image, min_side: int = 1024) -> Image.Image:
    """
    将图片等比例缩小，直到最短边 <= min_side。
    只返回缩放后的图片，因为 Qwen 输出的是相对坐标，不需要 scale 因子还原。
    """
    w, h = image.size
    short_edge = min(w, h)

    if short_edge > min_side:
        scale = min_side / float(short_edge)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return image


def build_messages_for_image(
    image_path: Path,
    prompt: str,
    use_depth: bool = False,
    depth_image_root: Optional[Path] = None,
    depth_image_suffix: str = "_depth.png",
) -> Tuple[List[Dict[str, Any]], tuple[int, int], Optional[Path], List[Image.Image]]:
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    resized_image = resize_image_for_inference(image)
    images = [resized_image]
    user_content = [{"type": "image", "image": resized_image}]
    depth_path = None
    if use_depth:
        if depth_image_root is None:
            raise ValueError("depth_image_root is required when --use-depth is enabled")
        depth_path = resolve_depth_image_path(image_path, depth_image_root, depth_image_suffix)
        if not depth_path.exists():
            raise FileNotFoundError(f"depth image not found: {depth_path}")
        depth_image = resize_image_for_inference(Image.open(depth_path).convert("RGB"))
        images.append(depth_image)
        user_content.append({"type": "image", "image": depth_image})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
    ]
    return messages, image_size, depth_path, images


def build_prediction_result(
    gen_text: str,
    image_path: Path,
    image_size: tuple[int, int],
    depth_path: Optional[Path],
    output_tokens: int,
    response_time_seconds: float,
    backend: str,
) -> Dict:
    parsed = try_parse_obj(gen_text)
    if parsed is None:
        raise ValueError(f"cannot parse model output as JSON: {gen_text[:200]}")

    result = sanitize_instance_info(parsed)
    bbox = extract_bbox_from_obj(parsed, image_size=image_size)
    if bbox is None:
        bbox = extract_trailing_bbox_from_text(gen_text, image_size=image_size)
    if bbox is not None:
        result["instance_info"][0]["bbox"] = bbox
        result["instance_info"][0]["bbox_type"] = "norm1"
    result["reason"] = extract_reason_from_obj(parsed)
    result["meta"] = {
        "image_path": str(image_path),
        "depth_image_path": str(depth_path) if depth_path is not None else "",
        "raw_text": gen_text,
        "usage": {
            "tokenizer": "qwen",
            "backend": backend,
            "counted_tokens": "output_only",
            "output_tokens": output_tokens,
            "total_tokens": output_tokens,
        },
        "total_tokens": output_tokens,
        "response_time_seconds": round(response_time_seconds, 6),
    }
    return result


def generate_prediction(
    model,
    processor,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    use_depth: bool = False,
    depth_image_root: Optional[Path] = None,
    depth_image_suffix: str = "_depth.png",
) -> Dict:
    messages, image_size, depth_path, _ = build_messages_for_image(
        image_path=image_path,
        prompt=prompt,
        use_depth=use_depth,
        depth_image_root=depth_image_root,
        depth_image_suffix=depth_image_suffix,
    )

    model_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                model_inputs[k] = v.to(device=model_device, dtype=model_dtype)
            else:
                model_inputs[k] = v.to(device=model_device)

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.eos_token_id = processor.tokenizer.eos_token_id
    generation_config.pad_token_id = processor.tokenizer.pad_token_id
    generation_config.do_sample = do_sample
    if do_sample:
        generation_config.temperature = temperature
        generation_config.top_p = top_p
        generation_config.top_k = top_k
    else:
        # Reset sampling-only knobs for greedy decoding to avoid warning:
        # \"generation flags are not valid and may be ignored\".
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50

    if model_device.type == "cuda":
        torch.cuda.synchronize(model_device)
    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**model_inputs, generation_config=generation_config)
    if model_device.type == "cuda":
        torch.cuda.synchronize(model_device)
    response_time_seconds = time.perf_counter() - start_time

    input_len = model_inputs["input_ids"].shape[1]
    gen_only_ids = output_ids[:, input_len:]
    gen_text = processor.tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)[0].strip()
    output_tokens = len(processor.tokenizer.encode(gen_text, add_special_tokens=False))
    total_tokens = output_tokens

    return build_prediction_result(
        gen_text=gen_text,
        image_path=image_path,
        image_size=image_size,
        depth_path=depth_path,
        output_tokens=total_tokens,
        response_time_seconds=response_time_seconds,
        backend="transformers",
    )


def _split_gpu_groups(gpus: str, gpus_per_process: int) -> List[str]:
    ids = [x.strip() for x in gpus.split(",") if x.strip()]
    if len(ids) == 0:
        return []
    if gpus_per_process <= 0:
        raise ValueError("--gpus_per_process must be >= 1")
    groups = []
    for i in range(0, len(ids), gpus_per_process):
        chunk = ids[i : i + gpus_per_process]
        if len(chunk) == gpus_per_process:
            groups.append(",".join(chunk))
    return groups


def _build_gen_model(args: argparse.Namespace):
    if args.model_path:
        if args.base_model_path or args.lora_path:
            raise ValueError("Use either --model_path OR (--base_model_path + --lora_path), not both.")
    else:
        if not (args.base_model_path and args.lora_path):
            raise ValueError("Please provide --model_path, or provide both --base_model_path and --lora_path.")

    torch_dtype = pick_dtype(args.dtype)
    if args.model_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        processor_path = args.model_path
    else:
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model_path,
            dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        processor_path = args.base_model_path

    model.eval()
    model.to(device=args.device, dtype=torch_dtype)

    processor = AutoProcessor.from_pretrained(processor_path, fix_mistral_regex=True)
    return model, processor


def pick_vllm_dtype(dtype_str: str) -> str:
    if dtype_str == "bf16":
        return "bfloat16"
    if dtype_str == "fp16":
        return "float16"
    return "float32"


def _build_vllm_model(args: argparse.Namespace, gpu_group: str):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise ImportError("--backend vllm requires vllm to be installed") from exc

    if not args.model_path:
        raise ValueError("For --backend vllm, provide --model_path pointing to a full/merged Qwen3-VL model.")
    if args.base_model_path or args.lora_path:
        raise ValueError("For --backend vllm, LoRA is not loaded dynamically. Merge LoRA first and pass the merged model via --model_path.")
    model_path = args.model_path
    processor_path = args.model_path

    gpu_ids = [gpu_id.strip() for gpu_id in gpu_group.split(",") if gpu_id.strip()]
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    visible_gpu_count = len(gpu_ids) if gpu_ids else max(1, int(args.gpus_per_process))
    tensor_parallel_size = args.vllm_tensor_parallel_size or visible_gpu_count
    if tensor_parallel_size <= 0:
        raise ValueError("--vllm_tensor_parallel_size must be positive when provided")

    llm_kwargs = {
        "model": model_path,
        "tokenizer": processor_path,
        "dtype": pick_vllm_dtype(args.dtype),
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 2 if args.use_depth else 1},
    }
    model = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(processor_path, fix_mistral_regex=True)
    sampling_temperature = args.temperature if args.do_sample else 0.0
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=args.top_p if args.do_sample else 1.0,
        top_k=args.top_k if args.do_sample else 0,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )
    return model, processor, sampling_params


def prepare_vllm_request(
    processor,
    image_path: Path,
    prompt: str,
    use_depth: bool,
    depth_image_root: Path,
    depth_image_suffix: str,
) -> Tuple[Dict[str, Any], tuple[int, int], Optional[Path]]:
    messages, image_size, depth_path, images = build_messages_for_image(
        image_path=image_path,
        prompt=prompt,
        use_depth=use_depth,
        depth_image_root=depth_image_root,
        depth_image_suffix=depth_image_suffix,
    )
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_payload: Any = images[0] if len(images) == 1 else images
    return {"prompt": prompt_text, "multi_modal_data": {"image": image_payload}}, image_size, depth_path


def _run_worker_vllm(args_dict: Dict, worker_idx: int, gpu_group: str, image_strs: List[str]) -> tuple[int, int, Dict[str, str], Dict[str, float]]:
    args = argparse.Namespace(**args_dict)
    if args.vllm_batch_size <= 0:
        raise ValueError("--vllm_batch_size must be >= 1")
    model, processor, sampling_params = _build_vllm_model(args, gpu_group)

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    depth_image_root = Path(args.depth_image_root)
    annotation_bbox_map = getattr(args, "annotation_bbox_map", {})
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    vis_meta: Dict[str, str] = {}
    metrics = {
        "total_tokens": 0.0,
        "token_counted": 0.0,
        "total_response_time": 0.0,
    }
    iterator = tqdm(range(0, len(image_strs), args.vllm_batch_size), desc=f"vllm-worker-{worker_idx}", position=worker_idx)
    for batch_start in iterator:
        batch_image_strs = image_strs[batch_start : batch_start + args.vllm_batch_size]
        prepared_items = []
        requests = []
        for image_str in batch_image_strs:
            img_path = Path(image_str)
            image_key = img_path.stem
            out_name = make_output_name(img_path, image_dir, args.filename_mode)
            out_path = output_dir / f"{out_name}.json"
            if out_path.exists() and not args.overwrite:
                reason = extract_reason_from_existing_output(out_path)
                vis_meta[image_key] = format_meta_reason_value(reason)
                continue
            try:
                request, image_size, depth_path = prepare_vllm_request(
                    processor=processor,
                    image_path=img_path,
                    prompt=args.prompt,
                    use_depth=args.use_depth,
                    depth_image_root=depth_image_root,
                    depth_image_suffix=args.depth_image_suffix,
                )
                requests.append(request)
                prepared_items.append(
                    {
                        "img_path": img_path,
                        "image_key": image_key,
                        "out_name": out_name,
                        "out_path": out_path,
                        "image_size": image_size,
                        "depth_path": depth_path,
                    }
                )
            except Exception as exc:
                failed += 1
                print(f"[WARN][worker {worker_idx}] failed to prepare {img_path}: {exc}")
        if not requests:
            continue

        start_time = time.perf_counter()
        try:
            request_outputs = model.generate(
                requests,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except Exception as exc:
            failed += len(prepared_items)
            print(f"[WARN][worker {worker_idx}] vLLM batch failed: {exc}")
            continue
        response_time_seconds = time.perf_counter() - start_time
        per_item_response_time = response_time_seconds / max(1, len(prepared_items))

        for prepared_item, request_output in zip(prepared_items, request_outputs):
            img_path = prepared_item["img_path"]
            try:
                completion = request_output.outputs[0]
                gen_text = str(completion.text).strip()
                token_ids = getattr(completion, "token_ids", None)
                output_tokens = len(token_ids) if token_ids is not None else len(processor.tokenizer.encode(gen_text, add_special_tokens=False))
                out_obj = build_prediction_result(
                    gen_text=gen_text,
                    image_path=img_path,
                    image_size=prepared_item["image_size"],
                    depth_path=prepared_item["depth_path"],
                    output_tokens=output_tokens,
                    response_time_seconds=per_item_response_time,
                    backend="vllm",
                )
                fallback_bbox = lookup_annotation_bbox(annotation_bbox_map, img_path, prepared_item["out_name"])
                inst = out_obj.get("instance_info", [{}])[0]
                if fallback_bbox is not None and isinstance(inst, dict) and inst.get("bbox") is None:
                    inst["bbox"] = fallback_bbox
                    inst["bbox_type"] = "norm1"
                meta = out_obj.get("meta", {})
                cur_tokens = meta.get("total_tokens")
                if isinstance(cur_tokens, int):
                    metrics["total_tokens"] += float(cur_tokens)
                    metrics["token_counted"] += 1.0
                cur_response_time = meta.get("response_time_seconds")
                if isinstance(cur_response_time, (int, float)):
                    metrics["total_response_time"] += float(cur_response_time)
                reason = str(out_obj.pop("reason", "") or "")
                with prepared_item["out_path"].open("w", encoding="utf-8") as output_file:
                    json.dump(out_obj, output_file, ensure_ascii=False, indent=2)
                vis_meta[prepared_item["image_key"]] = format_meta_reason_value(reason)
                ok += 1
            except Exception as exc:
                failed += 1
                print(f"[WARN][worker {worker_idx}] failed on {img_path}: {exc}")
    return ok, failed, vis_meta, metrics


def _run_worker(args_dict: Dict, worker_idx: int, gpu_group: str, image_strs: List[str]) -> tuple[int, int, Dict[str, str], Dict[str, float]]:
    args = argparse.Namespace(**args_dict)
    if args.backend == "vllm":
        return _run_worker_vllm(args_dict, worker_idx, gpu_group, image_strs)
    if gpu_group:
        gpu_ids = [x.strip() for x in gpu_group.split(",") if x.strip()]
        if len(gpu_ids) == 0:
            raise ValueError(f"Invalid gpu_group: {gpu_group}")
        # Pin this worker to a concrete physical gpu id.
        args.device = f"cuda:{gpu_ids[0]}"
    elif str(args.device).startswith("cuda"):
        args.device = "cuda:0"
    model, processor = _build_gen_model(args)

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    depth_image_root = Path(args.depth_image_root)
    annotation_bbox_map = getattr(args, "annotation_bbox_map", {})
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    vis_meta: Dict[str, str] = {}
    metrics = {
        "total_tokens": 0.0,
        "token_counted": 0.0,
        "total_response_time": 0.0,
    }
    iterator = tqdm(image_strs, desc=f"gen-worker-{worker_idx}", position=worker_idx)
    for image_str in iterator:
        img_path = Path(image_str)
        image_key = img_path.stem
        out_name = make_output_name(img_path, image_dir, args.filename_mode)
        out_path = output_dir / f"{out_name}.json"
        if out_path.exists() and not args.overwrite:
            reason = extract_reason_from_existing_output(out_path)
            vis_meta[image_key] = format_meta_reason_value(reason)
            continue

        try:
            out_obj = generate_prediction(
                model=model,
                processor=processor,
                image_path=img_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_depth=args.use_depth,
                depth_image_root=depth_image_root,
                depth_image_suffix=args.depth_image_suffix,
            )
            fallback_bbox = lookup_annotation_bbox(annotation_bbox_map, img_path, out_name)
            inst = out_obj.get("instance_info", [{}])[0]
            if fallback_bbox is not None and isinstance(inst, dict) and inst.get("bbox") is None:
                inst["bbox"] = fallback_bbox
                inst["bbox_type"] = "norm1"
            meta = out_obj.get("meta", {})
            cur_tokens = meta.get("total_tokens")
            if isinstance(cur_tokens, int):
                metrics["total_tokens"] += float(cur_tokens)
                metrics["token_counted"] += 1.0
            cur_response_time = meta.get("response_time_seconds")
            if isinstance(cur_response_time, (int, float)):
                metrics["total_response_time"] += float(cur_response_time)
            reason = str(out_obj.pop("reason", "") or "")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)
            vis_meta[image_key] = format_meta_reason_value(reason)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[WARN][worker {worker_idx}] failed on {img_path}: {e}")
    return ok, failed, vis_meta, metrics


def main():
    args = parse_args()
    if not str(args.prompt).strip():
        args.prompt = MIXTASK_PROMPT if args.mixtask else DEFAULT_PROMPT
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if args.limit > 0:
        images = images[: args.limit]
    elif args.max_images > 0:
        images = images[: args.max_images]
    print(f"Found {len(images)} images under {image_dir}")

    annotation_bbox_map = load_annotation_bbox_map(args.annotation_json)
    if annotation_bbox_map:
        print(f"Loaded {len(annotation_bbox_map)} annotation bbox keys from {args.annotation_json}")

    gpu_groups = _split_gpu_groups(args.gpus, args.gpus_per_process)
    vis_meta_all: Dict[str, str] = {}
    meta_output_path = resolve_meta_output_path(output_dir, args.meta_output_path)
    if len(gpu_groups) <= 1:
        args_dict = vars(args).copy()
        args_dict["annotation_bbox_map"] = annotation_bbox_map
        if len(gpu_groups) == 1:
            ok, failed, vis_meta, metrics = _run_worker(args_dict, 0, gpu_groups[0], [str(p) for p in images])
        else:
            ok, failed, vis_meta, metrics = _run_worker(args_dict, 0, "", [str(p) for p in images])
        vis_meta_all.update(vis_meta)
        meta_output_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_output_path.open("w", encoding="utf-8") as f:
            json.dump(vis_meta_all, f, ensure_ascii=False, indent=2)
        total_tokens = int(metrics.get("total_tokens", 0))
        token_counted = int(metrics.get("token_counted", 0))
        total_response_time = float(metrics.get("total_response_time", 0.0))
        avg_tokens = (total_tokens / token_counted) if token_counted else 0.0
        avg_response_time = (total_response_time / ok) if ok else 0.0
        print(
            "Done. "
            f"success={ok}, failed={failed}, "
            f"total_tokens={total_tokens}, avg_tokens={avg_tokens:.2f}, "
            f"total_response_time={total_response_time:.2f}s, avg_response_time={avg_response_time:.2f}s, "
            f"output_dir={output_dir}, meta_json={meta_output_path}"
        )
        return

    num_workers = len(gpu_groups)
    chunk_size = math.ceil(len(images) / num_workers)
    chunks = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]
    args_dict = vars(args).copy()
    args_dict["annotation_bbox_map"] = annotation_bbox_map
    print(f"Run multiprocessing with {num_workers} workers, gpu_groups={gpu_groups}")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        async_results = []
        for i in range(num_workers):
            subset = chunks[i] if i < len(chunks) else []
            async_results.append(
                pool.apply_async(
                    _run_worker,
                    (args_dict, i, gpu_groups[i], [str(p) for p in subset]),
                )
            )
        total_ok, total_failed = 0, 0
        total_tokens = 0.0
        token_counted = 0.0
        total_response_time = 0.0
        for r in async_results:
            ok, failed, vis_meta, metrics = r.get()
            total_ok += ok
            total_failed += failed
            vis_meta_all.update(vis_meta)
            total_tokens += float(metrics.get("total_tokens", 0.0))
            token_counted += float(metrics.get("token_counted", 0.0))
            total_response_time += float(metrics.get("total_response_time", 0.0))

    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_output_path.open("w", encoding="utf-8") as f:
        json.dump(vis_meta_all, f, ensure_ascii=False, indent=2)
    total_tokens_int = int(total_tokens)
    token_counted_int = int(token_counted)
    avg_tokens = (total_tokens_int / token_counted_int) if token_counted_int else 0.0
    avg_response_time = (total_response_time / total_ok) if total_ok else 0.0
    print(
        "Done. "
        f"success={total_ok}, failed={total_failed}, "
        f"total_tokens={total_tokens_int}, avg_tokens={avg_tokens:.2f}, "
        f"total_response_time={total_response_time:.2f}s, avg_response_time={avg_response_time:.2f}s, "
        f"output_dir={output_dir}, meta_json={meta_output_path}"
    )


if __name__ == "__main__":
    main()
