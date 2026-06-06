#!/usr/bin/env python3
"""Single-image inference for ShutterMuse photographer-side and subject-side tasks."""

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from utils import get_closest_ratio, parse_qwen_bbox, resize_image_for_inference

SUBJECT_PROMPT = (
    "你是一个人像摄影摆姿分析专家，请根据图片进行人像姿势推荐，以json格式给出推荐的人体17个关键点"
    "的相对坐标和是否在画面中可见，17个关键点的位置依次为：鼻子、左眼、右眼、左耳、右耳、"
    "左肩、右肩、左手肘、右肘、左手腕、右腕、左髋、右髋、左膝、右膝、左脚踝、右脚踝。"
)
COCO17_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
    (0, 2), (1, 3), (2, 4),
)


def build_photographer_prompt(image: Image.Image) -> str:
    image_resize = resize_image_for_inference(image)
    prompt_ratio = get_closest_ratio(image_resize.size[0], image_resize.size[1])
    return (
        f"请找出图片中构图最好的区域，请按照{prompt_ratio}的比例输出bounding box，"
        f"并按照(x1,y1),(x2,y2)的格式返回一个bounding box，其中(x1,y1)是左上角的顶点，"
        f"(x2,y2)是右下角的顶点。"
    )


def build_default_instruction(side: str, image: Image.Image) -> str:
    if side == "subject":
        return SUBJECT_PROMPT
    return build_photographer_prompt(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ShutterMuse inference on one image.")
    parser.add_argument("--side", choices=["photographer", "subject"], default="photographer")
    parser.add_argument("--model_path", type=str, required=True, help="Base or merged Qwen-VL model path.")
    parser.add_argument("--lora_path", type=str, default="", help="Optional LoRA adapter path.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--instruction", type=str, default="", help="Prompt. Defaults depend on --side.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for JSON and visualization outputs.")
    parser.add_argument("--output_name", type=str, default="", help="Output file stem. Defaults to timestamp + image stem.")
    parser.add_argument("--max_new_tokens", type=int, default=0, help="Maximum generated tokens. 0 uses side default.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Input tensor device.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True when loading model/processor.")
    parser.add_argument("--no_merge_lora", action="store_true", help="Keep LoRA adapter unmerged after loading.")
    return parser.parse_args()


def load_qwen_model(model_path: str, lora_path: str, trust_remote_code: bool, merge_lora: bool):
    from peft import PeftModel
    from transformers import AutoProcessor
    import transformers

    model_path_lower = model_path.lower()
    if "qwen3.5" in model_path_lower:
        candidate_class_names = ("Qwen3_5ForConditionalGeneration", "Qwen3VLForConditionalGeneration")
    elif "qwen2.5" in model_path_lower or "qwen2_5" in model_path_lower:
        candidate_class_names = ("Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText")
    else:
        candidate_class_names = (
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        )

    model_cls = None
    for class_name in candidate_class_names:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ImportError(f"No supported Qwen-VL model class found in transformers: {candidate_class_names}")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        if merge_lora:
            model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model.eval()
    return model, processor


def apply_chat_template(processor, messages):
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_inference(model, processor, image: Image.Image, instruction: str, max_new_tokens: int, device: str, side: str) -> str:
    from qwen_vl_utils import process_vision_info

    image_for_model = resize_image_for_inference(image)
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image_for_model},
            {"type": "text", "text": instruction},
        ],
    }
    if side == "subject":
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            user_message,
        ]
    else:
        messages = [user_message]

    text = apply_chat_template(processor, messages)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def draw_label(draw: ImageDraw.ImageDraw, width: int, label: str, output_text: str, fill: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, width // 45))
    except Exception:
        font = ImageFont.load_default()
    text_lines = [label]
    if output_text:
        text_lines.append(" ".join(str(output_text).split())[:180])
    text = "\n".join(text_lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    box_w = min(width, bbox[2] - bbox[0] + 16)
    box_h = bbox[3] - bbox[1] + 16
    draw.rectangle([0, 0, box_w, box_h], fill=(255, 255, 255))
    draw.multiline_text((8, 8), text, fill=fill, font=font, spacing=4)


def draw_photographer_visualization(image: Image.Image, pred_bbox: Optional[list], output_text: str, save_path: Path) -> None:
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    width, height = vis_img.size
    if pred_bbox:
        x1, y1, x2, y2 = [float(v) for v in pred_bbox]
        line_width = max(3, int(round(max(width, height) * 0.004)))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
        label = f"pred: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
    else:
        label = "pred: no valid bbox parsed"
    draw_label(draw, width, label, output_text, fill="red")
    vis_img.save(save_path)


def load_subject_helper_module():
    helper_path = Path(__file__).resolve().parents[1] / "subject-side" / "01_run_benchmark_gen.py"
    spec = importlib.util.spec_from_file_location("shuttermuse_subject_single_helper", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load subject helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_subject_record(output_text: str, image_path: Path, image_size: tuple[int, int]) -> Dict[str, Any]:
    helper = load_subject_helper_module()
    try:
        return helper.build_prediction_result(
            gen_text=output_text,
            image_path=image_path,
            image_size=image_size,
            depth_path=None,
            output_tokens=0,
            response_time_seconds=0.0,
            backend="single-image",
        )
    except Exception as exc:
        return {
            "instance_info": [],
            "meta": {"image_path": str(image_path), "raw_text": output_text, "parse_error": str(exc)},
        }


def draw_subject_visualization(image: Image.Image, record: Dict[str, Any], output_text: str, save_path: Path) -> None:
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    width, height = vis_img.size
    instance_info = record.get("instance_info") or []
    instance = instance_info[0] if instance_info and isinstance(instance_info[0], dict) else {}
    keypoints = instance.get("keypoints_xyn") or []
    visibility = instance.get("visibility") or []
    points = []
    for idx, point in enumerate(keypoints):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            points.append(None)
            continue
        visible = visibility[idx] if idx < len(visibility) else 1
        if visible in (-1, 0, False, None):
            points.append(None)
            continue
        points.append((float(point[0]) * width, float(point[1]) * height))

    line_width = max(3, int(round(max(width, height) * 0.004)))
    radius = max(4, int(round(max(width, height) * 0.006)))
    for a, b in COCO17_EDGES:
        if a < len(points) and b < len(points) and points[a] and points[b]:
            draw.line([points[a], points[b]], fill="lime", width=line_width)
    for point in points:
        if point:
            x, y = point
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red", outline="white")

    label = "subject pose"
    if not points or all(point is None for point in points):
        label = "subject pose: no valid keypoints parsed"
    draw_label(draw, width, label, output_text, fill="green")
    vis_img.save(save_path)


def build_output_stem(output_name: str, image_path: Path) -> str:
    if output_name:
        return output_name
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_path.stem}"


def save_json(record: Dict[str, Any], save_path: Path) -> None:
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else (10240 if args.side == "subject" else 512)

    with Image.open(image_path) as image_raw:
        image = ImageOps.exif_transpose(image_raw).convert("RGB")
    instruction = args.instruction or build_default_instruction(args.side, image)

    model, processor = load_qwen_model(
        args.model_path,
        args.lora_path,
        args.trust_remote_code,
        merge_lora=not args.no_merge_lora,
    )
    output_text = run_inference(
        model=model,
        processor=processor,
        image=image,
        instruction=instruction,
        max_new_tokens=max_new_tokens,
        device=args.device,
        side=args.side,
    )

    image_width, image_height = image.size
    stem = build_output_stem(args.output_name, image_path)
    json_path = output_dir / f"{stem}.json"
    vis_path = output_dir / f"{stem}.webp"

    if args.side == "subject":
        record = build_subject_record(output_text, image_path, (image_width, image_height))
        record.setdefault("meta", {})
        record["meta"].update(
            {
                "side": args.side,
                "model_path": args.model_path,
                "lora_path": args.lora_path,
                "instruction": instruction,
                "json_path": str(json_path),
                "visualization_path": str(vis_path),
            }
        )
        save_json(record, json_path)
        draw_subject_visualization(image, record, output_text, vis_path)
    else:
        pred_bbox = parse_qwen_bbox(output_text, image_width, image_height)
        record = {
            "side": args.side,
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "image": str(image_path),
            "image_size": {"width": image_width, "height": image_height},
            "instruction": instruction,
            "output_text": output_text,
            "pred_bbox": pred_bbox,
            "json_path": str(json_path),
            "visualization_path": str(vis_path),
        }
        save_json(record, json_path)
        draw_photographer_visualization(image, pred_bbox, output_text, vis_path)
        print(f"pred_bbox: {pred_bbox}")

    print(f"saved json: {json_path}")
    print(f"saved visualization: {vis_path}")


if __name__ == "__main__":
    main()
