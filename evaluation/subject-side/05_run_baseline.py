#!/usr/bin/env python3
"""Run Gemini image-edit baseline for pose recommendation.

This script edits each input image by adding a mannequin with an aesthetic,
environment-adapted, physically plausible pose.

Per image, it writes:
1) edited image: <edited_dir>/<stem>.png
2) meta json:    <output_dir>/<stem>.json
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

DEFAULT_EDIT_PROMPT = (
    "请编辑这张图片，在场景中添加一个人物mesh，用于拍照姿势推荐。"
    "人物mesh的姿势必须与当前环境高度适配，整体观感美观自然，且符合物理规律。\n"
    "硬性要求：\n"
    "1) 姿势要稳定，不悬空、不穿模，四肢比例自然，关节方向合理。\n"
    "2) 姿势要有审美性和拍照可参考性，避免僵硬、扭曲动作。\n"
    "3) 与环境关系合理（如站立/坐靠在可支撑位置，朝向与视线方向自然）。\n"
    "4) 保持原图场景与构图基本不变，不新增额外人物。\n"
    "5) 输出单张编辑后图像。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini image-edit baseline runner.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="保存每张图的json元信息")
    parser.add_argument("--edit_prompt", type=str, default=DEFAULT_EDIT_PROMPT)
    parser.add_argument("--edited_dir", type=str, default="", help="编辑图保存目录，默认 <output_dir>/edited")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=-1, help="限制测试样例数目；>0 时优先于 --max_images")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", type=str, default="gemini", choices=["gemini", "gpt"])

    parser.add_argument(
        "--gemini_model",
        type=str,
        default=os.getenv("GEMINI_MODEL_NAME", "gemini-3-pro-image-native"),
    )
    parser.add_argument(
        "--gemini_base_url",
        type=str,
        default=os.getenv("GEMINI_BASE_URL", "https://models-proxy.stepfun-inc.com/gemini"),
    )
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--gpt_model", type=str, default=os.getenv("GPT_MODEL_NAME", "gpt-image-2"))
    parser.add_argument(
        "--gpt_base_url",
        type=str,
        default=os.getenv("GPT_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"),
    )
    parser.add_argument("--gpt_api_key", type=str, default=os.getenv("GPT_API_KEY", ""))
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def resize_image_for_inference(image: Image.Image, min_side: int = 1024) -> Image.Image:
    """
    将图片等比例缩小，直到最短边 <= min_side。
    只返回缩放后的图片，调用侧直接将该图送入模型。
    """
    w, h = image.size
    short_edge = min(w, h)

    if short_edge > min_side:
        scale = min_side / float(short_edge)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return image


def extract_api_output_tokens(usage: Dict[str, Any], backend: str) -> Optional[int]:
    if backend == "gpt":
        value = usage.get("output_tokens")
    else:
        value = usage.get("candidates_token_count")

    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _extract_text_from_gemini_response(response: Any) -> str:
    txt = getattr(response, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    chunks: List[str] = []
    parts: List[Any] = []
    if hasattr(response, "parts") and response.parts:
        parts.extend(list(response.parts))
    candidates = getattr(response, "candidates", None)
    if candidates:
        for c in candidates:
            content = getattr(c, "content", None)
            if content is None:
                continue
            c_parts = getattr(content, "parts", None)
            if c_parts:
                parts.extend(list(c_parts))

    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str) and t.strip():
            chunks.append(t.strip())
    return "\n".join(chunks).strip()


def _extract_image_from_gemini_response(response: Any) -> Optional[Image.Image]:
    parts: List[Any] = []
    if hasattr(response, "parts") and response.parts:
        parts.extend(list(response.parts))
    candidates = getattr(response, "candidates", None)
    if candidates:
        for c in candidates:
            content = getattr(c, "content", None)
            if content is None:
                continue
            c_parts = getattr(content, "parts", None)
            if c_parts:
                parts.extend(list(c_parts))

    for part in parts:
        if hasattr(part, "as_image"):
            try:
                img = part.as_image()
                if img is not None:
                    return img.convert("RGB")
            except Exception:
                pass
        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None:
            data = getattr(inline_data, "data", None)
            if data:
                try:
                    if isinstance(data, str):
                        raw = base64.b64decode(data)
                    else:
                        raw = data
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    pass
    return None


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_usage_from_gemini_response(response: Any) -> Dict[str, int]:
    usage = _get_attr_or_key(response, "usage_metadata")
    if usage is None:
        return {}

    keys = (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
    )
    out: Dict[str, int] = {}
    for key in keys:
        value = _get_attr_or_key(usage, key)
        if value is None:
            continue
        try:
            out[key] = int(value)
        except Exception:
            continue
    return out


def _extract_usage_from_openai_response(response: Any) -> Dict[str, Any]:
    usage = _get_attr_or_key(response, "usage")
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump(exclude_none=True)
        return data if isinstance(data, dict) else {}
    if isinstance(usage, dict):
        return dict(usage)
    out: Dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
    ):
        value = getattr(usage, key, None)
        if value is None:
            continue
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        out[key] = value
    return out


def _call_gemini_edit_image(
    client: Any,
    model: str,
    prompt: str,
    image: Image.Image,
    max_retries: int,
) -> Tuple[Image.Image, str, Dict[str, int]]:
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            if types is not None:
                resp = client.models.generate_content(
                    model=model,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048),
                )
            else:
                resp = client.models.generate_content(model=model, contents=[prompt, image])

            out_img = _extract_image_from_gemini_response(resp)
            txt = _extract_text_from_gemini_response(resp)
            usage = _extract_usage_from_gemini_response(resp)
            if out_img is not None:
                return out_img, txt, usage
            last_err = RuntimeError("Gemini response has no image part")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Gemini image edit failed: {last_err}")


def _call_gpt_edit_image(
    client: OpenAI,
    model: str,
    prompt: str,
    image: Image.Image,
    max_retries: int,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    last_err: Optional[Exception] = None
    image_buf = io.BytesIO()
    image.convert("RGB").save(image_buf, format="PNG")
    image_buf.name = "input.png"

    for _ in range(max_retries):
        try:
            image_buf.seek(0)
            resp = client.images.edit(
                model=model,
                image=image_buf,
                prompt=prompt,
                response_format="b64_json",
                size="auto",
            )
            data = getattr(resp, "data", None) or []
            if not data:
                last_err = RuntimeError("GPT image edit response has no data")
                continue
            first = data[0]
            b64_json = _get_attr_or_key(first, "b64_json")
            if not b64_json:
                last_err = RuntimeError("GPT image edit response has no b64_json")
                continue
            raw = base64.b64decode(b64_json)
            out_img = Image.open(io.BytesIO(raw)).convert("RGB")
            revised_prompt = _get_attr_or_key(first, "revised_prompt") or ""
            return out_img, str(revised_prompt), _extract_usage_from_openai_response(resp)
        except Exception as e:
            print(e)
            last_err = e

    raise RuntimeError(f"GPT image edit failed: {last_err}")


def _predict_one(
    image_path: Path,
    output_json_path: Path,
    edited_dir: Path,
    args: argparse.Namespace,
    client: Any,
) -> Dict[str, Any]:
    src = Image.open(image_path).convert("RGB")
    inference_image = resize_image_for_inference(src)
    start_time = time.perf_counter()
    if args.backend == "gemini":
        model = args.gemini_model
        edited_img, raw_text, usage = _call_gemini_edit_image(
            client=client,
            model=model,
            prompt=args.edit_prompt,
            image=inference_image,
            max_retries=args.max_retries,
        )
    else:
        model = args.gpt_model
        edited_img, raw_text, usage = _call_gpt_edit_image(
            client=client,
            model=model,
            prompt=args.edit_prompt,
            image=inference_image,
            max_retries=args.max_retries,
        )
    output_tokens = extract_api_output_tokens(usage, args.backend)
    total_tokens = output_tokens
    response_time_seconds = time.perf_counter() - start_time

    edited_path = edited_dir / f"{image_path.stem}.png"
    edited_img.save(edited_path)

    meta: Dict[str, Any] = {
        "task_mode": "image_edit",
        "meta": {
            "image_path": str(image_path),
            "edited_image_path": str(edited_path),
            "backend": args.backend,
            "model": model,
            "prompt": args.edit_prompt,
            "raw_text": raw_text,
            "usage": usage,
            "token_usage": {
                "source": "api_usage",
                "counted_tokens": "output_only",
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "total_tokens": total_tokens,
            "response_time_seconds": round(response_time_seconds, 6),
        },
    }
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "usage": usage,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "response_time_seconds": response_time_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.backend == "gemini":
        if genai is None:
            raise RuntimeError("google-genai is not available. Please install google-genai.")
        if not args.gemini_api_key:
            raise ValueError("Missing --gemini_api_key (or GEMINI_API_KEY env).")
    elif not args.gpt_api_key:
        raise ValueError("Missing --gpt_api_key (or GPT_API_KEY env).")

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edited_dir = Path(args.edited_dir) if args.edited_dir else (output_dir / "edited")
    edited_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "gemini":
        client = genai.Client(
            api_key=args.gemini_api_key,
            http_options={"api_version": "v1alpha", "base_url": args.gemini_base_url},
        )
    else:
        client = OpenAI(
            api_key=args.gpt_api_key,
            base_url=args.gpt_base_url,
            http_client=httpx.Client(trust_env=False, timeout=120),
        )

    images = list_images(image_dir)
    if args.limit > 0:
        images = images[: args.limit]
    elif args.max_images > 0:
        images = images[: args.max_images]

    tasks: List[Tuple[Path, Path]] = []
    for img_path in images:
        out_json = output_dir / f"{img_path.stem}.json"
        out_img = edited_dir / f"{img_path.stem}.png"
        if (out_json.exists() and out_img.exists()) and not args.overwrite:
            continue
        tasks.append((img_path, out_json))

    print(f"Found {len(images)} images, pending {len(tasks)} ({args.backend} image_edit).")

    lock = threading.Lock()
    ok = 0
    failed = 0
    total_tokens = 0
    token_counted = 0
    total_response_time = 0.0

    def _job(item: Tuple[Path, Path]) -> Tuple[bool, str, str, Dict[str, Any]]:
        img_path, out_json = item
        try:
            metrics = _predict_one(img_path, out_json, edited_dir, args, client)
            return True, img_path.name, "", metrics
        except Exception as e:
            return False, img_path.name, str(e), {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(_job, t) for t in tasks]
        for fu in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="image_edit"):
            succ, name, err, metrics = fu.result()
            with lock:
                if succ:
                    ok += 1
                    cur_tokens = metrics.get("total_tokens")
                    if isinstance(cur_tokens, int):
                        total_tokens += cur_tokens
                        token_counted += 1
                    total_response_time += float(metrics.get("response_time_seconds", 0.0))
                else:
                    failed += 1
                    print(f"[WARN] {name}: {err}")

    avg_tokens = (total_tokens / token_counted) if token_counted else 0.0
    avg_response_time = (total_response_time / ok) if ok else 0.0
    print(
        "Done. "
        f"success={ok}, failed={failed}, "
        f"total_tokens={total_tokens}, avg_tokens={avg_tokens:.2f}, "
        f"total_response_time={total_response_time:.2f}s, avg_response_time={avg_response_time:.2f}s, "
        f"meta_dir={output_dir}, edited_dir={edited_dir}"
    )


if __name__ == "__main__":
    main()
