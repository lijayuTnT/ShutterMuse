#!/usr/bin/env python3
import argparse
import ast
import concurrent.futures
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from utils import calculate_iou, image_to_base64

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

COMPOSITION_KNOWLEDGE = """[构图专业知识]
以下是常见的摄影构图手法，请在评审时作为参考依据：
- 居中构图：把主体放在画面中心附近的位置，适用于对称场景或需要突出主体的情况。
- 三分法构图：把主体放在三分线上，或者九宫格辅助线的交点上，使画面更具动态感和视觉张力。
- 对角线构图：在画面中制造出对角线的走向，增强画面纵深感和动感。
- 对称构图：让拍摄主体在画面中形成对称关系，营造稳定、庄重的视觉效果。
- 引导线构图：借助环境中的引导线（道路、栏杆、河流等），把观察者的视线集中到主体上。
- 留白构图：纯色背景或者重复元素占据画面的80%，主体只占画面的一小部分区域。
- 框架构图：用"框架"将主体框起来，比如用镜子、门窗、山洞、树枝等自然框架。
- 三角构图：利用主体的形状或者动作姿态来组合成三角形，增强画面稳定性。
- 前景构图：有一个虚化的物体作为前景，增加画面层次感和空间纵深。
- 远景： 在人物摄影题材中，拍摄主体人物全身入境属于远景。
- 中景： 在人物摄影题材中，拍摄主体人物小腿中部以上或者大腿中部以上入境属于中景。
- 近景： 在人物摄影题材中，拍摄主体人物腰部以上入境，或者是脸部特写属于近景。
以下是常见的构图错误，评审时需重点关注：
- 主体不完整：**关键主体**被裁切掉重要部分（如人物被裁切掉了头顶）。或者在合影的时候没有包含所有合影主体。对于风景或者建筑构图，只要重要部分在画面内，就不算主体不完整，比如雕像的底座没有入镜，拍摄山峰的照片，山底没有入镜等。
- 干扰元素：背景中存在无关的路人、杂物等干扰元素，分散对主体的注意力。
- 画面歪斜：画面出现了明显歪斜，例如歪斜的地面，歪斜的人物, 严重歪斜的画面会导致没有可以推荐的构图。
- 没有留出视觉空间：画面中人物的身体或者视线有明显朝向，但是身体或者视线朝向没有留出适当的空间，常见于三分法构图中。
- 主体不突出：画面太杂乱，找不到明显主体，或者可能的主体会被其他画面元素严重干扰。
"""

GENERAL_EVAL_RULE = """[三档评分规则]
你只需要对“模型输出的构图框”进行打分，分值只能是 0、0.5、1。

- 0分（严重构图问题）：
  构图框中存在一下任意一个问题
  1) 前景杂乱（比如在景点中有杂乱的游客），画面杂乱、主体不突出、像随手拍
  2) 人像构图中，请你观察**红框**是否裁切到了主体人物的重要关节点(脚踝，膝关节，脖子)，底边离关节点很近的情况不算裁切到了关节点，请你不要误判, 只有当红色的边框穿过了关节点才算作裁切到了关节点，如果只是紧贴或者轻微裁切到，不视作严重构图问题。切到了背景路人也不视作严重构图错误。
  3) 裁剪图严重违背了原来的摄影意图，比如原图是想拍摄人景合影，裁剪图只包含人物或者景物，还有原图想要拍摄多人合影，但是裁剪图只包含个别人物。

- 0.5分（仅修复部分原有问题）：
  构图框解决了**部分**“差的原因”所指出的问题（例如重心偏左被修正，但是重心偏下没有被修正），请你一个一个判断每个问题是否被有效解决，如果只解决了部分问题则给0.5分。
- 1分（修复所有原有问题，构图有明显美学提升）：
  在解决了**所有原有问题**的基础上，没有明显的常见构图错误，同时使用有效的构图手法，使画面在视觉重心、节奏、层次、空间关系等方面更美观。

特别注意：
1、 对于裁切到脚情况，只要红框在脚的下方，就不算裁切到脚部了, 同时是允许中景和近景拍摄的， 也就是说裁切到小腿或者大腿是允许的，不应该视作严重问题, 只有关键主体被裁切才算构图错误。
2、 对于主体偏大和画面歪斜这两种问题，通过二次构图难以解决，所以只要模型的输出框解决了除了这两种问题以外的原图的其他问题，也能给1分。
3、 你要检查模型输出的构图框有没有引入新的问题，如果有给0分。比如如果模型为了解决主体偏小问题而新引入了主体不完整或者重心偏移，给0分。
4、 如果原图的主体本身就被原图画框裁切，模型输出的构图解决了主体不完整以外的其他所有问题，给0.5分。
5、 如果原图画面质量优秀，且原图差的原因为空，模型构图框基本保留了原图，给1分,但是如果模型输出框强行裁剪了原图，给0分。
6、 对于重心偏移问题，你要判断模型输出的构图框的裁剪力度是否真的解决了此类问题，如果裁剪力度不够，或者裁剪过度导致重心仍然偏移，视作没有解决此类问题。同时，正常的三分法构图本身就是存在重心偏移的，如果优化后的构图采用了三分构图，不应该视作有重心偏移。
7、 如果原图差的原因为重心偏移问题，但模型基本保留了原图，给0分。
"""


@dataclass
class EvalJob:
    idx: int
    sample_id: str
    image_path: str
    annotation: Dict[str, Any]
    prediction: Dict[str, Any]
    origin_bad_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 VLM 对模型输出的构图框进行三档评分（0/0.5/1）")
    parser.add_argument("--annotation-json", type=str, required=True, help="评测标注 JSON（id -> 标注字符串/字典）")
    parser.add_argument("--eval-json", type=str, default="", help="evaluate_benchmark.py 输出的 result_json")
    parser.add_argument("--image-root", type=str, required=True, help="图片目录")
    parser.add_argument("--output-jsonl", type=str, required=True, help="逐样本输出 jsonl")
    parser.add_argument("--error-jsonl", type=str, default="", help="错误样本输出 jsonl（默认: output-jsonl 同名 .error.jsonl）")
    parser.add_argument("--summary-json", type=str, default="", help="汇总统计输出 json")
    parser.add_argument("--vis-dir", type=str, default="", help="保存可视化拼接图的目录")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--request-timeout-sec", type=float, default=1200.0, help="兼容保留参数（当前不显式设置超时）")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--use-gt-box-overlay",
        action="store_true",
        help="VLM输入图上画GT主框（而非pred框），用于测试GT在当前规则/prompt下的得分。",
    )

    parser.add_argument("--backend", type=str, default="qwen", choices=["qwen", "gemini"])
    parser.add_argument("--qwen-model", type=str, default=os.getenv("QWEN_MODEL_NAME", "qwen3-vl-235b-a22b-instruct"))
    parser.add_argument("--qwen-base-url", type=str, default=os.getenv("QWEN_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"))
    parser.add_argument("--qwen-api-key", type=str, default=os.getenv("QWEN_API_KEY", ""))
    parser.add_argument("--gemini-model", type=str, default=os.getenv("GEMINI_MODEL_NAME", "gemini-3-pro-native"))
    parser.add_argument("--gemini-base-url", type=str, default=os.getenv("GEMINI_BASE_URL", "https://models-proxy.stepfun-inc.com/gemini"))
    parser.add_argument("--gemini-api-key", type=str, default=os.getenv("GEMINI_API_KEY", os.getenv("API_KEY", "")))

    return parser.parse_args()


def _img_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    b64 = image_to_base64(image, format=fmt, quality=quality)
    return f"data:image/jpeg;base64,{b64}"


def _resize_image_max_side(image: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _parse_json_obj(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    l = s.find("{")
    r = s.rfind("}")
    if l >= 0 and r > l:
        obj = json.loads(s[l : r + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"response is not a json object: {s[:200]}")


def _safe_load_ann_value(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return {}
        for fn in (json.loads, ast.literal_eval):
            try:
                obj = fn(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return {}


def _find_image_path(image_root: str, sample_id: str) -> Optional[str]:
    for ext in (".jpg", ".png", ".webp", ".jpeg", ".bmp"):
        p = os.path.join(image_root, sample_id + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(image_root, sample_id)
    if os.path.exists(p):
        return p
    return None


def _crop_by_bbox(image: Image.Image, pred_bbox: Optional[List[float]]) -> Image.Image:
    vis = image.copy().convert("RGB")
    if not pred_bbox or len(pred_bbox) != 4:
        return vis
    try:
        x1, y1, x2, y2 = [float(v) for v in pred_bbox]
    except Exception:
        return vis

    w, h = vis.size
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))

    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)

    l = int(math.floor(left))
    t = int(math.floor(top))
    r = int(math.ceil(right))
    b = int(math.ceil(bottom))

    if r <= l or b <= t:
        return vis
    return vis.crop((l, t, r, b))


def _draw_pred_bbox_on_image(image: Image.Image, pred_bbox: Optional[List[float]]) -> Image.Image:
    vis = image.copy().convert("RGB")
    if not pred_bbox or len(pred_bbox) != 4:
        return vis
    try:
        x1, y1, x2, y2 = [float(v) for v in pred_bbox]
    except Exception:
        return vis

    w, h = vis.size
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))

    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    if right <= left or bottom <= top:
        return vis

    draw = ImageDraw.Draw(vis)
    line_w = max(2, int(round(min(w, h) * 0.005)))
    draw.rectangle([left, top, right, bottom], outline="red", width=line_w)
    return vis


def _draw_single_bbox_on_image(image: Image.Image, bbox: Optional[List[float]]) -> Image.Image:
    vis = image.copy().convert("RGB")
    if not bbox or len(bbox) != 4:
        return vis
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return vis

    w, h = vis.size
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))

    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    if right <= left or bottom <= top:
        return vis

    draw = ImageDraw.Draw(vis)
    line_w = max(2, int(round(min(w, h) * 0.005)))
    draw.rectangle([left, top, right, bottom], outline="red", width=line_w)
    return vis


def _bbox_area_xyxy(bbox: Optional[List[float]]) -> float:
    if not bbox or len(bbox) != 4:
        return -1.0
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return -1.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _normalize_three_level_score(v: Any) -> float:
    if isinstance(v, str):
        s = v.strip()
        if s in {"0", "0.0"}:
            return 0.0
        if s in {"0.5", ".5", "1/2"}:
            return 0.5
        if s in {"1", "1.0"}:
            return 1.0
    try:
        fv = float(v)
    except Exception:
        return 0.0
    if abs(fv - 1.0) < 1e-6:
        return 1.0
    if abs(fv - 0.5) < 1e-6:
        return 0.5
    if abs(fv - 0.0) < 1e-6:
        return 0.0
    if fv >= 0.75:
        return 1.0
    if fv >= 0.25:
        return 0.5
    return 0.0


def _score_to_level(score: float) -> str:
    if score >= 0.999:
        return "L1"
    if score >= 0.499:
        return "L0.5"
    return "L0"


def _build_general_prompt(ann: Dict[str, Any], pred: Dict[str, Any], origin_bad_reason: str = "") -> str:
    img_legend = "你会看到一张图：原图上已用红框标出模型输出的构图框。"

    origin = ann.get("origin", {}) if isinstance(ann, dict) else {}
    origin_good = str(origin.get("好的原因", "") or origin.get("good_reason", "") or "").strip()
    origin_bad = str(origin.get("差的原因", "") or origin.get("bad_reason", "") or "").strip()
    if origin_bad_reason.strip():
        origin_bad = origin_bad_reason.strip()
    model_chain = str(pred.get("output_text", "") or "").strip()

    return (
        "你是严格的摄影构图评审员。你会看到原图及模型输出构图框（红框）叠加后的单张图。"
        "请评估该红框对应构图方案的质量。只输出一个JSON对象，不要输出markdown。\n"
        f"{img_legend}\n\n"
        f"原图好的原因: {origin_good if origin_good else '（空）'}\n"
        f"原图差的原因: {origin_bad if origin_bad else '（空）'}\n"
        f"{GENERAL_EVAL_RULE}\n\n"
        f"{COMPOSITION_KNOWLEDGE}\n\n"
        "输出格式必须严格为："
        "{\"score\":0或0.5或1,\"level\":\"L0或L0.5或L1\",\"reason\":\"简短中文\",\"fixed_origin_bad\":0或1,\"used_composition_technique\":\"可选，简短中文\"}"
    )


def _call_qwen(
    client: OpenAI,
    model: str,
    prompt: str,
    images: List[Image.Image],
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    image_blocks = [{"type": "image_url", "image_url": {"url": _img_to_data_url(img)}} for img in images]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "你是严格评审员，只输出JSON对象。"}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}, *image_blocks]},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            return _parse_json_obj(text)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(min(8.0, (1.8**attempt) + random.random()))
    raise RuntimeError(f"qwen call failed after {max_retries} retries: {last_err}")


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


def _call_gemini(
    client: Any,
    model: str,
    prompt: str,
    images: List[Image.Image],
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if types is not None:
                resp = client.models.generate_content(
                    model=model,
                    contents=[prompt, *images],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
            else:
                resp = client.models.generate_content(
                    model=model,
                    contents=[prompt, *images],
                )
            raw = _extract_text_from_gemini_response(resp)
            return _parse_json_obj(raw)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(min(8.0, (1.8**attempt) + random.random()))
    raise RuntimeError(f"gemini call failed after {max_retries} retries: {last_err}")


def _pick_font(size: int = 20) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text_by_pixel(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return []
    lines: List[str] = []
    cur = ""
    for ch in text:
        candidate = ch if not cur else cur + ch
        l, t, r, b = draw.textbbox((0, 0), candidate, font=font)
        if (r - l) <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
                cur = ch
            else:
                lines.append(candidate)
                cur = ""
    if cur:
        lines.append(cur)
    return lines


def _save_visualization(row: Dict[str, Any], vis_dir: Path) -> None:
    image_path = str(row.get("image_path", "")).strip()
    sample_id = str(row.get("id", "")).strip()
    if not image_path or not sample_id:
        return
    src = Path(image_path)
    if not src.exists():
        return

    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    draw_img = ImageDraw.Draw(img)
    gt_bboxes = row.get("gt_bboxes", [])
    overlay_src = str(row.get("overlay_bbox_source", "")).strip().lower()
    # In gt-overlay mode, always draw exactly one GT box: the largest-area GT.
    # This keeps visualization deterministic and aligned with GT-overlay selection rule.
    if overlay_src == "gt":
        if isinstance(gt_bboxes, list):
            valid_gts = [g for g in gt_bboxes if isinstance(g, list) and len(g) == 4]
            if valid_gts:
                draw_img.rectangle(max(valid_gts, key=_bbox_area_xyxy), outline="blue", width=3)
    elif isinstance(gt_bboxes, list):
        for gt in gt_bboxes:
            if isinstance(gt, list) and len(gt) == 4:
                draw_img.rectangle(gt, outline="blue", width=3)
    pred_bbox = row.get("pred_bbox", [])
    if isinstance(pred_bbox, list) and len(pred_bbox) == 4:
        draw_img.rectangle(pred_bbox, outline="red", width=4)

    w, h = img.size
    font_size = max(20, min(64, int(min(w, h) * 0.04)))
    font = _pick_font(font_size)

    ev = row.get("composition_eval", {}) if isinstance(row.get("composition_eval"), dict) else {}
    reason = str(ev.get("reason", "")).strip()
    origin_good = str(row.get("origin_good", "")).strip()
    origin_bad = str(row.get("origin_bad", "")).strip()

    lines: List[str] = []
    lines.append(f"Score: {ev.get('score', 0)}")
    if origin_good:
        tmp_draw = ImageDraw.Draw(img)
        max_text_w = int(w * 0.82)
        lines.extend(_wrap_text_by_pixel(tmp_draw, f"Origin Good: {origin_good}", font, max_text_w - 24))
    if origin_bad:
        tmp_draw = ImageDraw.Draw(img)
        max_text_w = int(w * 0.82)
        lines.extend(_wrap_text_by_pixel(tmp_draw, f"Origin Bad: {origin_bad}", font, max_text_w - 24))
    if reason:
        tmp_draw = ImageDraw.Draw(img)
        max_text_w = int(w * 0.82)
        lines.extend(_wrap_text_by_pixel(tmp_draw, f"Reason: {reason}", font, max_text_w - 24))

    l, t, r, b = ImageDraw.Draw(img).textbbox((0, 0), "Ag", font=font)
    line_h = max(24, int((b - t) * 1.35))
    pad = max(12, int(font_size * 0.5))
    max_text_w = int(w * 0.82)
    text_h = max(line_h + pad * 2, len(lines) * line_h + pad * 2)

    if lines:
        rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        x0 = pad
        y0 = pad
        overlay_draw.rectangle((x0, y0, x0 + max_text_w, y0 + text_h), fill=(0, 0, 0, 120))
        y = y0 + pad
        for line in lines:
            if y > y0 + text_h - line_h:
                break
            overlay_draw.text((x0 + pad, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_h
        img = Image.alpha_composite(rgba, overlay).convert("RGB")

    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{sample_id}.jpg"
    img.save(out_path, quality=92)


def _process_one(
    job: EvalJob,
    args: argparse.Namespace,
    qwen_client: Optional[OpenAI],
    gemini_client: Optional[Any],
) -> Tuple[int, str, Dict[str, Any]]:
    image_path = Path(job.image_path)
    if not image_path.exists():
        return job.idx, job.sample_id, {"ok": False, "error": f"image not found: {image_path}"}

    pred_bbox = job.prediction.get("pred_bbox", [])
    pred_bbox = pred_bbox if isinstance(pred_bbox, list) and len(pred_bbox) == 4 else None

    ann_boxes = job.annotation.get("composition_boxes", []) if isinstance(job.annotation, dict) else []
    origin = job.annotation.get("origin", {}) if isinstance(job.annotation, dict) else {}
    if isinstance(origin, dict):
        origin_good = str(origin.get("好的原因", "") or origin.get("good_reason", "") or "").strip()
        origin_bad = str(origin.get("差的原因", "") or origin.get("bad_reason", "") or "").strip()
        origin_type = str(origin.get("返回原图类型", "") or origin.get("返回构图类型", "") or "").strip()
        secondary_type = str(origin.get("二次构图类型", "") or "").strip()
    else:
        origin_good = ""
        origin_bad = ""
        origin_type = ""
        secondary_type = ""
    if job.origin_bad_reason.strip():
        origin_bad = job.origin_bad_reason.strip()
    need_add_full_image = ("原图构图好" in origin_type) or ("原图构图好" in secondary_type) or ("原图构图一般" in origin_type)

    pred_gt_bboxes = job.prediction.get("gt_bboxes", [])
    if not isinstance(pred_gt_bboxes, list):
        pred_gt_bboxes = []

    gt_box_count = 0
    if isinstance(ann_boxes, list):
        for b in ann_boxes:
            rect = b.get("rect", []) if isinstance(b, dict) else []
            if isinstance(rect, list) and len(rect) == 4:
                gt_box_count += 1
    has_gt_bbox = (gt_box_count > 0) or need_add_full_image
    iou_from_eval = float(job.prediction.get("iou_max", 0.0) or 0.0)

    if not args.use_gt_box_overlay:
        # 硬规则1：iou_max=1.0 直接给1分
        if abs(iou_from_eval - 1.0) < 1e-9:
            return job.idx, job.sample_id, {
                "ok": True,
                "id": job.sample_id,
                "image_path": str(image_path),
                "pred_bbox": pred_bbox if pred_bbox is not None else [],
                "best_gt_bbox": [],
                "gt_bboxes": pred_gt_bboxes,
                "origin_good": origin_good,
                "origin_bad": origin_bad,
                "iou_max": 1.0,
                "composition_eval": {
                    "score": 1.0,
                    "level": "L1",
                    "reason": "硬规则：eval-json中的iou_max为1.0，直接判定为1分。",
                    "fixed_origin_bad": 1,
                    "used_composition_technique": "",
                },
                "raw_eval": {"rule_based": True, "rule": "iou_max==1.0"},
            }

        # 硬规则2：gt有框但模型没有输出框 -> 0分
        if has_gt_bbox and pred_bbox is None:
            return job.idx, job.sample_id, {
                "ok": True,
                "id": job.sample_id,
                "image_path": str(image_path),
                "pred_bbox": [],
                "best_gt_bbox": [],
                "gt_bboxes": pred_gt_bboxes,
                "origin_good": origin_good,
                "origin_bad": origin_bad,
                "iou_max": round(iou_from_eval, 6),
                "composition_eval": {
                    "score": 0.0,
                    "level": "L0",
                    "reason": "硬规则：gt_bboxes不为空，但模型没有输出构图框，记0分。",
                    "fixed_origin_bad": 0,
                    "used_composition_technique": "",
                },
                "raw_eval": {"rule_based": True, "rule": "gt_exists_and_pred_missing"},
            }

        # 硬规则3：gt无框但模型输出了框 -> 0分
        if (not has_gt_bbox) and pred_bbox is not None:
            return job.idx, job.sample_id, {
                "ok": True,
                "id": job.sample_id,
                "image_path": str(image_path),
                "pred_bbox": pred_bbox,
                "best_gt_bbox": [],
                "gt_bboxes": pred_gt_bboxes,
                "origin_good": origin_good,
                "origin_bad": origin_bad,
                "iou_max": round(iou_from_eval, 6),
                "composition_eval": {
                    "score": 0.0,
                    "level": "L0",
                    "reason": "硬规则：gt_bboxes为空，但模型输出了构图框，记0分。",
                    "fixed_origin_bad": 0,
                    "used_composition_technique": "",
                },
                "raw_eval": {"rule_based": True, "rule": "gt_empty_and_pred_exists"},
            }

        # 硬规则4：gt无框且模型也无框 -> 1分
        if (not has_gt_bbox) and pred_bbox is None:
            return job.idx, job.sample_id, {
                "ok": True,
                "id": job.sample_id,
                "image_path": str(image_path),
                "pred_bbox": [],
                "best_gt_bbox": [],
                "gt_bboxes": pred_gt_bboxes,
                "origin_good": origin_good,
                "origin_bad": origin_bad,
                "iou_max": round(iou_from_eval, 6),
                "composition_eval": {
                    "score": 1.0,
                    "level": "L1",
                    "reason": "硬规则：gt_bboxes为空且模型未输出构图框，记1分。",
                    "fixed_origin_bad": 1,
                    "used_composition_technique": "",
                },
                "raw_eval": {"rule_based": True, "rule": "gt_empty_and_pred_missing"},
            }

    if args.dry_run:
        return job.idx, job.sample_id, {
            "ok": True,
            "id": job.sample_id,
            "image_path": str(image_path),
            "pred_bbox": pred_bbox,
            "gt_bboxes": pred_gt_bboxes,
            "origin_good": origin_good,
            "origin_bad": origin_bad,
            "dry_run": True,
        }

    if args.backend == "qwen" and qwen_client is None:
        return job.idx, job.sample_id, {"ok": False, "error": "qwen client is None"}
    if args.backend == "gemini" and gemini_client is None:
        return job.idx, job.sample_id, {"ok": False, "error": "gemini client is None"}

    try:
        with Image.open(image_path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        return job.idx, job.sample_id, {"ok": False, "error": f"image open failed: {e}"}

    w, h = image.size
    gt_boxes_norm = []
    for b in (job.annotation.get("composition_boxes", []) if isinstance(job.annotation, dict) else []):
        rect = b.get("rect", []) if isinstance(b, dict) else []
        if isinstance(rect, list) and len(rect) == 4:
            gt_boxes_norm.append(rect)
    if need_add_full_image:
        gt_boxes_norm.append([0.0, 0.0, 1.0, 1.0])
    gt_boxes_abs = [[r[0] * w, r[1] * h, r[2] * w, r[3] * h] for r in gt_boxes_norm]
    if args.use_gt_box_overlay and (not gt_boxes_abs):
        return job.idx, job.sample_id, {
            "ok": True,
            "id": job.sample_id,
            "image_path": str(image_path),
            "pred_bbox": pred_bbox if pred_bbox is not None else [],
            "gt_bboxes": [],
            "best_gt_bbox": [],
            "origin_good": origin_good,
            "origin_bad": origin_bad,
            "iou_max": 1.0,
            "skipped": True,
            "skip_reason": "use_gt_box_overlay enabled and gt_bboxes empty",
            "composition_eval": {
                "score": 1.0,
                "level": "L1",
                "reason": "硬规则：use_gt_box_overlay模式下，gt_bboxes为空样本按1分计入汇总。",
                "fixed_origin_bad": 1,
                "used_composition_technique": "",
            },
            "raw_eval": {"rule_based": True, "rule": "gt_overlay_mode_gt_empty_count_as_one"},
        }
    if args.use_gt_box_overlay and gt_boxes_abs:
        # In GT-overlay mode, default to the largest GT box instead of arbitrary first one.
        pred_bbox = max(gt_boxes_abs, key=_bbox_area_xyxy)
    best_gt_bbox: Optional[List[float]] = None
    iou_max = 0.0
    if gt_boxes_abs:
        best_gt_bbox = max(gt_boxes_abs, key=lambda g: calculate_iou(pred_bbox, g))
        iou_max = float(calculate_iou(pred_bbox, best_gt_bbox))

    if args.use_gt_box_overlay and gt_boxes_abs:
        overlay_bbox = pred_bbox
        image_with_box = _draw_single_bbox_on_image(image, overlay_bbox)
    else:
        overlay_bbox = pred_bbox
        image_with_box = _draw_single_bbox_on_image(image, overlay_bbox)
    # VLM 输入前统一缩放，限制最长边，避免超大图导致请求变慢或不稳定
    image_for_vlm = _resize_image_max_side(image_with_box, max_side=1024)

    out: Dict[str, Any] = {
        "ok": True,
        "id": job.sample_id,
        "image_path": str(image_path),
        "pred_bbox": pred_bbox,
        "gt_bboxes": gt_boxes_abs,
        "best_gt_bbox": best_gt_bbox if best_gt_bbox is not None else [],
        "origin_good": origin_good,
        "origin_bad": origin_bad,
        "iou_max": round(iou_max, 6),
        "overlay_bbox_source": "gt" if args.use_gt_box_overlay and gt_boxes_abs else "pred",
    }

    try:
        prompt = _build_general_prompt(job.annotation, job.prediction, job.origin_bad_reason)
        if args.backend == "gemini":
            raw = _call_gemini(
                client=gemini_client,
                model=args.gemini_model, 
                prompt=prompt,
                images=[image_for_vlm],
                max_retries=args.max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            raw = _call_qwen(
                client=qwen_client,
                model=args.qwen_model,
                prompt=prompt,
                images=[image_for_vlm],
                max_retries=args.max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

        score = _normalize_three_level_score(raw.get("score", 0))
        level = _score_to_level(score)
        fixed_origin_bad = 1 if str(raw.get("fixed_origin_bad", "0")).strip() in {"1", "true", "True", "是"} else 0
        reason = str(raw.get("reason", "")).strip()
        technique = str(raw.get("used_composition_technique", "")).strip()

        out["composition_eval"] = {
            "score": score,
            "level": level,
            "reason": reason,
            "fixed_origin_bad": fixed_origin_bad,
            "used_composition_technique": technique,
        }
        out["raw_eval"] = raw
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)

    return job.idx, job.sample_id, out


def _load_done_ids(output_jsonl: Path) -> set:
    done = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            # resume 时仅跳过已成功样本；失败样本(ok=false)允许重跑
            if obj.get("ok") is False:
                continue
            sid = obj.get("id") or obj.get("sample_id")
            if isinstance(sid, str) and sid:
                done.add(sid)
    return done


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}

    count = len(rows)
    score_sum = 0.0
    iou_sum = 0.0
    level_counts = {"L0": 0, "L0.5": 0, "L1": 0}

    for r in rows:
        ev = r.get("composition_eval", {}) or {}
        score = float(ev.get("score", 0))
        level = str(ev.get("level", _score_to_level(score)))
        score_sum += score
        iou_sum += float(r.get("iou_max", 0.0))
        if level not in level_counts:
            level = _score_to_level(score)
        level_counts[level] += 1

    return {
        "count": count,
        "mean": {
            "score": round(score_sum / count, 6),
            "iou_max": round(iou_sum / count, 6),
        },
        "level_ratio": {k: round(v / count, 6) for k, v in level_counts.items()},
        "level_count": level_counts,
    }


def _load_valid_rows(output_jsonl: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not output_jsonl.exists():
        return rows
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not obj.get("ok"):
                continue

            if isinstance(obj.get("composition_eval"), dict):
                rows.append(obj)
                continue

            # Backward compatibility:
            # old details may contain skipped rows (gt-overlay + empty gt) without composition_eval.
            if obj.get("skipped") and str(obj.get("skip_reason", "")).strip() == "use_gt_box_overlay enabled and gt_bboxes empty":
                obj["composition_eval"] = {
                    "score": 1.0,
                    "level": "L1",
                    "reason": "硬规则：use_gt_box_overlay模式下，gt_bboxes为空样本按1分计入汇总。",
                    "fixed_origin_bad": 1,
                    "used_composition_technique": "",
                }
                if "iou_max" not in obj:
                    obj["iou_max"] = 1.0
                rows.append(obj)
    return rows


def _load_jobs(annotation_json: str, eval_json: str, image_root: str) -> List[EvalJob]:
    with open(annotation_json, "r", encoding="utf-8") as f:
        ann_raw = json.load(f)
    with open(eval_json, "r", encoding="utf-8") as f:
        eval_raw = json.load(f)

    details = eval_raw.get("details", []) if isinstance(eval_raw, dict) else []
    jobs: List[EvalJob] = []

    for i, item in enumerate(details):
        sample_id = str(item.get("id", "") or "")
        if not sample_id:
            continue
        if sample_id not in ann_raw:
            continue

        image_path = _find_image_path(image_root, sample_id)
        if not image_path:
            continue

        ann = _safe_load_ann_value(ann_raw.get(sample_id))
        origin = ann.get("origin", {}) if isinstance(ann, dict) else {}
        if isinstance(origin, dict):
            origin_bad_reason = str(origin.get("差的原因", "") or origin.get("bad_reason", "") or "").strip()
        else:
            origin_bad_reason = ""
        jobs.append(
            EvalJob(
                idx=i,
                sample_id=sample_id,
                image_path=image_path,
                annotation=ann,
                prediction=item,
                origin_bad_reason=origin_bad_reason,
            )
        )
    return jobs


def _load_jobs_from_annotation(annotation_json: str, image_root: str, limit: int) -> List[EvalJob]:
    with open(annotation_json, "r", encoding="utf-8") as f:
        ann_raw = json.load(f)

    jobs: List[EvalJob] = []
    for i, (sample_id_raw, ann_v) in enumerate(ann_raw.items()):
        sample_id = str(sample_id_raw or "")
        if not sample_id:
            continue
        image_path = _find_image_path(image_root, sample_id)
        if not image_path:
            continue
        ann = _safe_load_ann_value(ann_v)
        origin = ann.get("origin", {}) if isinstance(ann, dict) else {}
        if isinstance(origin, dict):
            origin_bad_reason = str(origin.get("差的原因", "") or origin.get("bad_reason", "") or "").strip()
        else:
            origin_bad_reason = ""
        jobs.append(
            EvalJob(
                idx=i,
                sample_id=sample_id,
                image_path=image_path,
                annotation=ann,
                prediction={},
                origin_bad_reason=origin_bad_reason,
            )
        )
        if limit > 0 and len(jobs) >= limit:
            break
    return jobs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    error_jsonl = Path(args.error_jsonl) if args.error_jsonl else output_jsonl.with_suffix(".error.jsonl")
    error_jsonl.parent.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.use_gt_box_overlay:
        jobs = _load_jobs_from_annotation(args.annotation_json, args.image_root, args.limit)
    else:
        if not args.eval_json:
            raise ValueError("未开启 --use-gt-box-overlay 时，必须提供 --eval-json")
        jobs = _load_jobs(args.annotation_json, args.eval_json, args.image_root)
    if not jobs:
        print("No jobs to process.")
        return

    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()
    if args.overwrite and error_jsonl.exists():
        error_jsonl.unlink()

    if args.resume and not args.overwrite:
        done = _load_done_ids(output_jsonl)
        jobs = [j for j in jobs if j.sample_id not in done]

    if args.limit > 0 and not args.use_gt_box_overlay:
        jobs = jobs[: args.limit]

    if not jobs:
        print("No jobs to process after resume filtering.")
        return

    qwen_client: Optional[OpenAI] = None
    gemini_client: Optional[Any] = None
    if not args.dry_run:
        if args.backend == "gemini":
            if genai is None:
                raise ImportError("未安装 google-genai，无法使用 Gemini 后端。")
            if not args.gemini_api_key:
                raise ValueError("缺少 Gemini API key（--gemini-api-key 或 GEMINI_API_KEY）")
            gemini_client = genai.Client(
                http_options={"api_version": "v1alpha", "base_url": args.gemini_base_url},
                api_key=args.gemini_api_key,
            )
        else:
            if not args.qwen_api_key:
                raise ValueError("缺少 Qwen API key（--qwen-api-key 或 QWEN_API_KEY）")
            qwen_client = OpenAI(
                api_key=args.qwen_api_key,
                base_url=args.qwen_base_url,
                http_client=httpx.Client(trust_env=False),
            )

    error_count = 0
    ok_count = 0
    with output_jsonl.open("a", encoding="utf-8") as fw, error_jsonl.open("a", encoding="utf-8") as ew:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futures = [ex.submit(_process_one, job, args, qwen_client, gemini_client) for job in jobs]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="VLM scoring"):
                _, _, row = fut.result()
                if row.get("ok") is False:
                    ew.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ew.flush()
                    error_count += 1
                    continue

                fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                fw.flush()
                ok_count += 1
                if vis_dir is not None and row.get("ok") and not row.get("skipped", False):
                    try:
                        _save_visualization(row, vis_dir)
                    except Exception:
                        pass

    all_rows = _load_valid_rows(output_jsonl)
    summary = _summarize(all_rows)
    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. output_jsonl={output_jsonl}")
    print(f"Done. error_jsonl={error_jsonl}")
    print(f"Done. ok_rows={ok_count}, error_rows={error_count}")
    print(f"Done. summary_json={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
