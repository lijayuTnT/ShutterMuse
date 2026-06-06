#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import io
import json
import os
import random
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


@dataclass
class PairJob:
    idx: int
    pair_id: str
    gt_path: str
    pred_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VLM 自动评测脚本：使用 Gemini + Qwen API 从 3 个维度评估"
            "预测姿势图相对原图(gt姿势)的质量。"
        )
    )
    parser.add_argument("--gt-dir", type=str, default="", help="原图目录（gt 姿势图）。")
    parser.add_argument("--pred-dir", type=str, default="", help="预测图目录。")
    parser.add_argument(
        "--pairs-jsonl",
        type=str,
        default="",
        help=(
            "可选：显式图像配对 jsonl。每行一个对象，至少包含 "
            "{id, gt_path, pred_path}（可用 image_id 或 pair_id 替代 id）。"
        ),
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=".png,.jpg,.jpeg,.bmp,.webp",
        help="仅在 --gt-dir/--pred-dir 模式生效，按 stem 匹配时使用的扩展名。",
    )
    parser.add_argument("--output-jsonl", type=str, required=True, help="逐样本输出 jsonl。")
    parser.add_argument("--summary-json", type=str, default="", help="汇总统计输出 json。")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=10240)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="开启后覆盖已有 output_jsonl 与 summary_json，从头开始打分。",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--disable-pred-overlay",
        action="store_true",
        help="关闭将评测结果写回 pred 图片（默认开启并覆盖 pred 图）。",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="",
        help="可选可视化输出目录。设置后不覆盖 pred 原图，而是写入该目录。",
    )
    parser.add_argument(
        "--gt-eval",
        action="store_true",
        help=(
            "开启后表示评测 GT 数据：只评 interaction + pose_beauty；"
            "consistency/physical 固定满分（2分）。"
        ),
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1536,
        help="发送给 VLM 前的最长边缩放上限，0 表示不缩放。",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        choices=["gemini", "qwen"],
        help="选择使用哪个 VLM 后端打分（单选）。",
    )

    parser.add_argument(
        "--gemini-model",
        type=str,
        default=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-image-native"),
    )
    parser.add_argument(
        "--gemini-base-url",
        type=str,
        default=os.getenv("GEMINI_BASE_URL", "https://models-proxy.stepfun-inc.com/gemini"),
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=os.getenv("GEMINI_API_KEY", ""),
    )

    parser.add_argument(
        "--qwen-model",
        type=str,
        default=os.getenv("QWEN_MODEL_NAME", "qwen3-vl-235b-a22b-instruct"),
    )
    parser.add_argument(
        "--qwen-base-url",
        type=str,
        default=os.getenv("QWEN_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"),
    )
    parser.add_argument(
        "--qwen-api-key",
        type=str,
        default=os.getenv("QWEN_API_KEY", ""),
    )
    return parser.parse_args()


def _build_dimension_prompts() -> Dict[str, str]:
    common = (
        "你是严格的姿势评审员。负责评测模型推荐的姿势是否真的能够被人模仿摆出，17个关键点的位置依次为：鼻子、左眼、右眼、左耳、右耳、"
    "左肩、右肩、左手肘、右肘、左手腕、右腕、左髋、右髋、左膝、右膝、左脚踝、右脚踝。"
        "请只输出一个 JSON 对象，不要输出 markdown，不要输出额外解释。\n"
    )
    return {
        "consistency": (
            common
            + "本轮只评估 consistency（与原图姿势一致性）。\n"
            + "你会收到两张图：第一张是原图(gt姿势)，第二张是模型输出姿势可视化图(骨架图)。骨架图中红色的部分意味着模型认为人物的这一部分被物品遮挡了 \n"
            + "评分规则：\n"
            + "- score=1: 姿势类型完全不一致（如站姿 vs 坐姿）。\n"
            + "- score=2: 姿势类型相同但是姿势细节不同（如都是坐姿，但朝向/位置/肢体细节明显不同）。\n"
            + "- score=3: 姿势类型相同而且姿势细节也一样（姿势、朝向、位置基本一致）。\n"
            + "注意事项：原图是真实人物图，模型输出图是骨架图，你只需要关注骨架图展现出的姿势是否和真实人物图的一致性。\n"
            + "输出格式：{\"score\": 1|2|3, \"reason\": \"简体中文简短说明\"}"
        ),
        "physical": (
            common
            + "本轮只评估 physical（物理合理性）。\n"
            + "你只会收到一张图：模型输出姿势可视化图（骨架图）。骨架图中红色的部分意味着模型认为人物的这一部分被物品遮挡了 \n"
            + "姿势种类：姿势种类分为三种全身姿势，七分身姿势和半身姿势，全身姿势会显示17个keypoints，为了简洁，头部和身体没有进行连接，七分身姿势不会显示左脚踝和右脚踝。半身姿势不会显示下半身。 \n"
            + "评分规则：\n"
            + "- score=1: 模型输出图严重违背物理规律，姿势无法被人模仿摆出。**头部和身体分离不算严重违反物理规律，这是可视化方式导致的。不要因此打1分。** \n"
            + "- score=2: 人物姿势轻微违背物理规律，比如并没有物品遮挡红色部分，或者出现了穿模或者悬空，但仍然能够理解模型推荐的姿势应该怎么摆出来。\n"
            + "- score=3: 环境和人物比例自然，与环境接触可信，无明显悬空/穿模，能被人合理模仿摆出。\n"
            + "注意事项：0、评判不用特别严格，姿势能被理解并模仿就行。 1、人物站立时轻微悬空是正常现象，因为骨架图只显示到脚踝，这种情况下不视为悬空 2、人物腿部交叉并不意味着错误，因为两条腿可能是一前一后 3、人物坐着的时候，脚悬空是合理的 4、**不要把半身或者七分身误判为人体不完整和身体悬空。**\n"
            + "输出格式：{\"score\": 1|2|3, \"reason\": \"简体中文简短说明\"} \n"
            + "输出示例：{\"score\": 2, \"reason\": \"人物头部和身体分离，但这是可视化导致正常现象，人物腿部轻微穿模，但展示的站姿仍然能够理解。\"}"
        ),
        "interaction": (
            common
            + "本轮只评估 interaction（与场景交互程度）。\n"
            + "你只会收到一张图：模型输出姿势可视化图（骨架图）。\n"
            + "姿势种类：姿势种类分为三种全身姿势，七分身姿势和半身姿势，全身姿势会显示17个keypoints，七分身姿势不会输出左脚踝和右脚踝。半身姿势只会不会显示下半身。 \n"
            + "评分规则：\n"
            + "- score=1（弱）: 完全没有与任何事物互动；站立或蹲在地面不算互动。\n"
            + "- score=2（中）: 有简单互动，如坐椅子、靠墙、坐台阶、倚栏杆。\n"
            + "- score=3（强）: 与场景特色物体/景物有明显互动，含肢体或眼神互动，"
            + "例如与小动物、花朵、路牌、景色、景物互动。\n"
            + "输出格式：{\"score\": 1|2|3, \"reason\": \"简体中文简短说明\"}"
        ),
        "pose_beauty": (
            common
            + "本轮只评估 pose_beauty（姿势美观度）。\n"
            + "你只会收到一张图：模型输出姿势可视化图（骨架图）。\n"
            + "姿势种类：姿势种类分为三种全身姿势，七分身姿势和半身姿势，全身姿势会显示17个keypoints，七分身姿势不会输出左脚踝和右脚踝。半身姿势只会不会显示下半身。 \n"
            + "评分规则：\n"
            + "- score=1: 普通静态姿势，即不是动态动作也没有动作细节，如直立站立、正坐。\n"
            + "- score=2: 姿势有一定细节或动态性，如双腿交叉站立、单腿翘起、跷二郎腿、行走、回头看镜头，挥手等。\n"
            + "- score=3: 姿势的动作细节多、趣味性高、张力大，且与环境适配度高。\n"
            + "注意事项：只关注姿势，不要考虑人物长相、服装、构图等非姿势因素。\n"
            + "输出格式：{\"score\": 1|2|3, \"reason\": \"简体中文简短说明\"}"
        ),
    }


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


def _parse_json_obj(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response text")

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


def _safe_resize(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def _img_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    image = image.convert("RGB")
    image.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _norm_score(v: Any, lo: int, hi: int) -> int:
    try:
        iv = int(v)
    except Exception:
        return lo
    return max(lo, min(hi, iv))


def _normalize_dimension_obj(dim: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    lo, hi = 1, 3
    score = _norm_score(obj.get("score", lo), lo, hi)
    # 归一化: 1 -> 0.0, 2 -> 0.5, 3 -> 1.0
    score_norm = (score - lo) / float(hi - lo)
    return {
        "score": score,
        "score_norm": round(score_norm, 6),
        "reason": str(obj.get("reason", "")).strip(),
    }


def _merge_three_dimension_results(
    physical: Dict[str, Any],
    interaction: Dict[str, Any],
    pose_beauty: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "physical": physical,
        "interaction": interaction,
        "pose_beauty": pose_beauty,
        "overall_reason": (
            f"物理合理性{physical['score']}，交互度{interaction['score']}，姿势美观度{pose_beauty['score']}。"
        ),
    }
    out["total_score"] = out["physical"]["score_norm"] + out["interaction"]["score_norm"] + out["pose_beauty"]["score_norm"]
    out["avg_score"] = round(out["total_score"] / 3.0, 6)
    return out


def _call_gemini_dimension(
    client: genai.Client,
    model: str,
    prompt: str,
    images: List[Image.Image],
    max_retries: int,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt, *images],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            raw = _extract_text_from_gemini_response(resp)
            return _parse_json_obj(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(e)
            if attempt < max_retries:
                time.sleep(5)
    raise RuntimeError(f"Gemini call failed after {max_retries} retries: {last_err}")


def _call_qwen_dimension(
    client: OpenAI,
    model: str,
    prompt: str,
    images: List[Image.Image],
    max_retries: int,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, Any]:
    image_blocks = []
    for img in images:
        image_blocks.append({"type": "image_url", "image_url": {"url": _img_to_data_url(img)}})
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "你是严格的姿势评审员，只输出 JSON。"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, *image_blocks],
        },
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            return _parse_json_obj(text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                time.sleep(min(8.0, (1.8**attempt) + random.random()))
    raise RuntimeError(f"Qwen call failed after {max_retries} retries: {last_err}")


def _build_jobs_from_dirs(
    gt_dir: str,
    pred_dir: str,
    image_exts: List[str],
    limit: int,
) -> List[PairJob]:
    gt_root = Path(gt_dir)
    pred_root = Path(pred_dir)
    if not gt_root.exists():
        raise FileNotFoundError(f"--gt-dir not found: {gt_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"--pred-dir not found: {pred_root}")

    gt_map: Dict[str, str] = {}
    for ext in image_exts:
        for p in gt_root.glob(f"*{ext}"):
            gt_map[p.stem] = str(p)

    jobs: List[PairJob] = []
    idx = 0
    for ext in image_exts:
        for pred in sorted(pred_root.glob(f"*{ext}")):
            stem = pred.stem
            gt = gt_map.get(stem)
            if not gt:
                continue
            jobs.append(PairJob(idx=idx, pair_id=stem, gt_path=gt, pred_path=str(pred)))
            idx += 1
            if limit > 0 and len(jobs) >= limit:
                return jobs
    return jobs


def _build_jobs_from_jsonl(path: str, limit: int) -> List[PairJob]:
    jobs: List[PairJob] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            pair_id = (
                obj.get("id")
                or obj.get("pair_id")
                or obj.get("image_id")
                or str(i)
            )
            gt_path = obj.get("gt_path") or obj.get("gt_image") or obj.get("gt_image_path")
            pred_path = obj.get("pred_path") or obj.get("pred_image") or obj.get("pred_image_path")
            if not (isinstance(gt_path, str) and isinstance(pred_path, str)):
                continue
            jobs.append(PairJob(idx=i, pair_id=str(pair_id), gt_path=gt_path, pred_path=pred_path))
            if limit > 0 and len(jobs) >= limit:
                break
    return jobs


def _build_jobs_single_dir(pred_dir: str, image_exts: List[str], limit: int) -> List[PairJob]:
    """
    单目录构建任务（用于 --gt-eval）：把同一张图同时当作 gt/pred 路径占位。
    """
    pred_root = Path(pred_dir)
    if not pred_root.exists():
        raise FileNotFoundError(f"--pred-dir not found: {pred_root}")
    jobs: List[PairJob] = []
    idx = 0
    for ext in image_exts:
        for pred in sorted(pred_root.glob(f"*{ext}")):
            jobs.append(
                PairJob(
                    idx=idx,
                    pair_id=pred.stem,
                    gt_path=str(pred),
                    pred_path=str(pred),
                )
            )
            idx += 1
            if limit > 0 and len(jobs) >= limit:
                return jobs
    return jobs


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
            pid = obj.get("pair_id")
            if isinstance(pid, str) and pid:
                done.add(pid)
    return done


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    count = len(rows)
    sums = {
        "physical": 0.0,      # normalized
        "interaction": 0.0,   # normalized
        "pose_beauty": 0.0,   # normalized
        "total_score": 0.0,
        "avg_score": 0.0,
    }
    for r in rows:
        avg = r["eval"]
        for k in ("physical", "interaction", "pose_beauty"):
            v = avg.get(k, 0)
            if isinstance(v, dict):
                v = v.get("score_norm", 0)
            sums[k] += float(v)
        if "total_score" in avg:
            sums["total_score"] += float(avg["total_score"])
            sums["avg_score"] += float(avg.get("avg_score", 0))
        else:
            cur_total = float(
                (avg.get("physical", {}).get("score_norm", 0) if isinstance(avg.get("physical"), dict) else avg.get("physical", 0))
                + (avg.get("interaction", {}).get("score_norm", 0) if isinstance(avg.get("interaction"), dict) else avg.get("interaction", 0))
                + (avg.get("pose_beauty", {}).get("score_norm", 0) if isinstance(avg.get("pose_beauty"), dict) else avg.get("pose_beauty", 0))
            )
            sums["total_score"] += cur_total
            sums["avg_score"] += float(cur_total / 3.0)
    return {
        "count": count,
        "mean": {k: round(v / count, 6) for k, v in sums.items()},
    }


def _load_summary_rows_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if obj.get("ok") and isinstance(obj.get("eval"), dict):
                rows.append(obj)
    return rows


def _choose_overlay_font() -> Optional[str]:
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            return fp
    return None


def _overlay_eval_on_pred_image(
    pred_path: str,
    eval_obj: Dict[str, Any],
    font_path: Optional[str],
    save_path: Optional[str] = None,
    gt_path: Optional[str] = None,
) -> Optional[str]:
    """
    将各维度分数+理由写到 pred 图上并覆盖保存。
    返回 None 表示成功，否则返回错误字符串。
    """
    try:
        src = Path(pred_path)
        if not src.exists():
            return f"pred image not found: {pred_path}"
        dst = Path(save_path) if save_path else src

        img = Image.open(src).convert("RGB")
        # vis-dir 模式下，若给了 gt_path 且可读，则拼接为 [pred | gt]
        if gt_path:
            gp = Path(gt_path)
            if gp.exists():
                gt_img = Image.open(gp).convert("RGB")
                if gt_img.height != img.height:
                    scale = img.height / float(gt_img.height)
                    new_w = max(1, int(gt_img.width * scale))
                    gt_img = gt_img.resize((new_w, img.height), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (img.width + gt_img.width, img.height))
                canvas.paste(img, (0, 0))
                canvas.paste(gt_img, (img.width, 0))
                img = canvas
        draw = ImageDraw.Draw(img, "RGBA")
        fs = max(14, img.width // 58)
        if font_path:
            font = ImageFont.truetype(font_path, fs)
        else:
            font = ImageFont.load_default()

        order = ["physical", "interaction", "pose_beauty"]
        zh = {
            "physical": "物理合理性",
            "interaction": "交互度",
            "pose_beauty": "姿势美观度",
        }
        lines: List[str] = []
        for k in order:
            v = eval_obj.get(k)
            if isinstance(v, dict):
                score = v.get("score", "")
                score_norm = v.get("score_norm", "")
                reason = str(v.get("reason", "")).strip()
                head = f"{zh.get(k, k)}: {score} (norm={score_norm})"
                lines.append(head)
                if reason:
                    lines.extend(textwrap.wrap(f"理由: {reason}", width=max(18, img.width // max(10, fs))))
        total = eval_obj.get("total_score")
        if total is not None:
            lines.append(f"总分: {total}")
        overall = str(eval_obj.get("overall_reason", "")).strip()
        if overall:
            lines.extend(textwrap.wrap(f"总结: {overall}", width=max(18, img.width // max(10, fs))))

        if not lines:
            return "empty eval object"

        text = "\n".join(lines)
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad = max(4, fs // 4)
        x, y = pad, pad
        draw.rounded_rectangle(
            (x - pad, y - pad, x + tw + pad, y + th + pad),
            radius=6,
            fill=(0, 0, 0, 105),
            outline=(255, 255, 255, 210),
            width=1,
        )
        draw.multiline_text((x, y), text, font=font, fill=(255, 255, 255, 255), spacing=4)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)


def _process_one(
    job: PairJob,
    args: argparse.Namespace,
    prompts: Dict[str, str],
    gemini_client: Optional[genai.Client],
    qwen_client: Optional[OpenAI],
) -> Tuple[int, str, Dict[str, Any]]:
    gt_path = Path(job.gt_path)
    pred_path = Path(job.pred_path)
    if args.gt_eval:
        eval_img_path = pred_path if pred_path.exists() else gt_path
        if not eval_img_path.exists():
            return job.idx, job.pair_id, {"ok": False, "error": f"image not found: {eval_img_path}"}
    else:
        if not gt_path.exists():
            return job.idx, job.pair_id, {"ok": False, "error": f"gt not found: {gt_path}"}
        if not pred_path.exists():
            return job.idx, job.pair_id, {"ok": False, "error": f"pred not found: {pred_path}"}

    if args.dry_run:
        return job.idx, job.pair_id, {"ok": True, "dry_run": True}

    if args.gt_eval:
        eval_img = _safe_resize(Image.open(eval_img_path).convert("RGB"), args.max_image_side)
        gt_img = eval_img
        pred_img = eval_img
    else:
        gt_img = _safe_resize(Image.open(gt_path).convert("RGB"), args.max_image_side)
        pred_img = _safe_resize(Image.open(pred_path).convert("RGB"), args.max_image_side)

    out: Dict[str, Any] = {
        "pair_id": job.pair_id,
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "backend": args.backend,
        "gt_eval": bool(args.gt_eval),
        "ok": True,
    }

    try:
        if not all(k in prompts for k in ("consistency", "physical", "interaction", "pose_beauty")):
            raise RuntimeError("prompts missing required dimensions")

        if args.gt_eval:
            if args.backend == "gemini":
                if gemini_client is None:
                    raise RuntimeError("Gemini client is None")
                interaction_raw = _call_gemini_dimension(
                    client=gemini_client,
                    model=args.gemini_model,
                    prompt=prompts["interaction"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                pose_beauty_raw = _call_gemini_dimension(
                    client=gemini_client,
                    model=args.gemini_model,
                    prompt=prompts["pose_beauty"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            else:
                if qwen_client is None:
                    raise RuntimeError("Qwen client is None")
                interaction_raw = _call_qwen_dimension(
                    client=qwen_client,
                    model=args.qwen_model,
                    prompt=prompts["interaction"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                pose_beauty_raw = _call_qwen_dimension(
                    client=qwen_client,
                    model=args.qwen_model,
                    prompt=prompts["pose_beauty"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            physical = {"score": 3, "reason": "GT评测模式，默认满分。"}
            physical = _normalize_dimension_obj("physical", physical)
        else:
            if args.backend == "gemini":
                if gemini_client is None:
                    raise RuntimeError("Gemini client is None")
                physical_raw = _call_gemini_dimension(
                    client=gemini_client,
                    model=args.gemini_model,
                    prompt=prompts["physical"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                interaction_raw = _call_gemini_dimension(
                    client=gemini_client,
                    model=args.gemini_model,
                    prompt=prompts["interaction"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                pose_beauty_raw = _call_gemini_dimension(
                    client=gemini_client,
                    model=args.gemini_model,
                    prompt=prompts["pose_beauty"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            else:
                if qwen_client is None:
                    raise RuntimeError("Qwen client is None")
                physical_raw = _call_qwen_dimension(
                    client=qwen_client,
                    model=args.qwen_model,
                    prompt=prompts["physical"],
                    images=[gt_img, pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                interaction_raw = _call_qwen_dimension(
                    client=qwen_client,
                    model=args.qwen_model,
                    prompt=prompts["interaction"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                pose_beauty_raw = _call_qwen_dimension(
                    client=qwen_client,
                    model=args.qwen_model,
                    prompt=prompts["pose_beauty"],
                    images=[pred_img],
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            physical = _normalize_dimension_obj("physical", physical_raw)
        interaction = _normalize_dimension_obj("interaction", interaction_raw)
        pose_beauty = _normalize_dimension_obj("pose_beauty", pose_beauty_raw)
        out["eval"] = _merge_three_dimension_results(physical, interaction, pose_beauty)
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = str(e)

    return job.idx, job.pair_id, out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.pairs_jsonl:
        if args.gt_eval:
            if not args.pred_dir:
                raise ValueError("gt_eval 模式下请提供 --pred-dir（待评测图目录）")
        else:
            if not args.gt_dir or not args.pred_dir:
                raise ValueError("请提供 --pairs-jsonl，或者同时提供 --gt-dir 与 --pred-dir")

    image_exts = [x.strip().lower() for x in args.image_exts.split(",") if x.strip()]
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    prompts = _build_dimension_prompts()
    if args.pairs_jsonl:
        jobs = _build_jobs_from_jsonl(args.pairs_jsonl, args.limit)
        if args.gt_eval:
            # pairs_jsonl 模式下，若缺 gt_path，用 pred_path 占位，兼容只给单图路径
            fixed_jobs = []
            for j in jobs:
                gt_path = j.gt_path if j.gt_path else j.pred_path
                fixed_jobs.append(PairJob(idx=j.idx, pair_id=j.pair_id, gt_path=gt_path, pred_path=j.pred_path))
            jobs = fixed_jobs
    elif args.gt_eval:
        jobs = _build_jobs_single_dir(args.pred_dir, image_exts, args.limit)
    else:
        jobs = _build_jobs_from_dirs(args.gt_dir, args.pred_dir, image_exts, args.limit)

    if args.overwrite:
        if output_jsonl.exists():
            output_jsonl.unlink()

    if args.resume and not args.overwrite:
        done = _load_done_ids(output_jsonl)
        jobs = [j for j in jobs if j.pair_id not in done]

    if not jobs:
        print("No jobs to process.")
        return

    gemini_client: Optional[genai.Client] = None
    qwen_client: Optional[OpenAI] = None
    if not args.dry_run:
        if args.backend == "gemini":
            if not args.gemini_api_key:
                raise ValueError("缺少 Gemini API key（--gemini-api-key 或 GEMINI_API_KEY）")
            gemini_client = genai.Client(
                http_options={"api_version": "v1alpha", "base_url": args.gemini_base_url},
                api_key=args.gemini_api_key,
            )
        else:
            qwen_client = OpenAI(
                api_key=args.qwen_api_key,
                base_url=args.qwen_base_url,
                http_client=httpx.Client(trust_env=False, timeout=120),
            )

    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    overlay_font = None if args.disable_pred_overlay else _choose_overlay_font()
    vis_dir_path = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir_path is not None:
        vis_dir_path.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("a", encoding="utf-8") as fw:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futures = [
                ex.submit(_process_one, job, args, prompts, gemini_client, qwen_client)
                for job in jobs
            ]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="VLM scoring"):
                idx, pair_id, row = fut.result()
                record = {"idx": idx, "pair_id": pair_id, **row}
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")
                fw.flush()
                if (not args.disable_pred_overlay) and record.get("ok") and isinstance(record.get("eval"), dict):
                    pred_path = str(record.get("pred_path", "")).strip()
                    if pred_path:
                        overlay_save_path = None
                        if vis_dir_path is not None:
                            ext = Path(pred_path).suffix or ".jpg"
                            overlay_save_path = str(vis_dir_path / f"{record.get('pair_id', 'sample')}{ext}")
                        err = _overlay_eval_on_pred_image(
                            pred_path,
                            record["eval"],
                            overlay_font,
                            save_path=overlay_save_path,
                            gt_path=str(record.get("gt_path", "")).strip() if vis_dir_path is not None else None,
                        )
                        if err:
                            print(f"[overlay_warn] {pair_id}: {err}")

    # summary 必须基于完整 output_jsonl（兼容 resume 续跑）
    all_rows = _load_summary_rows_from_jsonl(output_jsonl)
    summary = _summarize(all_rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. output_jsonl={output_jsonl}")
    print(f"Done. summary_json={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
