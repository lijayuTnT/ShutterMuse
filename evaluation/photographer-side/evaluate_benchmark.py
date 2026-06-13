import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import sys
import tempfile
import random
import copy

import pandas as pd
import torch
import torch.multiprocessing as mp
import httpx
from openai import OpenAI
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    check_ratio_success,
    convert_to_custom_format,
    get_closest_ratio,
    calculate_bde,
    calculate_iou,
    load_annotation_data,
    load_train_data,
    parse_qwen_bbox,
    resize_image_for_inference,
)
from baseline_template import PROMPT_TEMPLATE

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

DEFAULT_VENUS_PROMPT = (
    "Please provide the bounding box coordinate of the most visually balanced "
    "and aesthetically pleasing composition area."
)
DEFAULT_INSTRUCTCROP_PROJECT_DIR = os.getenv(
    "INSTRUCTCROP_PROJECT_DIR",
    str(os.path.join(os.path.dirname(__file__), "third_party", "InstructCrop-main")),
)
DEFAULT_INSTRUCTCROP_PROMPT = (
    "Please suggest the best cropping area in this image, and explain the reason."
)

VALID_RATIOS = ("16:9", "9:16", "1:1","3:4","4:3")
USER_PROMPT_TEMPLATES = (
    "<image>推荐一个{ratio}构图。",
)

API_BASELINE_MODEL_ALIASES = {
    "gpt5.5": ("gpt", "gpt-5.5"),
    "gpt5.4": ("gpt", "gpt-5.4"),
    "gemini-3-flash": ("gemini", "gemini-3-flash-native"),
}


def _api_baseline_provider(baseline_name):
    if baseline_name in ("gemini", "gpt", "qwen"):
        return baseline_name
    if baseline_name in API_BASELINE_MODEL_ALIASES:
        return API_BASELINE_MODEL_ALIASES[baseline_name][0]
    if baseline_name.startswith("gpt"):
        return "gpt"
    if baseline_name.startswith("gemini"):
        return "gemini"
    if baseline_name.startswith("qwen"):
        return "qwen"
    return None


def _api_baseline_model_name(args):
    provider = _api_baseline_provider(args.baseline_name)
    alias = API_BASELINE_MODEL_ALIASES.get(args.baseline_name)
    if alias is not None:
        return alias[1]
    if provider == "gpt":
        if args.baseline_name == "gpt":
            return args.gpt_model
        if args.baseline_name.startswith("gpt-"):
            return args.baseline_name
        return args.baseline_name.replace("gpt", "gpt-", 1)
    if provider == "gemini":
        if args.baseline_name == "gemini":
            return args.gemini_model
        return args.baseline_name
    if provider == "qwen":
        if args.baseline_name == "qwen":
            return args.qwen_model
        return args.baseline_name
    return None


def _ratio_value(ratio_str):
    w, h = ratio_str.split(":")
    return float(w) / float(h)


def _bbox_ratio(bbox):
    if not bbox or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return w / h


def _closest_valid_ratio_for_bbox(bbox):
    bbox_ratio = _bbox_ratio(bbox)
    if bbox_ratio is None:
        return None
    return min(
        VALID_RATIOS,
        key=lambda ratio: abs(math.log((bbox_ratio + 1e-6) / (_ratio_value(ratio) + 1e-6))),
    )


def _get_composition_bboxes(item):
    return item.get("composition_bboxes") or []


def _get_ratio_following_target_ratios(item):
    ratios = []
    for bbox in _get_composition_bboxes(item):
        ratio = _closest_valid_ratio_for_bbox(bbox)
        if ratio is not None and ratio not in ratios:
            ratios.append(ratio)
    return ratios


def _build_ratio_following_prompt_text(item):
    """
    返回 (prompt_text, target_ratio_str)
    ratio 根据第一个 composition GT bbox 反归一化后的比例选择最接近的 VALID_RATIOS。
    """
    ratio = item.get("ratio_following_target_ratio")
    if ratio is None:
        ratios = _get_ratio_following_target_ratios(item)
        ratio = ratios[0] if ratios else None
    if ratio is None:
        raise ValueError(f"ratio_following requires non-empty composition_boxes, got id={item.get('id')}")
    template = random.choice(tuple(USER_PROMPT_TEMPLATES))
    return template.format(ratio=ratio), ratio


def _append_ratio_instruction(prompt_text, target_ratio):
    return f"{prompt_text}\nPlease output the bounding box in {target_ratio} aspect ratio."


def _format_prompt_with_ratio(prompt_template, prompt_ratio):
    prompt_template = str(prompt_template or "").strip()
    if prompt_template:
        if "{prompt_ratio}" in prompt_template:
            return prompt_template.format(prompt_ratio=prompt_ratio)
        if "{target_ratio}" in prompt_template:
            return prompt_template.format(target_ratio=prompt_ratio)
        return prompt_template
    return (
        "Please identify the region with the best composition in the image. "
        "Return a bounding box in the format (x1,y1),(x2,y2) with a "
        f"{prompt_ratio} aspect ratio, where (x1,y1) is the top-left vertex and "
        "(x2,y2) is the bottom-right vertex."
    )


def _expand_ratio_following_items(dataset):
    expanded = []
    for item in dataset:
        ratios = _get_ratio_following_target_ratios(item)
        for ratio in ratios:
            new_item = copy.deepcopy(item)
            new_item["source_id"] = item.get("id")
            new_item["ratio_following_target_ratio"] = ratio
            new_item["id"] = f"{item.get('id')}::ratio_{ratio.replace(':', 'x')}"
            expanded.append(new_item)
    return expanded


def _image_ratio_str(width, height):
    return f"{float(width)}:{float(height)}"


def _parse_image_ratio_str(ratio_str):
    try:
        width, height = str(ratio_str).split(":", 1)
        return float(width), float(height)
    except Exception:
        return None, None


def _resize_image_max_side(image, max_side=1024):
    """等比缩放图片到最长边不超过 max_side；返回(缩放后图, x映射系数, y映射系数)。"""
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_side:
        return image, 1.0, 1.0
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    sx = float(w) / float(new_w)
    sy = float(h) / float(new_h)
    return resized, sx, sy


def _map_bbox_back_to_original(pred_bbox, w_ori, h_ori, sx, sy):
    if not pred_bbox or len(pred_bbox) != 4:
        return pred_bbox
    x1, y1, x2, y2 = [float(v) for v in pred_bbox]
    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy
    x1 = max(0.0, min(float(w_ori), x1))
    y1 = max(0.0, min(float(h_ori), y1))
    x2 = max(0.0, min(float(w_ori), x2))
    y2 = max(0.0, min(float(h_ori), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _is_full_image_bbox(bbox, image_width, image_height, tolerance=1e-9):
    if not bbox or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        image_width = float(image_width)
        image_height = float(image_height)
    except Exception:
        return False
    if image_width <= 0 or image_height <= 0:
        return False

    return (
        abs(x1) <= image_width * tolerance
        and abs(y1) <= image_height * tolerance
        and abs(x2 - image_width) <= image_width * tolerance
        and abs(y2 - image_height) <= image_height * tolerance
    )


def _calculate_keep_success(gt_bboxes, pred_bbox, image_width, image_height):
    gt_bboxes = gt_bboxes or []
    gt_keep = bool(gt_bboxes) and all(_is_full_image_bbox(gt, image_width, image_height) for gt in gt_bboxes)
    pred_keep = _is_full_image_bbox(pred_bbox, image_width, image_height)
    return bool(gt_keep and pred_keep)


def _has_keep_gt(gt_bboxes, image_width, image_height):
    gt_bboxes = gt_bboxes or []
    return bool(gt_bboxes) and all(_is_full_image_bbox(gt, image_width, image_height) for gt in gt_bboxes)


def _calculate_bde_list(gt_bboxes, pred_bbox, image_width, image_height):
    if not pred_bbox:
        return [], None
    bde_list = []
    for gt in gt_bboxes or []:
        bde = calculate_bde(pred_bbox, gt, image_width, image_height)
        if bde is not None:
            bde_list.append(bde)
    bde_min = min(bde_list) if bde_list else None
    return bde_list, bde_min


def _should_count_iou(record):
    return (
        isinstance(record, dict)
        and not record.get("is_reject_case", False)
        and not record.get("is_keep_case", False)
        and record.get("iou_max") is not None
    )


def _has_non_reject_marker(output_text):
    return "<non>" in str(output_text or "").lower()


def _gt_decision_type(is_reject_case, gt_bboxes, image_width, image_height):
    if is_reject_case:
        return "reject"
    if _has_keep_gt(gt_bboxes, image_width, image_height):
        return "keep"
    return "crop"


def _pred_decision_type(pred_bbox, image_width, image_height, output_text=None):
    if _has_non_reject_marker(output_text):
        return "reject"
    if not pred_bbox:
        return "reject"
    if _is_full_image_bbox(pred_bbox, image_width, image_height):
        return "keep"
    return "crop"


def _fill_decision_fields(record, image_width, image_height):
    gt_type = _gt_decision_type(
        record.get("is_reject_case", False),
        record.get("gt_bboxes") or [],
        image_width,
        image_height,
    )
    pred_type = _pred_decision_type(
        record.get("pred_bbox"),
        image_width,
        image_height,
        output_text=record.get("output_text"),
    )

    if gt_type == "reject":
        rs = record.get("reject_success", None)
        success = bool(rs) if rs is not None else (pred_type == "reject")
    elif gt_type == "keep":
        ks = record.get("KS", record.get("keep_success", None))
        success = bool(ks) if ks is not None else (pred_type == "keep")
    else:
        success = pred_type == "crop"

    record["decision_type"] = gt_type
    record["pred_decision_type"] = pred_type
    record["decision_success"] = bool(success)
    return bool(success)


def _detect_local_model_family(model_path):
    """Return the local VLM family used by worker_process."""
    model_path_str = str(model_path or "")
    lower_path = model_path_str.lower()
    if "internvl" in lower_path:
        return "internvl"

    config_path = os.path.join(model_path_str, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            arch = " ".join(cfg.get("architectures") or [])
            auto_map = " ".join(str(v) for v in (cfg.get("auto_map") or {}).values())
            marker = f"{arch} {auto_map}".lower()
            if "internvl" in marker:
                return "internvl"
        except Exception:
            pass
    return "qwen"


def _internvl_build_transform(input_size):
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    # Same tiling strategy as the official InternVL3.5 model card.
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    grid_w = target_width // image_size
    for i in range(blocks):
        box = (
            (i % grid_w) * image_size,
            (i // grid_w) * image_size,
            ((i % grid_w) + 1) * image_size,
            ((i // grid_w) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_image_to_pixel_values(image, input_size=448, max_num=12):
    transform = _internvl_build_transform(input_size=input_size)
    images = _internvl_dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=True,
    )
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def _internvl_load_model_and_tokenizer(args):
    from transformers import AutoModel, AutoTokenizer

    common_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    try:
        model = AutoModel.from_pretrained(args.model_path, use_flash_attn=True, **common_kwargs).eval()
    except Exception as e:
        print(f"⚠️ InternVL use_flash_attn=True 加载失败，回退到 use_flash_attn=False: {e}")
        model = AutoModel.from_pretrained(args.model_path, use_flash_attn=False, **common_kwargs).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def _internvl_generate_text(model, tokenizer, image, prompt_text, args):
    pixel_values = _internvl_image_to_pixel_values(
        image,
        input_size=args.internvl_input_size,
        max_num=args.internvl_max_tiles,
    ).to(torch.bfloat16).cuda()
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    question = prompt_text if str(prompt_text).lstrip().startswith("<image>") else f"<image>\n{prompt_text}"
    with torch.no_grad():
        return model.chat(tokenizer, pixel_values, question, generation_config)


def _extract_text_from_gemini_response(response):
    txt = getattr(response, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    chunks = []
    parts = []
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


def _call_gemini_bbox(client, model, prompt_text, image, max_retries=4):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if types is not None:
                resp = client.models.generate_content(
                    model=model,
                    contents=[prompt_text, image],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=15000),
                )
            else:
                resp = client.models.generate_content(model=model, contents=[prompt_text, image])
            return _extract_text_from_gemini_response(resp)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                continue
    raise RuntimeError(f"Gemini call failed after {max_retries} retries: {last_err}")


def _call_gpt_bbox(client, model, prompt_text, image, max_retries=4):
    import base64
    from io import BytesIO

    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a strict composition evaluator. Output bbox only."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                continue
    raise RuntimeError(f"GPT call failed after {max_retries} retries: {last_err}")


def _call_qwen_bbox(client, model, prompt_text, image, max_retries=4):
    import base64
    from io import BytesIO

    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "你是严格的摄影构图评测模型，按用户要求输出构图分析和bbox。"}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=5120,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                continue
    raise RuntimeError(f"Qwen call failed after {max_retries} retries: {last_err}")


def _venus_build_query(tokenizer, image_path, prompt):
    # Venus tokenizer 期望 image 字段是路径字符串
    return tokenizer.from_list_format([{"image": image_path}, {"text": prompt}])


def _venus_load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def _instructcrop_get_dtype(precision):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _instructcrop_import_modules(project_dir):
    project_dir = os.path.abspath(project_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPImageProcessor
    from torchvision import transforms
    from model.text_baseline import LICAForCausalLM, Cropping_LICA, Instruct_Model
    from model.llava import conversation as conversation_lib
    from model.llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from model.llava.mm_utils import tokenizer_image_token

    return {
        "transformers": transformers,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "CLIPImageProcessor": CLIPImageProcessor,
        "transforms": transforms,
        "LICAForCausalLM": LICAForCausalLM,
        "Cropping_LICA": Cropping_LICA,
        "Instruct_Model": Instruct_Model,
        "conversation_lib": conversation_lib,
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "tokenizer_image_token": tokenizer_image_token,
    }


def _instructcrop_find_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                x not in name
                for x in [
                    "vision_tower",
                    "mm_projector",
                    "text_hidden_fcs",
                    "loc_decoder",
                    "composition",
                    "crop",
                ]
            )
            and any(x in name for x in lora_target_modules)
        ):
            lora_module_names.add(name)
    return sorted(lora_module_names)


def _instructcrop_load_model(args):
    modules = _instructcrop_import_modules(args.instructcrop_project_dir)
    transformers = modules["transformers"]
    LICAForCausalLM = modules["LICAForCausalLM"]
    Cropping_LICA = modules["Cropping_LICA"]
    Instruct_Model = modules["Instruct_Model"]
    LoraConfig = modules["LoraConfig"]
    get_peft_model = modules["get_peft_model"]
    conversation_lib = modules["conversation_lib"]
    DEFAULT_IM_START_TOKEN = modules["DEFAULT_IM_START_TOKEN"]
    DEFAULT_IM_END_TOKEN = modules["DEFAULT_IM_END_TOKEN"]

    device = torch.device(args.instructcrop_device)
    dtype = _instructcrop_get_dtype(args.instructcrop_precision)
    torch_dtype = torch.bfloat16

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.instructcrop_base_model_path,
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        empty_init=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[CRP]", special_tokens=True)
    crop_token_idx = tokenizer("[CRP]", add_special_tokens=False).input_ids[0]
    if args.instructcrop_use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model_args = {
        "train_mask_decoder": True,
        "out_dim": args.instructcrop_out_dim,
        "ce_loss_weight": 0,
        "crop_loss_weight": 0,
        "com_loss_weight": 0,
        "crop_token_idx": crop_token_idx,
        "vision_tower": args.instructcrop_vision_tower,
        "use_mm_start_end": args.instructcrop_use_mm_start_end,
    }
    hf_logging = transformers.utils.logging
    old_hf_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model = LICAForCausalLM.from_pretrained(
            args.instructcrop_base_model_path,
            torch_dtype=torch_dtype,
            empty_init=True,
            **model_args,
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

        model.get_model().initialize_vision_modules(model.get_model().config)
        model.get_model().to(dtype=torch_dtype, device=device)
        model.to(dtype=torch_dtype, device=device)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype, device=device)
        model.get_model().mm_projector.to(dtype=torch_dtype, device=device)
        model.get_model().initialize_lisa_modules(model.get_model().config, torch_dtype, device)
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.instructcrop_conv_type]

        if args.instructcrop_lora_r > 0:
            lora_target_modules = _instructcrop_find_linear_layers(
                model,
                args.instructcrop_lora_target_modules.split(","),
            )
            lora_config = LoraConfig(
                r=args.instructcrop_lora_r,
                lora_alpha=args.instructcrop_lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=args.instructcrop_lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        model.resize_token_embeddings(len(tokenizer))
    finally:
        hf_logging.set_verbosity(old_hf_verbosity)

    checkpoint = torch.load(args.instructcrop_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(dtype=dtype, device=device).eval()

    model_cropping = Cropping_LICA(dtype, device).to(dtype=dtype, device=device).eval()
    if "model_cropping_state_dict" in checkpoint:
        model_cropping.load_state_dict(checkpoint["model_cropping_state_dict"], strict=True)
    else:
        print("[InstructCrop] Warning: model_cropping_state_dict not found in main checkpoint.")

    instruct_model = Instruct_Model(dtype, device).to(dtype=dtype, device=device).eval()
    ckpt_instruct = torch.load(args.instructcrop_instruct_ckpt_path, map_location="cpu", weights_only=False)
    if "instruct_model_state_dict" in ckpt_instruct:
        instruct_state = ckpt_instruct["instruct_model_state_dict"]
    elif "model_state_dict" in ckpt_instruct:
        instruct_state = ckpt_instruct["model_state_dict"]
    else:
        instruct_state = ckpt_instruct
    instruct_model.load_state_dict(instruct_state, strict=False)

    clip_processor = modules["CLIPImageProcessor"].from_pretrained(args.instructcrop_vision_tower)
    image_transform = modules["transforms"].Compose(
        [
            modules["transforms"].ToTensor(),
            modules["transforms"].Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return {
        "modules": modules,
        "tokenizer": tokenizer,
        "model": model,
        "model_cropping": model_cropping,
        "instruct_model": instruct_model,
        "clip_processor": clip_processor,
        "image_transform": image_transform,
        "dtype": dtype,
        "device": device,
    }


def _instructcrop_predict(ctx, image, args, prompt_text=None):
    modules = ctx["modules"]
    tokenizer = ctx["tokenizer"]
    model = ctx["model"]
    model_cropping = ctx["model_cropping"]
    instruct_model = ctx["instruct_model"]
    clip_processor = ctx["clip_processor"]
    image_transform = ctx["image_transform"]
    dtype = ctx["dtype"]
    device = ctx["device"]

    raw_image = image.convert("RGB")
    orig_w, orig_h = raw_image.size
    resized_image = raw_image.resize(
        (args.instructcrop_input_size, args.instructcrop_input_size),
        Image.Resampling.LANCZOS,
    )
    images_tensor = image_transform(resized_image).unsqueeze(0).to(device, dtype=dtype)
    images_clip = clip_processor(raw_image, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)

    qs = (
        modules["DEFAULT_IM_START_TOKEN"]
        + modules["DEFAULT_IMAGE_TOKEN"]
        + modules["DEFAULT_IM_END_TOKEN"]
        + "\n"
        + (prompt_text or args.instructcrop_prompt)
    )
    conv = modules["conversation_lib"].conv_templates[args.instructcrop_conv_type].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], "Sure, [CRP] .")
    input_ids = modules["tokenizer_image_token"](
        conv.get_prompt(),
        tokenizer,
        modules["IMAGE_TOKEN_INDEX"],
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    input_dict = {
        "input_ids": input_ids,
        "images": images_tensor,
        "images_clip": images_clip,
        "attention_masks": torch.ones_like(input_ids).long().to(device),
        "offset": torch.tensor([0, 1]).long().to(device),
        "resize_list": [[orig_w, orig_h]],
        "inference": True,
    }

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            images=images_clip,
            max_new_tokens=args.instructcrop_max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
        text_feature = model(**input_dict)

        if len(text_feature.shape) == 2:
            text_feature = text_feature.unsqueeze(1)
        elif len(text_feature.shape) == 1:
            text_feature = text_feature.unsqueeze(0).unsqueeze(0)

        dummy_crop = torch.zeros((1, 4), device=device, dtype=dtype)
        stage1_boxes = model_cropping(images_tensor, text_feature, dummy_crop, inference=True, stage=2)
        final_boxes = instruct_model(images_tensor, stage1_boxes, dummy_crop, text_feature, inference=True)

    output_ids_tensor = output_ids[0]
    output_ids_filtered = output_ids_tensor[output_ids_tensor != modules["IMAGE_TOKEN_INDEX"]]
    text_output = tokenizer.decode(output_ids_filtered, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    if "ASSISTANT: " in text_output:
        text_output = text_output.split("ASSISTANT: ")[-1]

    crop = final_boxes.clone().float()
    im_h, im_w = images_tensor.shape[-2], images_tensor.shape[-1]
    crop[:, 0::2] = crop[:, 0::2] / im_w * orig_w
    crop[:, 1::2] = crop[:, 1::2] / im_h * orig_h
    pred_crop = crop * args.instructcrop_input_size
    pred_crop[:, 0::2] = torch.clamp(pred_crop[:, 0::2], min=0, max=orig_w)
    pred_crop[:, 1::2] = torch.clamp(pred_crop[:, 1::2], min=0, max=orig_h)
    pred_bbox = pred_crop[0].detach().cpu().numpy().astype(float).tolist()
    if pred_bbox[2] <= pred_bbox[0] or pred_bbox[3] <= pred_bbox[1]:
        pred_bbox = None

    return pred_bbox, text_output


def _instructcrop_worker_process(gpu_ids_str, subset_data, args, process_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[InstructCrop {process_idx}] 启动 | 使用 GPU: {gpu_ids_str} | 数据量: {len(subset_data)}")
    try:
        ctx = _instructcrop_load_model(args)
    except Exception as e:
        print(f"[InstructCrop {process_idx}] 模型加载失败: {e}")
        return [], [], []

    local_results, local_ious, local_ratios = [], [], []
    iterator = tqdm(subset_data, desc=f"InstructCrop Proc {process_idx}", position=process_idx) if subset_data else []
    for item in iterator:
        try:
            with Image.open(item["image_path"]) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
            if args.ratio_following:
                instructcrop_prompt, target_ratio = _build_ratio_following_prompt_text(item)
            else:
                target_ratio = get_closest_ratio(image.size[0], image.size[1])
                instructcrop_prompt = _append_ratio_instruction(args.instructcrop_prompt, target_ratio)
            pred_bbox, output_text = _instructcrop_predict(ctx, image, args, prompt_text=instructcrop_prompt)
            record, ratio_val = _evaluate_item_with_pred(
                item,
                pred_bbox,
                output_text,
                args,
                target_ratio_str=target_ratio,
                prompt_text=instructcrop_prompt,
            )
            local_results.append(record)
            if _should_count_iou(record):
                local_ious.append(float(record.get("iou_max", 0.0)))
            if ratio_val is not None:
                local_ratios.append(ratio_val)
        except Exception as e:
            print(f"[InstructCrop {process_idx}] Error on {item.get('id')}: {e}")
            try:
                record, ratio_val = _evaluate_item_with_pred(
                    item,
                    None,
                    f"[instructcrop_error] {e}",
                    args,
                    prompt_text=args.instructcrop_prompt,
                )
                local_results.append(record)
                if _should_count_iou(record):
                    local_ious.append(float(record.get("iou_max", 0.0)))
                if ratio_val is not None:
                    local_ratios.append(ratio_val)
            except Exception as e2:
                print(f"[InstructCrop {process_idx}] Fallback failed on {item.get('id')}: {e2}")
    return local_results, local_ious, local_ratios


def worker_process(gpu_ids_str, subset_data, args, process_idx):
    """
    参数 gpu_ids_str: 例如 "0,1" 或 "2,3"
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    print(f"[Process {process_idx}] 启动 | 使用 GPU: {gpu_ids_str} | 数据量: {len(subset_data)}")
    model_family = _detect_local_model_family(args.model_path)
    print(f"[Process {process_idx}] 本地模型类型: {model_family}")

    try:
        if model_family == "internvl":
            if args.lora_path:
                raise ValueError("InternVL 本地推理暂不支持 --lora_path；请使用已合并权重或不传 lora_path。")
            model, tokenizer = _internvl_load_model_and_tokenizer(args)
            processor = None
            process_vision_info = None
        else:
            from peft import PeftModel
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            try:
                from transformers import Qwen3_5ForConditionalGeneration
            except Exception:
                Qwen3_5ForConditionalGeneration = None

            if "Qwen3.5" in args.model_path and Qwen3_5ForConditionalGeneration is not None:
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype="auto",
                    device_map="auto",
                )
            else:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype="auto",
                    device_map="auto",
                )

            if args.lora_path:
                model = PeftModel.from_pretrained(model, args.lora_path)
                model = model.merge_and_unload()

            processor = AutoProcessor.from_pretrained(args.model_path)
            tokenizer = None
            model.eval()

    except Exception as e:
        print(f"[Process {process_idx}] 模型加载失败: {e}")
        return [], [], []

    local_results = []
    local_ious = []  # 这里存每个样本的 iou_max
    local_ratios = []

    iterator = tqdm(subset_data, desc=f"Proc {process_idx}", position=process_idx) if len(subset_data) > 0 else []

    for item in iterator:
        image_path = item["image_path"]
        gt_bboxes = item["gt_bboxes"]

        try:
            with Image.open(image_path) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
            w_ori, h_ori = image.size
            image_resize = resize_image_for_inference(image)
            ratio_gt = _image_ratio_str(w_ori, h_ori)

            use_prompt_template = args.eval_mode == "model" and not args.lora_path
            if use_prompt_template:
                if args.ratio_following:
                    _, prompt_ratio = _build_ratio_following_prompt_text(item)
                else:
                    prompt_ratio = get_closest_ratio(image.size[0], image.size[1])
                prompt_text = PROMPT_TEMPLATE.format(target_ratio=prompt_ratio)
            elif args.ratio_following:
                prompt_text, prompt_ratio = _build_ratio_following_prompt_text(item)
                # print(prompt_text)
                # print(prompt_ratio)
            else:
                prompt_ratio = get_closest_ratio(image_resize.size[0], image_resize.size[1])
                prompt_text = _format_prompt_with_ratio(args.prompt, prompt_ratio)
            if model_family == "internvl":
                output_text = _internvl_generate_text(model, tokenizer, image_resize, prompt_text, args)
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_resize},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            pred_bbox = parse_qwen_bbox(output_text, w_ori, h_ori)
            if pred_bbox:
                target_ratio_str = prompt_ratio
                is_ratio_succ = check_ratio_success(pred_bbox, target_ratio_str, tolerance=0.1)
                local_ratios.append(1 if is_ratio_succ else 0)
            else:
                is_ratio_succ = "no bbox"
            is_reject_case = item.get("is_reject_case", False)
            is_keep_case = _has_keep_gt(gt_bboxes, w_ori, h_ori)
            keep_success = _calculate_keep_success(gt_bboxes, pred_bbox, w_ori, h_ori)
            bde_list, bde_min = _calculate_bde_list(gt_bboxes, pred_bbox, w_ori, h_ori)

            record = {
                "id": item["id"],
                "ratio": ratio_gt,
                "prompt_ratio": prompt_ratio,
                "ratio_target_for_eval": prompt_ratio,
                "prompt_text": prompt_text,
                "is_ratio_succ": is_ratio_succ,
                "pred_bbox": pred_bbox,
                "gt_bboxes": gt_bboxes,
                "is_reject_case": is_reject_case,
                "is_keep_case": is_keep_case,
                "KS": keep_success,
                "keep_success": keep_success,
                "reject_success": None,
                "output_text": output_text,
                "iou_list": [],
                "iou_max": None,
                "bde_list": [],
                "bde_min": None,
            }

            if is_reject_case:
                vis_img = image.copy()
                draw = ImageDraw.Draw(vis_img)
                if pred_bbox is None:
                    draw.rectangle([0, 0, 0.05, 0.05], outline="red", width=3)
                    reject_success = 1
                else:
                    draw.rectangle(pred_bbox, outline="red", width=3)
                    reject_success = 0

                draw.rectangle([0, 0, 0.05, 0.05], outline="green", width=3)
                if args.output_dir:
                    save_path = os.path.join(args.output_dir, f"{item['id']}.webp")
                    vis_img.save(save_path)

                record["reject_success"] = reject_success
                record["iou_max"] = None
                record["iou_list"] = []
                record["bde_min"] = None
                record["bde_list"] = []
            else:
                if pred_bbox:
                    for gt in gt_bboxes:
                        record["iou_list"].append(calculate_iou(pred_bbox, gt))
                    iou_max = max(record["iou_list"]) if record["iou_list"] else 0.0
                    record["iou_max"] = iou_max
                    record["bde_list"] = bde_list
                    record["bde_min"] = bde_min
                    if _should_count_iou(record):
                        local_ious.append(iou_max)

                    if args.output_dir:
                        vis_img = image.copy()
                        draw = ImageDraw.Draw(vis_img)
                        for gt in gt_bboxes:
                            draw.rectangle(gt, outline="blue", width=2)
                        draw.rectangle(pred_bbox, outline="red", width=3)
                        save_path = os.path.join(args.output_dir, f"{item['id']}.webp")
                        vis_img.save(save_path)
                else:
                    record["iou_max"] = 0.0
                    record["iou_list"] = [0.0]
                    record["bde_min"] = None
                    record["bde_list"] = []
                    if _should_count_iou(record):
                        local_ious.append(0.0)

            _fill_decision_fields(record, w_ori, h_ori)
            local_results.append(record)
        except Exception as e:
            print(f"[Proc {process_idx}] Error: {e}")

    return local_results, local_ious, local_ratios


def _evaluate_item_with_pred(item, pred_bbox, output_text, args, target_ratio_str=None, prompt_text=None):
    image_path = item["image_path"]
    gt_bboxes = item["gt_bboxes"]
    with Image.open(image_path) as image_raw:
        image = ImageOps.exif_transpose(image_raw).convert("RGB")
    ratio_gt = _image_ratio_str(image.size[0], image.size[1])
    ratio_target = target_ratio_str or ratio_gt

    if pred_bbox:
        is_ratio_succ = check_ratio_success(pred_bbox, ratio_target, tolerance=0.1)
        ratio_val = 1 if is_ratio_succ else 0
    else:
        is_ratio_succ = "no bbox"
        ratio_val = None
    is_keep_case = _has_keep_gt(gt_bboxes, image.size[0], image.size[1])
    keep_success = _calculate_keep_success(gt_bboxes, pred_bbox, image.size[0], image.size[1])
    bde_list, bde_min = _calculate_bde_list(gt_bboxes, pred_bbox, image.size[0], image.size[1])

    record = {
        "id": item["id"],
        "ratio": ratio_gt,
        "prompt_ratio": target_ratio_str,
        "ratio_target_for_eval": ratio_target,
        "prompt_text": prompt_text,
        "is_ratio_succ": is_ratio_succ,
        "pred_bbox": pred_bbox,
        "gt_bboxes": gt_bboxes,
        "is_reject_case": item.get("is_reject_case", False),
        "is_keep_case": is_keep_case,
        "KS": keep_success,
        "keep_success": keep_success,
        "reject_success": None,
        "output_text": output_text,
        "iou_list": [],
        "iou_max": None,
        "bde_list": [],
        "bde_min": None,
    }

    is_reject_case = record["is_reject_case"]
    if is_reject_case:
        # API baseline 的拒绝成功以 <non> 占位符为准，避免“无框但未明确拒绝”虚高
        if args.eval_mode == "baseline" and _api_baseline_provider(args.baseline_name) is not None:
            record["reject_success"] = 1 if _has_non_reject_marker(output_text) else 0
        else:
            if pred_bbox is None:
                record["reject_success"] = 1
            else:
                record["reject_success"] = 0
        record["iou_max"] = None
        record["iou_list"] = []
        record["bde_min"] = None
        record["bde_list"] = []
    else:
        if pred_bbox:
            ious = [calculate_iou(pred_bbox, gt) for gt in gt_bboxes]
            record["iou_list"] = ious
            record["iou_max"] = max(ious) if ious else 0.0
            record["bde_list"] = bde_list
            record["bde_min"] = bde_min
        else:
            record["iou_list"] = [0.0]
            record["iou_max"] = 0.0
            record["bde_min"] = None
            record["bde_list"] = []

    _fill_decision_fields(record, image.size[0], image.size[1])

    if args.output_dir:
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)
        for gt in gt_bboxes:
            draw.rectangle(gt, outline="blue", width=2)
        if pred_bbox:
            draw.rectangle(pred_bbox, outline="red", width=3)
        save_path = os.path.join(args.output_dir, f"{item['id']}.webp")
        vis_img.save(save_path)

    return record, ratio_val


def _evaluate_baseline_api(full_dataset, args):
    provider = _api_baseline_provider(args.baseline_name)
    model_name = _api_baseline_model_name(args)

    if provider == "gemini":
        if genai is None:
            raise ImportError("google-genai 未安装，无法运行 Gemini baseline。")
        if not args.gemini_api_key:
            raise ValueError("缺少 Gemini API key。")
        client = genai.Client(
            http_options={"api_version": "v1alpha", "base_url": args.gemini_base_url},
            api_key=args.gemini_api_key,
        )
    elif provider == "gpt":
        if not args.gpt_api_key:
            raise ValueError("缺少 GPT API key。")
        client = OpenAI(
            api_key=args.gpt_api_key,
            base_url=args.gpt_base_url,
            http_client=httpx.Client(trust_env=False, timeout=120),
        )
    elif provider == "qwen":
        if not args.qwen_api_key:
            raise ValueError("缺少 Qwen API key。")
        client = OpenAI(
            api_key=args.qwen_api_key,
            base_url=args.qwen_base_url,
            http_client=httpx.Client(trust_env=False, timeout=120),
        )
    else:
        raise ValueError(f"不支持的 API baseline: {args.baseline_name}")

    def _run_one(item):
        with Image.open(item["image_path"]) as image_raw:
            image = ImageOps.exif_transpose(image_raw).convert("RGB")
        w_ori, h_ori = image.size
        image_for_vlm, sx, sy = _resize_image_max_side(image, max_side=1024)
        if args.ratio_following:
            prompt_text, target_ratio = _build_ratio_following_prompt_text(item)
        else:
            target_ratio = get_closest_ratio(image.size[0], image.size[1])
            prompt_text = PROMPT_TEMPLATE.format(target_ratio=target_ratio)
        if provider == "gemini":
            output_text = _call_gemini_bbox(client, model_name, prompt_text, image_for_vlm, max_retries=args.api_max_retries)
        elif provider == "gpt":
            output_text = _call_gpt_bbox(client, model_name, prompt_text, image_for_vlm, max_retries=args.api_max_retries)
        elif provider == "qwen":
            output_text = _call_qwen_bbox(client, model_name, prompt_text, image_for_vlm, max_retries=args.api_max_retries)
        else:
            raise ValueError(f"不支持的 API baseline: {args.baseline_name}")
        if str(output_text or "").strip() == "":
            raise RuntimeError(f"empty output_text from {args.baseline_name} api")
        pred_bbox_resized = parse_qwen_bbox(output_text, image_for_vlm.size[0], image_for_vlm.size[1])
        pred_bbox = _map_bbox_back_to_original(pred_bbox_resized, w_ori, h_ori, sx, sy)
        record, ratio_val = _evaluate_item_with_pred(
            item,
            pred_bbox,
            output_text,
            args,
            target_ratio_str=target_ratio,
            prompt_text=prompt_text,
        )
        return record, ratio_val

    all_results = []
    all_ious = []
    all_ratios = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.api_max_workers)) as ex:
        futures = [ex.submit(_run_one, item) for item in full_dataset]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Baseline {args.baseline_name}"):
            try:
                rec, ratio_val = fut.result()
                all_results.append(rec)
                if _should_count_iou(rec):
                    all_ious.append(float(rec.get("iou_max", 0.0)))
                if ratio_val is not None:
                    all_ratios.append(ratio_val)
            except Exception as e:
                print(f"[Baseline {args.baseline_name}] Error: {e}")
    return all_results, all_ious, all_ratios


def _venus_worker_process(gpu_ids_str, subset_data, args, process_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    print(f"[Venus {process_idx}] 启动 | 使用 GPU: {gpu_ids_str} | 数据量: {len(subset_data)}")
    try:
        model, tokenizer = _venus_load_model_and_tokenizer(args.model_path)
    except Exception as e:
        print(f"[Venus {process_idx}] 模型加载失败: {e}")
        return [], [], []

    local_results, local_ious, local_ratios = [], [], []
    iterator = tqdm(subset_data, desc=f"Venus Proc {process_idx}", position=process_idx) if subset_data else []
    for item in iterator:
        try:
            if not os.path.exists(item["image_path"]):
                raise FileNotFoundError(f"image not found: {item['image_path']}")
            with Image.open(item["image_path"]) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
            w_ori, h_ori = image.size
            image_for_vlm, sx, sy = _resize_image_max_side(image, max_side=1024)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp_img = tf.name
            try:
                image_for_vlm.save(tmp_img, format="JPEG", quality=95)
                if args.ratio_following:
                    venus_prompt, target_ratio = _build_ratio_following_prompt_text(item)
                else:
                    target_ratio = get_closest_ratio(image.size[0], image.size[1])
                    venus_prompt = _append_ratio_instruction(DEFAULT_VENUS_PROMPT, target_ratio)
                query = _venus_build_query(tokenizer, tmp_img, venus_prompt)
                with torch.no_grad():
                    output_text, _ = model.chat(tokenizer, query=query, history=None)
            finally:
                if os.path.exists(tmp_img):
                    os.remove(tmp_img)

            pred_bbox_resized = parse_qwen_bbox(output_text, image_for_vlm.size[0], image_for_vlm.size[1])
            pred_bbox = _map_bbox_back_to_original(pred_bbox_resized, w_ori, h_ori, sx, sy)
            record, ratio_val = _evaluate_item_with_pred(
                item,
                pred_bbox,
                output_text,
                args,
                target_ratio_str=target_ratio,
                prompt_text=venus_prompt,
            )
            local_results.append(record)
            if _should_count_iou(record):
                local_ious.append(float(record.get("iou_max", 0.0)))
            if ratio_val is not None:
                local_ratios.append(ratio_val)
        except Exception as e:
            # 单样本失败不丢数据，按“未输出框”计入评测
            print(f"[Venus {process_idx}] Error on {item.get('id')}: {e}")
            try:
                record, ratio_val = _evaluate_item_with_pred(item, None, f"[venus_error] {e}", args, prompt_text=venus_prompt if 'venus_prompt' in locals() else None)
                local_results.append(record)
                if _should_count_iou(record):
                    local_ious.append(float(record.get("iou_max", 0.0)))
                if ratio_val is not None:
                    local_ratios.append(ratio_val)
            except Exception as e2:
                print(f"[Venus {process_idx}] Fallback failed on {item.get('id')}: {e2}")
    return local_results, local_ious, local_ratios


def _load_existing_details(result_json_path):
    if not result_json_path or not os.path.exists(result_json_path):
        return []
    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        details = obj.get("details", []) if isinstance(obj, dict) else []
        return details if isinstance(details, list) else []
    except Exception:
        return []


def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.data_format == "train":
        print("📥 正在加载训练集格式 (JSONL)...")
        full_dataset = load_train_data(args.annotation_json, args.image_root)
    else:
        print("📥 正在加载评测集格式 (JSON)...")
        full_dataset = load_annotation_data(args.annotation_json, args.image_root)

    print(f"总数据量: {len(full_dataset)}")
    if len(full_dataset) == 0:
        return

    if args.ratio_following:
        before = len(full_dataset)
        full_dataset = _expand_ratio_following_items(full_dataset)
        print(f"📐 Ratio-following 模式：按 composition_boxes 唯一比例展开 {len(full_dataset)} 条评测样本，来源原样本 {before} 条")
        if len(full_dataset) == 0:
            return

    existing_details = []
    if args.resume and (not args.overwrite):
        existing_details = _load_existing_details(args.result_json)
        done_ids = {str(x.get("id", "")) for x in existing_details if isinstance(x, dict) and x.get("id") is not None}
        if done_ids:
            before = len(full_dataset)
            full_dataset = [x for x in full_dataset if str(x.get("id", "")) not in done_ids]
            print(f"🔁 Resume 模式：已完成 {before - len(full_dataset)} 条，待处理 {len(full_dataset)} 条")

    if args.limit and args.limit > 0:
        before = len(full_dataset)
        full_dataset = full_dataset[: int(args.limit)]
        print(f"🧪 Limit 模式：从 {before} 条截断到 {len(full_dataset)} 条")

    if len(full_dataset) == 0 and existing_details:
        print("✅ 无待处理样本，直接使用已有结果重算指标。")

    all_results = []
    all_ious = []
    all_ratios = []

    if len(full_dataset) > 0:
        if args.eval_mode == "baseline" and _api_baseline_provider(args.baseline_name) is not None:
            all_results, all_ious, all_ratios = _evaluate_baseline_api(full_dataset, args)
        else:
        # model 模式、以及 baseline-venus 都走本地模型多进程
            all_gpus = [g for g in args.gpus.split(",") if g]
            gpus_per_proc = args.gpus_per_process

            if len(all_gpus) < gpus_per_proc:
                print(f"❌ 错误: 指定了 {len(all_gpus)} 张卡，但要求每个进程 {gpus_per_proc} 张卡。")
                return

            gpu_groups = []
            for i in range(0, len(all_gpus), gpus_per_proc):
                group = all_gpus[i: i + gpus_per_proc]
                if len(group) == gpus_per_proc:
                    gpu_groups.append(",".join(group))

            num_processes = len(gpu_groups)
            print(f"🚀 将启动 {num_processes} 个进程，每个进程使用 GPU: {gpu_groups}")

            chunk_size = math.ceil(len(full_dataset) / num_processes)
            chunks = [full_dataset[i:i + chunk_size] for i in range(0, len(full_dataset), chunk_size)]

            with mp.Pool(processes=num_processes) as pool:
                tasks = []
                for i in range(num_processes):
                    subset = chunks[i] if i < len(chunks) else []
                    tasks.append((gpu_groups[i], subset, args, i))

                if args.eval_mode == "baseline" and args.baseline_name == "venus":
                    results_async = [pool.apply_async(_venus_worker_process, t) for t in tasks]
                elif args.eval_mode == "baseline" and args.baseline_name == "instructcrop":
                    results_async = [pool.apply_async(_instructcrop_worker_process, t) for t in tasks]
                else:
                    if not args.model_path:
                        raise ValueError("model 模式需要提供 --model_path")
                    results_async = [pool.apply_async(worker_process, t) for t in tasks]

                for res in results_async:
                    res_data, res_ious, res_ratios = res.get()
                    all_results.extend(res_data)
                    all_ious.extend(res_ious)
                    all_ratios.extend(res_ratios)

            if len(all_results) == 0:
                raise RuntimeError("本地模型评测没有生成任何样本结果，请检查前面的模型加载或推理错误。")

    # resume 时合并新旧结果（按 id 去重，新的覆盖旧的）
    if existing_details:
        merged = {str(x.get("id", "")): x for x in existing_details if isinstance(x, dict) and x.get("id") is not None}
        for x in all_results:
            merged[str(x.get("id", ""))] = x
        all_results = list(merged.values())

    # 指标统一基于 all_results 重算，保证 resume 一致
    iou_rows = [x for x in all_results if _should_count_iou(x)]
    all_ious = [
        float((x.get("iou_max", 0.0) or 0.0))
        for x in iou_rows
    ]
    bde_rows = iou_rows
    all_bdes = []
    for x in bde_rows:
        bde_min = x.get("bde_min", None)
        bde_list = x.get("bde_list", None)
        if bde_min is None:
            w, h = _parse_image_ratio_str(x.get("ratio", ""))
            if w is not None and h is not None:
                bde_list, bde_min = _calculate_bde_list(
                    x.get("gt_bboxes") or [],
                    x.get("pred_bbox"),
                    w,
                    h,
                )
                x["bde_list"] = bde_list
                x["bde_min"] = bde_min
        if bde_min is not None:
            all_bdes.append(float(bde_min))
    reject_rows = [x for x in all_results if isinstance(x, dict) and x.get("is_reject_case", False)]
    reject_success_vals = []
    for x in reject_rows:
        rs = x.get("reject_success", None)
        if rs is None:
            pb = x.get("pred_bbox", None)
            rs = 1 if (not pb) else 0
        reject_success_vals.append(1 if bool(rs) else 0)

    keep_success_vals = []
    for x in all_results:
        if not isinstance(x, dict):
            continue
        w, h = _parse_image_ratio_str(x.get("ratio", ""))
        if w is None or h is None:
            continue
        is_keep_case = x.get("is_keep_case", None)
        if is_keep_case is None:
            is_keep_case = _has_keep_gt(x.get("gt_bboxes") or [], w, h)
            x["is_keep_case"] = is_keep_case
        if not is_keep_case:
            continue
        ks = x.get("KS", x.get("keep_success", None))
        if ks is None:
            ks = _calculate_keep_success(x.get("gt_bboxes") or [], x.get("pred_bbox"), w, h)
            x["KS"] = ks
            x["keep_success"] = ks
        keep_success_vals.append(1 if bool(ks) else 0)

    decision_success_vals = []
    for x in all_results:
        if not isinstance(x, dict):
            continue
        w, h = _parse_image_ratio_str(x.get("ratio", ""))
        if w is None or h is None:
            continue
        decision_success_vals.append(1 if _fill_decision_fields(x, w, h) else 0)

    all_ratios = []
    for x in all_results:
        if not isinstance(x, dict):
            continue
        rs = x.get("is_ratio_succ", None)
        if rs is True:
            all_ratios.append(1)
        elif rs is False:
            all_ratios.append(0)

    avg_iou_max = sum(all_ious) / len(all_ious) if all_ious else 0.0
    avg_bde = sum(all_bdes) / len(all_bdes) if all_bdes else 0.0
    ratio_succ_avg = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0
    reject_success_avg = (sum(reject_success_vals) / len(reject_success_vals)) if reject_success_vals else 0.0
    keep_success_avg = (sum(keep_success_vals) / len(keep_success_vals)) if keep_success_vals else 0.0
    decision_success_avg = (sum(decision_success_vals) / len(decision_success_vals)) if decision_success_vals else 0.0

    total_samples = len(all_results)
    iou_sample_count = len(all_ious)
    bde_sample_count = len(all_bdes)
    reject_sample_count = len(reject_success_vals)
    keep_sample_count = len(keep_success_vals)
    decision_sample_count = len(decision_success_vals)
    if iou_sample_count > 0:
        acc_max_08 = sum(1 for iou in all_ious if iou >= 0.8) / iou_sample_count
        acc_max_07 = sum(1 for iou in all_ious if iou >= 0.7) / iou_sample_count
    else:
        acc_max_08 = 0.0
        acc_max_07 = 0.0

    print("-" * 40)
    print(f"📊 全流程测试完成 (Total: {total_samples})")
    print(f"🧾 IoU样本数(排除reject/keep): {iou_sample_count} | Reject样本数: {reject_sample_count}")
    print(f"✅ Avg IoU (Max) : {avg_iou_max:.4f}")
    print(f"📏 Avg BDE (Min) : {avg_bde:.4f}")
    print(f"🎯 Acc (Max >= 0.8): {acc_max_08:.2%}")
    print(f"🎯 Acc (Max >= 0.7): {acc_max_07:.2%}")
    print(f"🛑 Reject Success : {reject_success_avg:.2%}")
    print(f"🖼️ Keep Success   : {keep_success_avg:.2%}")
    print(f"🧭 Decision Success: {decision_success_avg:.2%}")
    print(f"📐 Ratio Success   : {ratio_succ_avg:.2%}")
    print("-" * 40)

    if args.result_json:
        all_results.sort(key=lambda x: x["id"])
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {
                        "avg_iou_max": avg_iou_max,
                        "avg_bde": avg_bde,
                        "ratio_succ_avg": ratio_succ_avg,
                        "acc_max_0.8": acc_max_08,
                        "acc_max_0.7": acc_max_07,
                        "reject_success_avg": reject_success_avg,
                        "ksr": keep_success_avg,
                        "keep_success_avg": keep_success_avg,
                        "decision_success_avg": decision_success_avg,
                        "decision_success_rate": decision_success_avg,
                        "iou_sample_count": iou_sample_count,
                        "bde_sample_count": bde_sample_count,
                        "reject_sample_count": reject_sample_count,
                        "keep_sample_count": keep_sample_count,
                        "decision_sample_count": decision_sample_count,
                    },
                    "details": all_results,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        print(f"结果已保存至: {args.result_json}")

    if args.result_json and args.output_dir:
        convert_to_custom_format(args.result_json, f"{args.output_dir}/meta.json")

    if args.output_dir:
        model_name = os.path.basename(args.output_dir)
        match = re.search(r"_step_(\d+)", args.output_dir)
        step = match.group(1) if match else "unknown"

        # 可选：从 VLM 评测汇总中读取分维度平均分
        vqa_g_avg = None
        vqa_c_avg = None
        vqa_r_avg = None
        vqa_all_avg = None
        if args.vlm_summary_json and os.path.exists(args.vlm_summary_json):
            try:
                with open(args.vlm_summary_json, "r", encoding="utf-8") as f:
                    vlm_summary = json.load(f)
                q_acc = (vlm_summary.get("question_accuracy", {}) if isinstance(vlm_summary, dict) else {}) or {}
                mean_obj = (vlm_summary.get("mean", {}) if isinstance(vlm_summary, dict) else {}) or {}

                g_keys = ["G1", "G2", "G3", "G4", "G5"]
                c_keys = ["C2", "C3"]
                r_keys = ["R2", "R3", "R4"]

                def _avg(keys):
                    vals = [float(q_acc[k]) for k in keys if k in q_acc]
                    return (sum(vals) / len(vals)) if vals else None

                vqa_g_avg = _avg(g_keys)
                vqa_c_avg = _avg(c_keys)
                vqa_r_avg = _avg(r_keys)
                if "avg_score" in mean_obj:
                    vqa_all_avg = float(mean_obj["avg_score"])
                else:
                    vals = [float(q_acc[k]) for k in q_acc]
                    vqa_all_avg = (sum(vals) / len(vals)) if vals else None
            except Exception as e:
                print(f"⚠️ 读取 VLM 汇总失败: {e}")

        row_data = {
            "方法": model_name,
            "训练设置": "",
            "Steps": step,
            "IoU_avg_最匹配候选框 ⬆️": f"{avg_iou_max * 100:.2f}%",
            "BDE_最匹配候选框 ⬇️": f"{avg_bde:.4f}",
            "Acc_0.8_最匹配候选框 ⬆️": f"{acc_max_08 * 100:.2f}%",
            "Acc_0.7_最匹配候选框 ⬆️": f"{acc_max_07 * 100:.2f}%",
            "Reject_Success_拒绝成功率⬆️": f"{reject_success_avg * 100:.2f}%",
            "Keep_Success_保留原图成功率⬆️": f"{keep_success_avg * 100:.2f}%",
            "Decision_Success_决策成功率⬆️": f"{decision_success_avg * 100:.2f}%",
            "比例准确率⬆️": f"{ratio_succ_avg * 100:.2f}%",
            "VQA_G维度均分⬆️": f"{vqa_g_avg * 100:.2f}%" if vqa_g_avg is not None else "",
            "VQA_C维度均分⬆️": f"{vqa_c_avg * 100:.2f}%" if vqa_c_avg is not None else "",
            "VQA_R维度均分⬆️": f"{vqa_r_avg * 100:.2f}%" if vqa_r_avg is not None else "",
            "VQA_所有问题均分⬆️": f"{vqa_all_avg * 100:.2f}%" if vqa_all_avg is not None else "",
        }
        os.makedirs(args.output_dir, exist_ok=True)
        excel_path = os.path.join(args.output_dir, "record_result.xlsx")
        df = pd.DataFrame([row_data])

        try:
            # 某些挂载盘/网络文件系统不支持 xlsxwriter 所需的 seek，
            # 先写到本地临时文件再复制到目标目录更稳妥。
            with tempfile.NamedTemporaryFile(prefix="record_result_", suffix=".xlsx", delete=False) as tmp:
                tmp_excel = tmp.name

            try:
                df.to_excel(tmp_excel, index=False)
                shutil.copyfile(tmp_excel, excel_path)
            finally:
                if os.path.exists(tmp_excel):
                    os.remove(tmp_excel)

            print(f"结果 Excel 已保存至: {excel_path}")
        except Exception as e:
            csv_path = os.path.join(args.output_dir, "record_result.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"⚠️ Excel 写入失败: {e}")
            print(f"已回退保存为 CSV: {csv_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", type=str, default="model", choices=["model", "baseline"], help="评测模式")
    parser.add_argument("--baseline_name", type=str, default="gemini", help="baseline 模型名或 API 模型别名")
    parser.add_argument("--annotation_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="", help="model模式或venus baseline使用")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Debug")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--result_json", type=str, default="eval_results.json")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="可用 GPU 列表")
    parser.add_argument("--gpus_per_process", type=int, default=2, help="每个进程使用的 GPU 数量")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="本地模型生成最大 token 数")
    parser.add_argument("--internvl_max_tiles", type=int, default=12, help="InternVL 动态切图最大 tile 数")
    parser.add_argument("--internvl_input_size", type=int, default=448, help="InternVL 视觉输入 tile 尺寸")
    parser.add_argument("--vlm_summary_json", type=str, default="", help="可选：VLM 评测 summary.json 路径")
    parser.add_argument("--api_max_workers", type=int, default=4, help="API baseline 并发线程数")
    parser.add_argument("--api_max_retries", type=int, default=4, help="API baseline 重试次数")
    parser.add_argument("--resume", action="store_true", help="断点续跑：仅处理 result_json 中缺失的样本")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 result_json，强制全量重跑")
    parser.add_argument("--ratio_following", action="store_true", help="启用 ratio-following：prompt 比例随机采样，并以该比例评估 ratio success")
    parser.add_argument("--limit", type=int, default=0, help="评测条数上限；0 表示全量评测")
    parser.add_argument("--gemini_model", type=str, default="gemini-3-pro-native")
    parser.add_argument("--gemini_base_url", type=str, default=os.getenv("GEMINI_BASE_URL", "https://models-proxy.stepfun-inc.com/gemini"))
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--gpt_model", type=str, default="gpt-5.5")
    parser.add_argument("--gpt_base_url", type=str, default=os.getenv("GPT_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"))
    parser.add_argument("--gpt_api_key", type=str, default=os.getenv("GPT_API_KEY", ""))
    parser.add_argument("--qwen_model", type=str, default=os.getenv("QWEN_MODEL_NAME", "qwen3-vl-235b-a22b-instruct"))
    parser.add_argument("--qwen_base_url", type=str, default=os.getenv("QWEN_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"))
    parser.add_argument("--qwen_api_key", type=str, default=os.getenv("QWEN_API_KEY", ""))
    parser.add_argument("--instructcrop_project_dir", type=str, default=DEFAULT_INSTRUCTCROP_PROJECT_DIR)
    parser.add_argument("--instructcrop_base_model_path", type=str, default="/mnt/workspacedir/lijiayu/checkpoints/llava-v1.6-7b")
    parser.add_argument("--instructcrop_checkpoint_path", type=str, default="/mnt/data/lijiayuoss/InstructCrop/stage1.pth")
    parser.add_argument("--instructcrop_instruct_ckpt_path", type=str, default="/mnt/data/lijiayuoss/InstructCrop/stage2.pth")
    parser.add_argument("--instructcrop_vision_tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--instructcrop_prompt", type=str, default=DEFAULT_INSTRUCTCROP_PROMPT)
    parser.add_argument("--instructcrop_device", type=str, default="cuda")
    parser.add_argument("--instructcrop_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--instructcrop_input_size", type=int, default=256)
    parser.add_argument("--instructcrop_max_new_tokens", type=int, default=128)
    parser.add_argument("--instructcrop_out_dim", type=int, default=256)
    parser.add_argument("--instructcrop_use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--instructcrop_conv_type", type=str, default="llava_v1")
    parser.add_argument("--instructcrop_lora_r", type=int, default=8)
    parser.add_argument("--instructcrop_lora_alpha", type=int, default=16)
    parser.add_argument("--instructcrop_lora_dropout", type=float, default=0.05)
    parser.add_argument("--instructcrop_lora_target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument(
        "--data_format",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="数据格式: 'eval' (评测集JSON) 或 'train' (训练集JSONL)",
    )

    args = parser.parse_args()
    main(args)
