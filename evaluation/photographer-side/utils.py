from PIL import Image, ImageOps
import json
import os
import re
import ast
from io import BytesIO
import base64
import math
from tqdm import tqdm

def image_to_base64(image_pil, format="JPEG", quality=85):
    buffered = BytesIO()
    image_pil = image_pil.convert("RGB")
    image_pil.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def convert_to_custom_format(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    try:
        # 1. 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        details_list = data.get("details", [])
        new_data = {}
        
        # 2. 遍历处理
        for item in details_list:
            item_id = item.get("id")
            if not item_id:
                continue

            # --- 提取并处理字段 ---
            
            # (A) output_text
            # 注意：如果文本里原本就有单引号，可能会破坏格式，这里假设文本比较干净
            text_val = str(item.get("output_text", "") or "").strip()
            if not text_val:
                # output_text 为空时跳过，避免写入无效 meta 条目
                continue

            # (B) iou_list
            # 目标格式是：去掉[], 并在每个数字前加缩进(示例中约为16个空格)，数字间换行
            raw_list = item.get("iou_list", [])
            # 将数字转为字符串
            str_list = [str(x) for x in raw_list]
            # 拼接逻辑：用 ",\n                " 连接
            # 这里的空格数量是根据你的示例目测的 (约16个空格)
            indent = "                " 
            list_val = indent + (",\n" + indent).join(str_list)

            # (C) iou_main
            main_val = str(item.get("iou_main", ""))
            target_ratio = str(item.get("ratio",""))
            prompt_text = str(item.get("prompt_text", "") or "").strip()
            # 轻量转义，避免破坏 pseudo-dict 的单引号包裹
            prompt_text_safe = prompt_text.replace("'", "\\'")
            # --- 3. 手动拼装那个“奇怪”的字符串 ---
            # 格式要求: 
            # "{'output_text':'...','iou_list':'...',"iou_main":'...'}"
            # 注意 iou_main 的 key 是双引号，其他是单引号
            
            custom_str = (
                "{"
                f"'target_ratio':'{target_ratio}',"
                f"'prompt_text':'{prompt_text_safe}',"
                f"'output_text':'{text_val}',"
                f"'iou_list':'{list_val}',"
                f"\"iou_main\":'{main_val}'"
                "}"
            )

            # 将拼接好的字符串作为 Value
            new_data[item_id] = custom_str
        
        # 4. 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文正常显示
            # 自动转义：json.dump 会自动处理 custom_str 内部的双引号和换行符，
            # 生成合法的 JSON 文件（即你看到的 value 外部有引号，内部字符被转义）
            json.dump(new_data, f, indent=4, ensure_ascii=False)
            
        print(f"转换成功！文件已保存为: {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")

def resize_image_for_inference(image, min_side=1024):
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
        # 使用 LANCZOS 保持高质量
        resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized_img
    
    return image

def get_closest_ratio(img_w, img_h):
    """
    计算【原图】长宽比，匹配 [4:3, 3:4, 16:9, 9:16] 中最近的一个
    """
    if img_h == 0: return "4:3"
    current_ratio = img_w / img_h
    
    # 定义目标比例
    target_ratios = {
        "4:3": 4/3,
        "3:4": 3/4,
        "16:9": 16/9,
        "9:16": 9/16,
        "1:1": 1/1,
        "3:2": 3/2
        # 可以视情况加入 "1:1": 1.0, "3:2": 1.5 等
    }
    
    best_ratio_name = "4:3"
    min_diff = float('inf')
    
    for name, ratio_val in target_ratios.items():
        diff = abs(current_ratio - ratio_val)
        if diff < min_diff:
            min_diff = diff
            best_ratio_name = name
            
    return best_ratio_name

def calculate_iou(box1, box2):
    """计算单个 IoU"""
    if not box1 or not box2: return 0.0
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

def _extract_first_json_object(text):
    start = text.find("{")
    if start < 0:
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


def _safe_json_or_literal_load(text):
    s = (text or "").strip()
    if not s:
        return None
    for fn in (json.loads, ast.literal_eval):
        try:
            obj = fn(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    block = _extract_first_json_object(s)
    if not block:
        return None
    for fn in (json.loads, ast.literal_eval):
        try:
            obj = fn(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _normalize_bbox_with_auto_scale(vals, img_width, img_height):
    """
    支持三种 bbox 数值域：
    1) 0~1 归一化坐标
    2) 0~1000 归一化坐标（旧版逻辑）
    3) 直接像素坐标
    """
    if not vals or len(vals) != 4:
        return None
    try:
        x1_n, y1_n, x2_n, y2_n = [float(v) for v in vals]
    except Exception:
        return None

    max_abs = max(abs(x1_n), abs(y1_n), abs(x2_n), abs(y2_n))
    if max_abs <= 1.5:
        x1, y1, x2, y2 = x1_n * img_width, y1_n * img_height, x2_n * img_width, y2_n * img_height
    elif max_abs <= 1200:
        x1, y1, x2, y2 = (x1_n / 1000.0) * img_width, (y1_n / 1000.0) * img_height, (x2_n / 1000.0) * img_width, (y2_n / 1000.0) * img_height
    else:
        x1, y1, x2, y2 = x1_n, y1_n, x2_n, y2_n

    x1 = max(0.0, min(float(img_width), x1))
    y1 = max(0.0, min(float(img_height), y1))
    x2 = max(0.0, min(float(img_width), x2))
    y2 = max(0.0, min(float(img_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _extract_bbox_vals_from_json_obj(obj):
    if not isinstance(obj, dict):
        return None

    # 新格式优先: {"instance_info":[{"composition_xy": ...}]}
    inst = None
    if isinstance(obj.get("instance_info"), list) and len(obj["instance_info"]) > 0 and isinstance(obj["instance_info"][0], dict):
        inst = obj["instance_info"][0]
    else:
        inst = obj

    # 候选字段按优先级遍历
    for key in ("composition_xy", "bbox", "composition_bbox"):
        if key not in inst:
            continue
        v = inst.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return list(v)
        if isinstance(v, str):
            parsed = _extract_bbox_vals_from_text(v)
            if parsed:
                return parsed
    return None


def _extract_bbox_vals_from_text(text):
    s = str(text) if text is not None else ""
    # 兼容整数/小数/负号
    num = r"[-+]?\d*\.?\d+"

    # "(x1,y1),(x2,y2)"
    p1 = rf'\(\s*({num})\s*,\s*({num})\s*\)\s*,\s*\(\s*({num})\s*,\s*({num})\s*\)'
    m1 = re.search(p1, s)
    if m1:
        return [float(x) for x in m1.groups()]

    # "[x1,y1,x2,y2]"
    p2 = rf'\[\s*({num})\s*,\s*({num})\s*,\s*({num})\s*,\s*({num})\s*\]'
    m2 = re.search(p2, s)
    if m2:
        return [float(x) for x in m2.groups()]

    return None


def parse_qwen_bbox(text_output, img_width, img_height):
    """解析 Qwen 输出，兼容旧文本格式与新 JSON 结构化输出。"""
    # 1) 先尝试 JSON（新格式）
    obj = _safe_json_or_literal_load(text_output)
    if obj is not None:
        vals = _extract_bbox_vals_from_json_obj(obj)
        bbox = _normalize_bbox_with_auto_scale(vals, img_width, img_height)
        if bbox is not None:
            return bbox

    # 2) 回退到纯文本正则（旧格式）
    vals = _extract_bbox_vals_from_text(text_output)
    return _normalize_bbox_with_auto_scale(vals, img_width, img_height)

def clip_norm(val):
    return max(0.0, min(1.0, val))

def load_train_data(jsonl_path, image_root_dir=None):
    """
    加载 Qwen 训练格式的数据 (train_nbp.jsonl)
    格式: {"messages": [...], "images": ["path"], "objects": {"bbox": [[x1,y1,x2,y2]], "bbox_type": "norm1"}}
    """
    print(f"正在加载训练集数据: {jsonl_path}")
    dataset = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in tqdm(enumerate(f), desc="Parsing Train Data"):
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
                
                # 1. 获取图片路径
                # 训练数据里的 images 列表通常已经是绝对路径
                # 如果 image_root_dir 不为空，则尝试拼接（视你的数据生成逻辑而定）
                img_path_raw = item['images'][0]
                
                if image_root_dir:
                    # 如果提供了 root，假设 jsonl 里是相对路径或文件名
                    image_path = os.path.join(image_root_dir, os.path.basename(img_path_raw))
                else:
                    # 否则直接使用 jsonl 里的路径
                    image_path = img_path_raw
                
                if not os.path.exists(image_path):
                    # print(f"⚠️ Image not found: {image_path}")
                    continue

                # 2. 读取图片尺寸 (用于反归一化)
                # 为了速度，可以在 Worker 中读，但为了统一接口，这里先读
                with Image.open(image_path) as img:
                    img = ImageOps.exif_transpose(img)
                    w, h = img.size

                # 3. 获取 GT BBox
                objects = item.get('objects', {})
                bboxes = objects.get('bbox', [])
                bbox_type = objects.get('bbox_type', 'real')
                
                gt_bboxes = []
                if bboxes:
                    # 训练集通常只有一个主框，但 bbox 是个列表的列表 [[x1,y1,x2,y2]]
                    # 也有可能有多个框
                    for box in bboxes:
                        if bbox_type == 'norm1':
                            # 归一化 -> 绝对坐标
                            abs_box = [
                                box[0] * w,
                                box[1] * h,
                                box[2] * w,
                                box[3] * h
                            ]
                            gt_bboxes.append(abs_box)
                        else:
                            gt_bboxes.append(box)
                
                if not gt_bboxes:
                    continue
                
                # 4. 提取 Prompt 中的比例要求 (用于后续统计 Ratio Success)
                # 你的 prompt 格式: "...请按照4:3的比例..."
                user_content = item['messages'][1]['content']
                target_ratio_str = "4:3" # 默认值
                
                # 简单的正则提取
                ratio_match = re.search(r'按照(\d+:\d+)的比例', user_content)
                if ratio_match:
                    target_ratio_str = ratio_match.group(1)
                
                # 5. 构造标准数据项
                dataset.append({
                    "id": f"train_{line_idx}_{os.path.basename(image_path)}", # 构造一个唯一ID
                    "image_path": image_path,
                    "gt_bboxes": gt_bboxes,     # 训练集只有一个框，也放入列表
                    "best_bbox": gt_bboxes[0],  # 训练集的主框就是 best_bbox
                    "prompt_ratio": target_ratio_str # 记录 Prompt 里的比例要求
                })
                
            except Exception as e:
                print(f"Error parsing line {line_idx}: {e}")
                continue
                
    return dataset

def load_annotation_data(json_path, image_root_dir):
    """加载数据"""
    print(f"正在加载标注文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = []
    for filename, content_str in tqdm(raw_data.items(), desc="Parsing Data"):
        image_path = None
        for ext in ['.jpg', '.png', '.webp', '.jpeg']:
            if os.path.exists(os.path.join(image_root_dir, filename + ext)):
                image_path = os.path.join(image_root_dir, filename + ext)
                break
            if os.path.exists(os.path.join(image_root_dir, filename)):
                image_path = os.path.join(image_root_dir, filename)
                break
        if not image_path: continue
        data = ast.literal_eval(content_str)
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            w, h = img.size
        gt_bboxes = []
        composition_bboxes = []
        boxes = data.get('composition_boxes', [])
        if len(boxes) > 0:
            sorted_boxes = sorted(boxes, key=lambda x: int(x['id']))
            for b in sorted_boxes:
                rect = b['rect']
                abs_box = [clip_norm(rect[0]) * w, clip_norm(rect[1]) * h, clip_norm(rect[2]) * w, clip_norm(rect[3]) * h]
                composition_bboxes.append(abs_box)
                gt_bboxes.append(abs_box)

        origin_info = data.get('origin', {})
        good_reason = origin_info.get('好的原因', '')
        bad_reason = origin_info.get('差的原因', '')
        # 兼容新旧字段名
        origin_type = origin_info.get('返回原图类型', '') or origin_info.get('返回构图类型', '')
        secondary_type = origin_info.get('二次构图类型', '')

        # 新评测规则：
        # 1) 返回原图类型为“原图构图一般/原图构图好” => 加入原图框
        # 2) 二次构图类型包含“原图构图好” => 加入原图框
        need_add_full_image = (
            ("原图构图一般" in origin_type)
            or ("原图构图好" in origin_type)
            or ("原图构图好" in secondary_type)
            or ("原图构图好" in good_reason)  # 兼容历史数据
        )
        if need_add_full_image:
            gt_bboxes.append([0, 0, w, h])

        if len(gt_bboxes) == 0:
            if (
                "原图特别差" in bad_reason
                or origin_type == "原图构图差"
            ):
                pass
            else:
                print(f"{filename}数据错误！")
                continue
        dataset.append({
            "id": filename,
            "image_path": image_path,
            "gt_bboxes": gt_bboxes,
            "composition_bboxes": composition_bboxes,
            "best_bbox": gt_bboxes[0] if gt_bboxes else [],
            "is_reject_case": (len(gt_bboxes)==0)
        })
    return dataset

def check_ratio_success(bbox, target_ratio_str, tolerance=0.1):
    """
    判断 bbox 的比例是否符合 target_ratio_str
    tolerance: 允许的误差范围 (默认 10%)
    """
    if not bbox: return False
    
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if h == 0: return False
    
    pred_ratio = w / h
    # print(target_ratio_str)
    rw, rh = map(float, target_ratio_str.split(':'))
    target_val = rw / rh
    log_diff = abs(math.log((pred_ratio + 1e-6) / (target_val + 1e-6)))
    tolerance = 0.1
    if log_diff <= tolerance:
        return True
    else:
        return False


def calculate_bde(pred_box, gt_box, image_width=None, image_height=None):
    """
    计算 BDE（Boundary Distance Error）:
        BDE = 1/4 * (|lp - lgt| + |rp - rgt| + |tp - tgt| + |bp - bgt|)

    当提供 image_width/image_height 时，先将 x 边界除以 width、
    y 边界除以 height，再计算归一化 BDE。
    未提供图像尺寸时保留旧行为，返回 pixel-level BDE。
    """
    if not pred_box or not gt_box:
        return None
    lp, tp, rp, bp = [float(v) for v in pred_box]
    lgt, tgt, rgt, bgt = [float(v) for v in gt_box]
    if image_width is not None or image_height is not None:
        if image_width is None or image_height is None:
            return None
        image_width = float(image_width)
        image_height = float(image_height)
        if image_width <= 0 or image_height <= 0:
            return None
        lp, rp = lp / image_width, rp / image_width
        lgt, rgt = lgt / image_width, rgt / image_width
        tp, bp = tp / image_height, bp / image_height
        tgt, bgt = tgt / image_height, bgt / image_height
    return (abs(lp - lgt) + abs(rp - rgt) + abs(tp - tgt) + abs(bp - bgt)) / 4.0


# convert_to_custom_format("/data/datasets/evaluation/vis_results_e10_lora_v4/eval_records.json","/data/datasets/evaluation/vis_results_e10_lora_v4/meta.json")
