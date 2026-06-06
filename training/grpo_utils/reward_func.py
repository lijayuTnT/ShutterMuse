import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from swift.plugin.orm import ORM, orms


def compute_iou(box1: List[float], box2: List[float]) -> float:
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    intersection_w = max(0, ix2 - ix1)
    intersection_h = max(0, iy2 - iy1)
    intersection_area = intersection_w * intersection_h
    if intersection_area == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    return 0.0 if union_area <= 0 else intersection_area / union_area


class RatioORM(ORM):
    RATIO_RE = re.compile(r"(\d+)\s*[：:]\s*(\d+)")
    BBOX_RE = re.compile(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\),\s*\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")

    @staticmethod
    def _extract_image_path(raw_img_data: Any) -> str:
        if isinstance(raw_img_data, str):
            return raw_img_data
        if isinstance(raw_img_data, dict):
            return raw_img_data.get("path", "") or raw_img_data.get("image", "")
        if isinstance(raw_img_data, (list, tuple)) and raw_img_data:
            return RatioORM._extract_image_path(raw_img_data[0])
        return ""

    @classmethod
    def _is_ratio_following_image(cls, raw_img_data: Any) -> bool:
        prefix = os.environ.get("RATIO_FOLLOWING_IMAGE_PREFIX", "")
        if not prefix:
            return True
        image_path = os.path.normpath(cls._extract_image_path(raw_img_data))
        return image_path.startswith(os.path.normpath(prefix))

    def __call__(self, completions: List[str], im_size: List[List[int]], **kwargs) -> List[float]:
        rewards = []
        tolerance = 0.15
        target_ratios = kwargs.get("target_ratio", [""] * len(completions))
        task_types = kwargs.get("task_type", ["composition"] * len(completions))
        images = kwargs.get("images", [None] * len(completions))

        for completion, ratio_str, size, task_type, raw_img_data in zip(completions, target_ratios, im_size, task_types, images):
            if task_type != "composition" or not self._is_ratio_following_image(raw_img_data):
                rewards.append(0.0)
                continue
            ratio_match = self.RATIO_RE.search(ratio_str)
            if not ratio_match:
                rewards.append(0.0)
                continue
            target_ar = float(ratio_match.group(1)) / float(ratio_match.group(2))
            width, height = float(size[0]), float(size[1])
            if width <= 0 or height <= 0:
                rewards.append(0.0)
                continue
            matches = list(self.BBOX_RE.finditer(completion))
            if not matches:
                rewards.append(0.0)
                continue
            x1, y1, x2, y2 = map(float, matches[-1].groups())
            if x2 <= x1 or y2 <= y1:
                rewards.append(0.0)
                continue
            pred_ar = ((x2 - x1) / (y2 - y1)) * (width / height)
            log_diff = abs(math.log(pred_ar / target_ar))
            rewards.append(max(0.0, 1.0 - log_diff / tolerance))
        return rewards


class IoUORM(ORM):
    BBOX_RE = re.compile(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\),\s*\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")

    def __call__(self, completions: List[str], ground_truth_bbox: List[List[int]], **kwargs) -> List[float]:
        rewards = []
        iou_threshold = 0.7
        task_types = kwargs.get("task_type", ["composition"] * len(completions))
        categories = kwargs.get("category", [""] * len(completions))
        for completion, gt_box, task_type, category in zip(completions, ground_truth_bbox, task_types, categories):
            if task_type != "composition" or str(category).upper() != "REFINE":
                rewards.append(0.0)
                continue
            matches = list(self.BBOX_RE.finditer(completion))
            if not matches or not gt_box:
                rewards.append(0.0)
                continue
            x1, y1, x2, y2 = map(float, matches[-1].groups())
            if x2 <= x1 or y2 <= y1:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if compute_iou([x1, y1, x2, y2], gt_box) > iou_threshold else 0.0)
        return rewards


class SaliencyORM(ORM):
    BBOX_RE = re.compile(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\),\s*\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")
    COVER_TOLERANCE_PIXELS = 2
    COVER_RATIO_THRESHOLD = 0.9

    def __init__(self):
        super().__init__()
        self._loaded = False
        self._precomputed_bboxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {}

    @staticmethod
    def _normalize_category(category: Any) -> str:
        if category is None:
            return ""
        category = str(category).strip().upper()
        return "REFINE" if category in {"REFINE", "CORRECT", "VARIATION"} else category

    @staticmethod
    def _extract_image_path(raw_img_data: Any) -> str:
        if isinstance(raw_img_data, str):
            return raw_img_data
        if isinstance(raw_img_data, dict):
            return raw_img_data.get("path", "") or raw_img_data.get("image", "")
        if isinstance(raw_img_data, (list, tuple)) and raw_img_data:
            return SaliencyORM._extract_image_path(raw_img_data[0])
        return ""

    @classmethod
    def _extract_pred_box(cls, completion: str, image_w: int, image_h: int) -> Optional[Tuple[int, int, int, int]]:
        matches = list(cls.BBOX_RE.finditer(completion))
        if not matches:
            return None
        x1, y1, x2, y2 = map(float, matches[-1].groups())
        if x2 <= x1 or y2 <= y1:
            return None
        if max(x1, y1, x2, y2) <= 1.5:
            scale_x, scale_y = image_w, image_h
        else:
            scale_x, scale_y = image_w / 1000.0, image_h / 1000.0
        return (
            max(0, min(image_w, int(round(x1 * scale_x)))),
            max(0, min(image_h, int(round(y1 * scale_y)))),
            max(0, min(image_w, int(round(x2 * scale_x)))),
            max(0, min(image_h, int(round(y2 * scale_y)))),
        )

    def _load_precomputed_bboxes(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        precompute_path = os.environ.get("SALIENCY_PRECOMPUTE_JSONL", "")
        if not precompute_path or not os.path.exists(precompute_path):
            return
        try:
            with open(precompute_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    image_path = item.get("image_path", "")
                    if not image_path:
                        continue
                    bbox = item.get("saliency_bbox")
                    key = os.path.normpath(image_path)
                    if isinstance(bbox, list) and len(bbox) == 4:
                        try:
                            self._precomputed_bboxes[key] = tuple(int(round(float(v))) for v in bbox)
                        except (TypeError, ValueError):
                            self._precomputed_bboxes[key] = None
                    else:
                        self._precomputed_bboxes[key] = None
        except Exception as exc:
            print(f"[SaliencyORM] Failed to load {precompute_path}: {exc}")

    def _get_saliency_bbox(self, image_path: str) -> Optional[Tuple[int, int, int, int]]:
        self._load_precomputed_bboxes()
        return self._precomputed_bboxes.get(os.path.normpath(image_path))

    @classmethod
    def _covers(cls, pred_box: Tuple[int, int, int, int], saliency_box: Tuple[int, int, int, int]) -> bool:
        px1, py1, px2, py2 = pred_box
        sx1, sy1, sx2, sy2 = saliency_box
        tol = cls.COVER_TOLERANCE_PIXELS
        saliency_area = max(0, sx2 - sx1) * max(0, sy2 - sy1)
        if saliency_area <= 0:
            return False
        ix1 = max(px1, sx1 - tol)
        iy1 = max(py1, sy1 - tol)
        ix2 = min(px2, sx2 + tol)
        iy2 = min(py2, sy2 + tol)
        covered_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return covered_area / saliency_area >= cls.COVER_RATIO_THRESHOLD

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        task_types = kwargs.get("task_type", ["composition"] * len(completions))
        categories = kwargs.get("category", [""] * len(completions))
        images = kwargs.get("images", [None] * len(completions))
        im_sizes = kwargs.get("im_size", [None] * len(completions))

        for completion, task_type, category, raw_img_data, im_size in zip(completions, task_types, categories, images, im_sizes):
            if task_type != "composition" or self._normalize_category(category) != "REFINE":
                rewards.append(0.0)
                continue
            image_path = self._extract_image_path(raw_img_data)
            if not image_path:
                rewards.append(0.0)
                continue
            image_w = image_h = None
            if isinstance(im_size, (list, tuple)) and len(im_size) >= 2:
                try:
                    image_w, image_h = int(im_size[0]), int(im_size[1])
                except (TypeError, ValueError):
                    image_w = image_h = None
            if not image_w or not image_h:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    rewards.append(0.0)
                    continue
                image_h, image_w = image.shape[:2]
            pred_box = self._extract_pred_box(completion, image_w, image_h)
            saliency_box = self._get_saliency_bbox(image_path)
            rewards.append(1.0 if pred_box is not None and saliency_box is not None and self._covers(pred_box, saliency_box) else 0.0)
        return rewards


class PoseVisibilityORM(ORM):
    VISIBILITY_RE = re.compile(r'"visibility"\s*:\s*(\[[^\]]*\])')

    @staticmethod
    def _normalize_visibility(visibility: Any) -> Optional[List[int]]:
        if visibility is None or not isinstance(visibility, list):
            return None
        try:
            return [int(v) for v in visibility]
        except (TypeError, ValueError):
            return None

    def _extract_visibility(self, completion: str) -> Optional[List[int]]:
        try:
            data = json.loads(completion)
            instance_info = data.get("instance_info", [])
            if instance_info:
                visibility = self._normalize_visibility(instance_info[0].get("visibility"))
                if visibility is not None:
                    return visibility
        except Exception:
            pass
        match = self.VISIBILITY_RE.search(completion)
        if not match:
            return None
        try:
            return self._normalize_visibility(json.loads(match.group(1)))
        except Exception:
            return None

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        gt_visibilities = kwargs.get("visibility", [[] for _ in completions])
        task_types = kwargs.get("task_type", ["pose"] * len(completions))
        for completion, gt_visibility, task_type in zip(completions, gt_visibilities, task_types):
            if task_type != "pose":
                rewards.append(0.0)
                continue
            pred_visibility = self._extract_visibility(completion)
            target_visibility = self._normalize_visibility(gt_visibility)
            rewards.append(1.0 if pred_visibility is not None and target_visibility is not None and pred_visibility == target_visibility else 0.0)
        return rewards


orms["ratio_orm"] = RatioORM
orms["iou_orm"] = IoUORM
orms["pose_visibility_orm"] = PoseVisibilityORM
orms["saliency_orm"] = SaliencyORM
