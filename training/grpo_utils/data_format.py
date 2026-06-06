import json
import os
import re
from typing import Any, Dict, List

from PIL import Image
from swift.llm.dataset import DatasetMeta, ResponsePreprocessor, SubsetDataset, register_dataset


class AestheticPreprocessor(ResponsePreprocessor):
    RATIO_RE = re.compile(r"(\d+)\s*[：:]\s*(\d+)")

    @staticmethod
    def _get_message_content(messages: List[Dict[str, Any]], role: str) -> str:
        for msg in messages:
            if msg.get("role") == role:
                return msg.get("content", "")
        return ""

    @staticmethod
    def _extract_instance_info(content: str) -> Dict[str, Any]:
        try:
            data = json.loads(content)
            instance_info = data.get("instance_info", [])
            if instance_info:
                return instance_info[0]
        except Exception:
            pass
        return {}

    @staticmethod
    def _norm_bbox_to_1000(bbox: Any) -> List[int]:
        if not bbox or len(bbox) != 4:
            return []
        bbox = [float(x) for x in bbox]
        if max(bbox) <= 1.0:
            return [int(x * 1000) for x in bbox]
        return [int(x) for x in bbox]

    @staticmethod
    def _format_bbox(gt_bbox: List[int]) -> str:
        if not gt_bbox:
            return ""
        return f"({gt_bbox[0]},{gt_bbox[1]}),({gt_bbox[2]},{gt_bbox[3]})"

    def _build_response(self, assistant_content: str, task_type: str, gt_bbox: List[int]) -> str:
        if task_type != "composition":
            return assistant_content

        bbox_text = self._format_bbox(gt_bbox)
        try:
            data = json.loads(assistant_content)
            instance_info = data.get("instance_info", [])
            if instance_info:
                instance_info[0]["composition_xy"] = bbox_text
                return json.dumps(data, ensure_ascii=False)
        except Exception:
            pass

        return assistant_content.replace("<bbox>", bbox_text)

    def _extract_target_ratio(self, row: Dict[str, Any], query: str) -> str:
        if row.get("ratio"):
            return row["ratio"]

        ratio_aug = row.get("objects", {}).get("ratio_following_aug", {})
        if ratio_aug.get("source_bbox_ratio"):
            return ratio_aug["source_bbox_ratio"]

        match = self.RATIO_RE.search(query)
        if match:
            return f"{match.group(1)}:{match.group(2)}"
        return ""

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        objects = row.get("objects", {})
        bboxes = objects.get("bbox", [])
        image_path = row["images"][0]
        try:
            with Image.open(image_path) as img:
                image_w, image_h = img.size
        except Exception:
            image_w, image_h = row.get("im_size", [1, 1])

        gt_bbox = self._norm_bbox_to_1000(bboxes[0]) if bboxes else []
        messages = row.get("messages", [])
        if not messages:
            raise ValueError("no message in data")

        query = self._get_message_content(messages, "user").replace("<image>", "").strip()
        assistant_content = self._get_message_content(messages, "assistant")
        instance_info = self._extract_instance_info(assistant_content)
        task_type = instance_info.get("task_type") or ("pose" if "keypoints_xyn" in instance_info else "composition")
        response = self._build_response(assistant_content, task_type, gt_bbox)

        return super().preprocess(
            {
                "query": query,
                "ground_truth_bbox": gt_bbox,
                "category": objects.get("category") or "",
                "task_type": task_type,
                "keypoints_xyn": instance_info.get("keypoints_xyn", []),
                "visibility": instance_info.get("visibility", []),
                "response": response,
                "images": row["images"],
                "im_size": [image_w, image_h],
                "tag": row.get("tag", "cut_feet"),
                "target_ratio": self._extract_target_ratio(row, query),
                "system": "You are a helpful assistant.",
            }
        )


def _register_dataset(dataset_name: str, dataset_path: str) -> None:
    register_dataset(
        DatasetMeta(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            subsets=[SubsetDataset(name="default", subset="default", split=["train"])],
            preprocess_func=AestheticPreprocessor(),
        )
    )


_register_dataset("composition_rl", os.environ.get("GRPO_COMPOSITION_DATASET_PATH", "/path/to/composition_rl.jsonl"))
_register_dataset("pose_rl", os.environ.get("GRPO_POSE_DATASET_PATH", "/path/to/pose_rl.jsonl"))
_register_dataset("rl", os.environ.get("GRPO_DATASET_PATH", "/path/to/grpo_dataset.jsonl"))
