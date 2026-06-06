import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation


def iter_refine_images(dataset_path: Path) -> Iterable[str]:
    seen = set()
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            objects = row.get("objects") or {}
            if str(objects.get("category", "")).upper() != "REFINE":
                continue
            images = row.get("images") or []
            if images and images[0] not in seen:
                seen.add(images[0])
                yield images[0]


def load_done(output_path: Path) -> Dict[str, dict]:
    done = {}
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = item.get("image_path")
            if image_path:
                done[image_path] = item
    return done


def largest_component_bbox(mask: np.ndarray, min_area_ratio: float = 0.0005) -> Optional[Tuple[int, int, int, int]]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return None
    image_area = mask_u8.shape[0] * mask_u8.shape[1]
    min_area = max(16, int(image_area * min_area_ratio))
    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    if best_label is None or best_area < min_area:
        return None
    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    width = int(stats[best_label, cv2.CC_STAT_WIDTH])
    height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    return x, y, x + width, y + height


def predict_mask(model, transform, image: Image.Image, device: torch.device, threshold: float) -> np.ndarray:
    original_size = image.size
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = model(tensor)[-1].sigmoid().cpu()[0, 0].numpy()
    pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)
    return pred >= threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute BiRefNet saliency boxes for GRPO saliency reward.")
    parser.add_argument("--dataset", required=True, help="GRPO jsonl dataset path.")
    parser.add_argument("--output", required=True, help="Output jsonl consumed by SALIENCY_PRECOMPUTE_JSONL.")
    parser.add_argument("--model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    done = load_done(output_path)
    count = 0
    with output_path.open("a", encoding="utf-8") as out:
        for image_path in tqdm(iter_refine_images(dataset_path)):
            if image_path in done:
                continue
            if args.limit and count >= args.limit:
                break
            item = {"image_path": image_path, "saliency_bbox": None, "error": None}
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    mask = predict_mask(model, transform, image, device, args.threshold)
                    bbox = largest_component_bbox(mask)
                    item["saliency_bbox"] = list(bbox) if bbox is not None else None
            except Exception as exc:
                item["error"] = str(exc)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            count += 1


if __name__ == "__main__":
    main()
