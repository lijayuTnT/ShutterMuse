import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Draw 17-keypoint visualizations from prediction JSON files.")
    parser.add_argument(
        "--json-dir",
        type=str,
        default="",
        help="Single prediction json folder (contains *.json).",
    )
    parser.add_argument(
        "--json-dir-glob",
        type=str,
        default="",
        help="Glob pattern for multiple prediction json folders.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Image folder for matching image files by stem.",
    )
    parser.add_argument(
        "--gt-image-dir",
        type=str,
        default="",
        help="Optional GT image folder. If provided, output will be [vis | gt].",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Valid only when a single --json-dir is used.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_kptsV",
        help="Suffix for auto output path when processing each json dir independently.",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=".png,.jpg,.jpeg,.bmp,.webp",
        help="Comma-separated image extensions for matching.",
    )
    parser.add_argument(
        "--mixtask",
        action="store_true",
        help="If set, draw composition boxes: blue GT located_bbox_xyxy and red predicted bbox.",
    )
    parser.add_argument(
        "--bbox-json",
        type=str,
        default="",
        help="BBox json used by --mixtask. Reads records[*].located_bbox_xyxy as blue boxes.",
    )
    return parser.parse_args()


def load_located_bbox_map(bbox_json_path: str) -> Dict[str, list]:
    if not bbox_json_path:
        return {}
    p = Path(bbox_json_path)
    if not p.exists():
        raise FileNotFoundError(f"--bbox-json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    if isinstance(records, dict):
        iterable = records.items()
    elif isinstance(records, list):
        iterable = enumerate(records)
    else:
        raise ValueError(f"Invalid bbox json records: {p}")

    bbox_map = {}
    for key, rec in iterable:
        if not isinstance(rec, dict):
            continue
        bbox = rec.get("located_bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        names = {str(key)}
        output_image = rec.get("output_image")
        source_image = rec.get("source_image")
        if output_image:
            names.add(str(output_image))
        if source_image:
            names.add(str(source_image))
        for name in names:
            bbox_map[name] = bbox
            bbox_map[Path(name).stem] = bbox
    return bbox_map


class KeypointVisualizer:
    def __init__(self, image_path, json_path, mixtask=False, gt_bbox_xyxy=None):
        """
        初始化可视化器
        
        参数:
        - image_path: 图片路径
        - json_path: JSON文件路径
        """
        self.image_path = image_path
        self.json_path = json_path
        self.mixtask = mixtask
        self.gt_bbox_xyxy = gt_bbox_xyxy
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        self.height, self.width = self.img.shape[:2]
        self.keypoint_names = [
            '鼻子', '左眼', '右眼', '左耳', '右耳',
            '左肩', '右肩', '左手肘', '右肘', '左手腕', '右腕',
            '左髋', '右髋', '左膝', '右膝', '左脚踝', '右脚踝'
        ]
        self.keypoints_data = self.load_keypoints()
    
    def load_keypoints(self):
        """
        从JSON文件加载关键点数据
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data.get("instance_info"):
            raise ValueError("JSON文件中没有instance_info数据")
        
        instance = data["instance_info"][0]
        
        # 提取关键点坐标和置信度
        keypoints_xyn = instance.get("keypoints_xyn", [])
        keypoints_conf = instance.get("keypoints_conf", [])
        visibility = instance.get("visibility", [])
        box_xyxyn = instance.get("box_xyxyn", [0, 0, 1, 1])
        pred_bbox = instance.get("bbox", data.get("bbox"))
        pred_bbox_type = instance.get("bbox_type", data.get("bbox_type", "norm1"))
        objects = data.get("objects")
        if pred_bbox is None and isinstance(objects, dict):
            pred_bbox = objects.get("bbox")
            pred_bbox_type = objects.get("bbox_type", pred_bbox_type)

        # 有些预测结果没有 keypoints_conf，仅有 visibility；这里做兼容兜底
        if not keypoints_conf:
            if visibility:
                keypoints_conf = [float(v) for v in visibility]
            else:
                keypoints_conf = [1.0] * len(keypoints_xyn)
        else:
            keypoints_conf = [float(v) for v in keypoints_conf]

        # 对齐长度，避免脏数据导致 zip 后关键点被截断
        if len(keypoints_conf) < len(keypoints_xyn):
            keypoints_conf = keypoints_conf + [1.0] * (len(keypoints_xyn) - len(keypoints_conf))
        elif len(keypoints_conf) > len(keypoints_xyn):
            keypoints_conf = keypoints_conf[:len(keypoints_xyn)]

        # 可见性对齐：缺失时默认可见(1)
        if not visibility:
            visibility = [1] * len(keypoints_xyn)
        if len(visibility) < len(keypoints_xyn):
            visibility = visibility + [1] * (len(keypoints_xyn) - len(visibility))
        elif len(visibility) > len(keypoints_xyn):
            visibility = visibility[:len(keypoints_xyn)]
        
        # 创建关键点字典
        keypoints = {}
        for i, (name, (x, y), conf, vis) in enumerate(zip(
            self.keypoint_names, 
            keypoints_xyn, 
            keypoints_conf,
            visibility,
        )):
            keypoints[name] = {
                "x": float(x),
                "y": float(y),
                "confidence": float(conf),
                "visibility": int(vis),
            }
        
        # 添加边界框信息
        keypoints["_box"] = {
            "x1": float(box_xyxyn[0]),
            "y1": float(box_xyxyn[1]),
            "x2": float(box_xyxyn[2]),
            "y2": float(box_xyxyn[3])
        }
        keypoints["_pred_bbox"] = self.normalize_pred_bbox(pred_bbox, pred_bbox_type)
        
        return keypoints

    def normalize_pred_bbox(self, bbox, bbox_type="norm1") -> Optional[list]:
        if bbox is None:
            return None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 1 and isinstance(bbox[0], (list, tuple)):
            bbox = bbox[0]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 2:
            p1, p2 = bbox
            if isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) >= 2 and len(p2) >= 2:
                bbox = [p1[0], p1[1], p2[0], p2[1]]
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return None
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if max_abs <= 1.0:
            norm = [x1, y1, x2, y2]
        elif bbox_type in {"qwen1000", "norm1000"} or max_abs <= 1000.0:
            norm = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
        else:
            norm = [x1 / self.width, y1 / self.height, x2 / self.width, y2 / self.height]
        return [max(0.0, min(1.0, float(v))) for v in norm]

    def bbox_norm_to_abs(self, bbox_norm) -> Optional[tuple]:
        if not bbox_norm or len(bbox_norm) != 4:
            return None
        x1, y1, x2, y2 = bbox_norm
        return (
            int(round(float(x1) * self.width)),
            int(round(float(y1) * self.height)),
            int(round(float(x2) * self.width)),
            int(round(float(y2) * self.height)),
        )

    def clamp_abs_bbox(self, bbox_xyxy) -> Optional[tuple]:
        if not bbox_xyxy or len(bbox_xyxy) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
        except Exception:
            return None
        x1 = max(0, min(self.width - 1, x1))
        x2 = max(0, min(self.width - 1, x2))
        y1 = max(0, min(self.height - 1, y1))
        y2 = max(0, min(self.height - 1, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    def draw_bbox(self, image, bbox_xyxy, color, label=None):
        bbox = self.clamp_abs_bbox(bbox_xyxy)
        if bbox is None:
            return image
        x1, y1, x2, y2 = bbox
        thickness = max(2, int(round(min(self.width, self.height) * 0.004)))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if label:
            font_scale = max(0.45, min(self.width, self.height) / 1800.0)
            label_thickness = max(1, thickness // 2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
            y_text = max(th + baseline + 4, y1 - 4)
            cv2.rectangle(image, (x1, y_text - th - baseline - 4), (x1 + tw + 6, y_text + baseline), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 3, y_text - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA,
            )
        return image

    def draw_mixtask_boxes(self, image):
        if not self.mixtask:
            return image
        if self.gt_bbox_xyxy:
            self.draw_bbox(image, self.gt_bbox_xyxy, (255, 0, 0), "GT")
        pred_bbox_norm = self.keypoints_data.get("_pred_bbox")
        pred_bbox_xyxy = self.bbox_norm_to_abs(pred_bbox_norm)
        if pred_bbox_xyxy:
            self.draw_bbox(image, pred_bbox_xyxy, (0, 0, 255), "Pred")
        return image
    
    def get_absolute_coordinates(self, use_box_normalization=False):
        """
        将相对坐标转换为绝对坐标
        
        参数:
        - use_box_normalization: 是否使用边界框归一化
        """
        abs_keypoints = {}
        
        if use_box_normalization:
            # 使用边界框归一化
            box = self.keypoints_data["_box"]
            box_x1 = int(box["x1"] * self.width)
            box_y1 = int(box["y1"] * self.height)
            box_x2 = int(box["x2"] * self.width)
            box_y2 = int(box["y2"] * self.height)
            
            box_width = box_x2 - box_x1
            box_height = box_y2 - box_y1
            
            for name, data in self.keypoints_data.items():
                if not name.startswith("_"):
                    # 关键点坐标是相对于边界框的
                    x_in_box = data["x"] * box_width
                    y_in_box = data["y"] * box_height
                    
                    # 转换为图像绝对坐标
                    abs_x = int(box_x1 + x_in_box)
                    abs_y = int(box_y1 + y_in_box)
                    
                    abs_keypoints[name] = {
                        "x": abs_x,
                        "y": abs_y,
                        "confidence": data["confidence"],
                        "visibility": data.get("visibility", 1),
                    }
        else:
            # 直接使用图像尺寸归一化
            for name, data in self.keypoints_data.items():
                if not name.startswith("_"):
                    abs_x = int(data["x"] * self.width)
                    abs_y = int(data["y"] * self.height)
                    
                    abs_keypoints[name] = {
                        "x": abs_x,
                        "y": abs_y,
                        "confidence": data["confidence"],
                        "visibility": data.get("visibility", 1),
                    }
        
        return abs_keypoints
    
    def visualize_basic_keypoints(self, save_path=None, show=False):
        """
        基础关键点可视化
        """
        # 创建副本
        result = self.img.copy()
        
        # 获取绝对坐标
        abs_kps = self.get_absolute_coordinates()
        
        # 定义颜色
        colors = {
            '鼻子': (0, 255, 255),      # 黄色
            '左眼': (255, 0, 0),        # 蓝色
            '右眼': (255, 0, 0),        # 蓝色
            '左耳': (0, 255, 0),        # 绿色
            '右耳': (0, 255, 0),        # 绿色
            '左肩': (255, 165, 0),      # 橙色
            '右肩': (255, 165, 0),      # 橙色
            '左手肘': (128, 0, 128),    # 紫色
            '右肘': (128, 0, 128),      # 紫色
            '左手腕': (255, 192, 203),  # 粉色
            '右腕': (255, 192, 203),    # 粉色
            '左髋': (0, 0, 255),        # 红色
            '右髋': (0, 0, 255),        # 红色
            '左膝': (0, 255, 255),      # 青色
            '右膝': (0, 255, 255),      # 青色
            '左脚踝': (255, 255, 0),    # 青色
            '右脚踝': (255, 255, 0)     # 青色
        }
        
        # 绘制骨架连接
        skeleton = [
            ('鼻子', '左眼'), ('鼻子', '右眼'),
            ('左眼', '左耳'), ('右眼', '右耳'),
            ('左肩', '右肩'),
            ('左肩', '左手肘'), ('左手肘', '左手腕'),
            ('右肩', '右肘'), ('右肘', '右腕'),
            ('左肩', '左髋'), ('右肩', '右髋'),
            ('左髋', '左膝'), ('左膝', '左脚踝'),
            ('右髋', '右膝'), ('右膝', '右脚踝'),
            ('左髋', '右髋')
        ]
        
        for start_name, end_name in skeleton:
            if start_name in abs_kps and end_name in abs_kps:
                start_x = abs_kps[start_name]["x"]
                start_y = abs_kps[start_name]["y"]
                end_x = abs_kps[end_name]["x"]
                end_y = abs_kps[end_name]["y"]
                start_vis = abs_kps[start_name].get("visibility", 1)
                end_vis = abs_kps[end_name].get("visibility", 1)
                # 任一端点在画面外(-1)则不绘制该连线
                if start_vis < 0 or end_vis < 0:
                    continue
                # 只要有一个关键点不可见，则该边标红
                line_color = (0, 0, 255) if (start_vis == 0 or end_vis == 0) else colors.get(start_name, (255, 255, 255))
                cv2.line(result, (start_x, start_y), (end_x, end_y), (0, 0, 0), 3)
                cv2.line(result, (start_x, start_y), (end_x, end_y), line_color, 1)
        
        # 绘制关键点
        head_points = {'鼻子', '左眼', '右眼', '左耳', '右耳'}
        for name, data in abs_kps.items():
            if not name.startswith("_"):
                x, y = data["x"], data["y"]
                conf = data["confidence"]
                vis = data.get("visibility", 1)
                # 画面外关键点(-1)不绘制点和标签
                if vis < 0:
                    continue
                # visibility=0 也绘制出来，采用红色醒目标记
                radius = int(4 + max(0.0, conf) * 4)
                if name in head_points:
                    # 头部关键点不做可见性染色，统一按固定颜色显示
                    cv2.circle(result, (x, y), radius + 2, (0, 0, 0), -1)
                    cv2.circle(result, (x, y), radius, colors.get(name, (255, 255, 255)), -1)
                elif vis == 0:
                    cv2.circle(result, (x, y), radius + 2, (0, 0, 0), -1)
                    cv2.circle(result, (x, y), radius, (0, 0, 255), -1)
                else:
                    cv2.circle(result, (x, y), radius + 2, (0, 0, 0), -1)
                    cv2.circle(result, (x, y), radius, colors.get(name, (255, 255, 255)), -1)

                label = f"{name} ({conf:.2f},v={vis})"
                cv2.putText(result, label, (x + 10, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(result, label, (x + 10, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 显示或保存
        result = self.draw_mixtask_boxes(result)

        if save_path:
            cv2.imwrite(save_path, result)
            print(f"结果已保存到: {save_path}")
        
        if show:
            cv2.imshow('Basic Keypoints', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result
    
    def create_body_silhouette(self, save_path=None):
        """
        创建人体轮廓形态
        """
        # 创建副本
        result = self.img.copy()
        overlay = self.img.copy()
        
        # 获取绝对坐标
        abs_kps = self.get_absolute_coordinates()
        
        # 定义身体部位多边形
        body_parts = {
            'head': ['左耳', '左眼', '鼻子', '右眼', '右耳'],
            'torso': ['左肩', '右肩', '右髋', '左髋'],
            'left_upper_arm': ['左肩', '左手肘', '左手腕'],
            'right_upper_arm': ['右肩', '右肘', '右腕'],
            'left_lower_arm': ['左手肘', '左手腕'],
            'right_lower_arm': ['右肘', '右腕'],
            'left_leg': ['左髋', '左膝', '左脚踝'],
            'right_leg': ['右髋', '右膝', '右脚踝']
        }
        
        # 定义部位颜色 BGRA
        part_colors = {
            'head': (255, 229, 204, 180),
            'torso': (102, 178, 255, 180),
            'left_upper_arm': (255, 178, 102, 180),
            'right_upper_arm': (153, 255, 153, 180),
            'left_lower_arm': (255, 178, 102, 180),
            'right_lower_arm': (178, 102, 255, 180),
            'left_leg': (255, 153, 204, 180),
            'right_leg': (153, 204, 255, 180)
        }

        head_hull = None
        torso_hull = None
        head_poly_color = (0, 255, 0)
        torso_poly_color = (0, 255, 0)
        # 绿色线条在 alpha 混合后会变浅，记录后在 result 上重绘为不透明
        opaque_green_lines = []
        opaque_green_points = []

        def _edge_center_from_hull(hull_pts, mode='bottom', tol=2):
            """
            从凸包点中估计边中心点：
            - mode='bottom': 取底边中心
            - mode='top': 取顶边中心
            """
            pts = hull_pts.reshape(-1, 2)
            ys = pts[:, 1]
            if mode == 'bottom':
                y_target = ys.max()
            else:
                y_target = ys.min()
            sel = pts[np.abs(ys - y_target) <= tol]
            if sel.size == 0:
                sel = pts[ys == y_target]
            x = int(np.mean(sel[:, 0]))
            y = int(np.mean(sel[:, 1]))
            return (x, y)
        
        # 绘制每个身体部位
        for part_name, point_names in body_parts.items():
            # 收集点（包含可见性）
            points = []
            for name in point_names:
                if name in abs_kps:
                    points.append(
                        {
                            "x": abs_kps[name]["x"],
                            "y": abs_kps[name]["y"],
                            "visibility": abs_kps[name].get("visibility", 1),
                        }
                    )
            
            if len(points) >= 2:
                # 对于四肢，绘制有宽度的线条
                if part_name in ['left_upper_arm', 'right_upper_arm', 
                               'left_lower_arm', 'right_lower_arm',
                               'left_leg', 'right_leg'] and len(points) >= 2:
                    # 绘制有宽度的线条
                    for i in range(len(points) - 1):
                        start = points[i]
                        end = points[i + 1]
                        # 任一端点在画面外(-1)则不绘制该线段与该端点
                        if start["visibility"] < 0 or end["visibility"] < 0:
                            continue
                        start_pt = (start["x"], start["y"])
                        end_pt = (end["x"], end["y"])
                        
                        # 根据部位设置不同宽度
                        if 'arm' in part_name:
                            thickness = 15
                        elif 'leg' in part_name:
                            thickness = 20
                        else:
                            thickness = 10

                        # visibility=0 的端点参与的边使用红色；可见边统一绿色
                        seg_color = (0, 0, 255) if (start["visibility"] == 0 or end["visibility"] == 0) else (0, 255, 0)
                        cv2.line(overlay, start_pt, end_pt, seg_color, thickness)
                        if seg_color == (0, 255, 0):
                            opaque_green_lines.append((start_pt, end_pt, thickness))
                        
                        # 绘制圆形端点
                        start_color = (0, 0, 255) if start["visibility"] == 0 else (0, 255, 0)
                        end_color = (0, 0, 255) if end["visibility"] == 0 else (0, 255, 0)
                        cv2.circle(overlay, start_pt, thickness//2, start_color, -1)
                        cv2.circle(overlay, end_pt, thickness//2, end_color, -1)
                        if start_color == (0, 255, 0):
                            opaque_green_points.append((start_pt, thickness // 2))
                        if end_color == (0, 255, 0):
                            opaque_green_points.append((end_pt, thickness // 2))
                
                # 对于头部和躯干，绘制多边形
                elif part_name in ['head', 'torso'] and len(points) >= 3:
                    # 画面外(-1)关键点不参与多边形
                    valid_points = [p for p in points if p["visibility"] >= 0]
                    if len(valid_points) < 3:
                        continue
                    pts = np.array([(p["x"], p["y"]) for p in valid_points], dtype=np.int32)
                    if part_name == 'head':
                        # 头部仅当左眼/右眼/鼻子都不可见时标红，否则保持绿色
                        key_vis = [
                            abs_kps.get('左眼', {}).get("visibility", 1),
                            abs_kps.get('右眼', {}).get("visibility", 1),
                            abs_kps.get('鼻子', {}).get("visibility", 1),
                        ]
                        head_all_core_invisible = all(v == 0 for v in key_vis)
                        poly_color = (0, 0, 255) if head_all_core_invisible else (0, 255, 0)
                    else:
                        has_invisible = any(p["visibility"] == 0 for p in valid_points)
                        poly_color = (0, 0, 255) if has_invisible else (0, 255, 0)
                    
                    # 创建凸包以获得更好的形状
                    if len(points) >= 3:
                        hull = cv2.convexHull(pts)
                        cv2.fillPoly(overlay, [hull], poly_color)
                        if part_name == 'head':
                            head_hull = hull
                            head_poly_color = poly_color
                        elif part_name == 'torso':
                            torso_hull = hull
                            torso_poly_color = poly_color

        # 混合图像
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        for start_pt, end_pt, thickness in opaque_green_lines:
            cv2.line(result, start_pt, end_pt, (0, 255, 0), thickness)
        for center, radius in opaque_green_points:
            cv2.circle(result, center, radius, (0, 255, 0), -1)
        
        # 绘制骨架连接
        skeleton = [
            ('鼻子', '左眼'), ('鼻子', '右眼'),
            ('左眼', '左耳'), ('右眼', '右耳'),
            ('左肩', '右肩'),
            ('左肩', '左手肘'), ('左手肘', '左手腕'),
            ('右肩', '右肘'), ('右肘', '右腕'),
            ('左肩', '左髋'), ('右肩', '右髋'),
            ('左髋', '左膝'), ('左膝', '左脚踝'),
            ('右髋', '右膝'), ('右膝', '右脚踝'),
            ('左髋', '右髋')
        ]
        
        for start_name, end_name in skeleton:
            if start_name in abs_kps and end_name in abs_kps:
                
                start_x = abs_kps[start_name]["x"]
                start_y = abs_kps[start_name]["y"]
                end_x = abs_kps[end_name]["x"]
                end_y = abs_kps[end_name]["y"]
                start_vis = abs_kps[start_name].get("visibility", 1)
                end_vis = abs_kps[end_name].get("visibility", 1)
                # 任一端点在画面外(-1)则不绘制该连线
                if start_vis < 0 or end_vis < 0:
                    continue
                line_color = (0, 0, 255) if (start_vis == 0 or end_vis == 0) else (0, 255, 0)
                cv2.line(result, (start_x, start_y), (end_x, end_y), line_color, 2)
        
        # 绘制关键点
        head_points = {'鼻子', '左眼', '右眼', '左耳', '右耳'}
        for name, data in abs_kps.items():
            if not name.startswith("_"):
                x, y = data["x"], data["y"]
                vis = data.get("visibility", 1)
                # 画面外关键点(-1)不绘制
                if vis < 0:
                    continue
                if name in head_points:
                    # 头部关键点不染色，统一白底黑心
                    cv2.circle(result, (x, y), 6, (255, 255, 255), -1)
                    cv2.circle(result, (x, y), 4, (0, 0, 0), -1)
                elif vis == 0:
                    cv2.circle(result, (x, y), 6, (0, 0, 255), -1)
                    cv2.circle(result, (x, y), 4, (255, 255, 255), -1)
                else:
                    cv2.circle(result, (x, y), 6, (0, 255, 0), -1)
                    cv2.circle(result, (x, y), 4, (255, 255, 255), -1)
        
        result = self.draw_mixtask_boxes(result)

        if save_path:
            cv2.imwrite(save_path, result)
            print(f"结果已保存到: {save_path}")
        
        # cv2.imshow('Body Silhouette', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return result

    @staticmethod
    def concat_with_gt_right(vis_img, gt_img):
        """
        将 GT 图拼接到可视化图右侧，按高度对齐。
        """
        if gt_img is None:
            return vis_img
        vh, vw = vis_img.shape[:2]
        gh, gw = gt_img.shape[:2]
        if gh != vh:
            scale = vh / float(gh)
            new_w = max(1, int(gw * scale))
            gt_img = cv2.resize(gt_img, (new_w, vh), interpolation=cv2.INTER_AREA)
        return np.hstack([vis_img, gt_img])
    
    def visualize(self, output_dir, prefix="output", gt_image_path=None):
        # print("=" * 50)
        # print("开始关键点可视化处理")
        # print("=" * 50)
        
        # # 显示关键点信息
        # print("\n关键点信息:")
        # print("-" * 30)
        # abs_kps = self.get_absolute_coordinates()
        # for name, data in abs_kps.items():
        #     if name != "_box":
        #         print(f"{name}: ({data['x']}, {data['y']}), 置信度: {data['confidence']:.3f}")
        vis = self.create_body_silhouette()
        if gt_image_path:
            gt_img = cv2.imread(gt_image_path)
            if gt_img is not None:
                vis = self.concat_with_gt_right(vis, gt_img)
        cv2.imwrite(os.path.join(output_dir, f"{prefix}.jpg"), vis)
        

if __name__ == "__main__":
    args = parse_args()

    if not args.json_dir and not args.json_dir_glob:
        raise ValueError("Please provide --json-dir or --json-dir-glob")
    if args.json_dir and args.json_dir_glob:
        raise ValueError("Use only one of --json-dir or --json-dir-glob")

    image_exts = [x.strip().lower() for x in args.image_exts.split(",") if x.strip()]
    image_path_obj = Path(args.image_dir)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"--image-dir not found: {image_path_obj}")
    gt_image_path_obj = None
    if args.gt_image_dir:
        gt_image_path_obj = Path(args.gt_image_dir)
        if not gt_image_path_obj.exists():
            raise FileNotFoundError(f"--gt-image-dir not found: {gt_image_path_obj}")

    if args.json_dir:
        folders = [args.json_dir]
    else:
        folders = sorted(glob.glob(args.json_dir_glob))

    if not folders:
        raise FileNotFoundError("No json folders found.")
    located_bbox_map = load_located_bbox_map(args.bbox_json) if args.mixtask else {}
    print(f"Found {len(folders)} json folder(s).")

    for json_dir in folders:
        json_path_obj = Path(json_dir)
        if not json_path_obj.exists():
            print(f"Warning: json dir not found, skip: {json_dir}")
            continue

        if args.output_dir:
            if len(folders) > 1:
                output_dir = str(Path(args.output_dir) / json_path_obj.name)
            else:
                output_dir = args.output_dir
        else:
            output_dir = f"{json_dir}{args.output_suffix}"
        os.makedirs(output_dir, exist_ok=True)

        json_files = sorted(json_path_obj.glob("*.json"))
        print(f"[{json_dir}] 找到 {len(json_files)} 个 JSON 文件，开始处理...")

        for json_file in json_files:
            prefix = json_file.stem
            image_path = None
            for ext in image_exts:
                temp_path = image_path_obj / f"{prefix}{ext}"
                if temp_path.exists():
                    image_path = str(temp_path)
                    break

            if image_path:
                print(f"正在处理: {prefix} -> 匹配到图像: {Path(image_path).name}")
                try:
                    image_name = Path(image_path).name
                    gt_bbox_xyxy = located_bbox_map.get(image_name) or located_bbox_map.get(prefix)
                    visualizer = KeypointVisualizer(
                        image_path,
                        str(json_file),
                        mixtask=args.mixtask,
                        gt_bbox_xyxy=gt_bbox_xyxy,
                    )
                    gt_image_path = None
                    if gt_image_path_obj is not None:
                        for ext in image_exts:
                            cand = gt_image_path_obj / f"{prefix}{ext}"
                            if cand.exists():
                                gt_image_path = str(cand)
                                break
                    visualizer.visualize(output_dir, prefix, gt_image_path=gt_image_path)
                except Exception as e:
                    print(f"处理 {prefix} 时出错: {e}")
            else:
                print(f"警告: 未能为 {json_file.name} 找到对应的图像文件，跳过。")
