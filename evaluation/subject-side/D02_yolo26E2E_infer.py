# import os
# import json
# import pdb

# from tqdm import tqdm
# from ultralytics import YOLO


# def predict_box_and_keypoints(input_root, output_root, batch_size=8, device="cuda:0", pe_checkpoint = "<YOLO_POSE_CHECKPOINT>"):
#     # data prepare
#     input_img_path = sorted([
#         os.path.join(input_root, f) 
#         for f in os.listdir(input_root) 
#         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))
#         ])
#     grouped_list = [input_img_path[i:i+batch_size] for i in range(0, len(input_img_path), batch_size)]

#     ## Pose estimation
#     # model init
#     pose_estimater = YOLO(pe_checkpoint).to(device)

#     for path_batch in tqdm(grouped_list, desc="Processing files"):
#         ## pose estimation
#         pe_results = pose_estimater.predict(source=path_batch, imgsz=1024, verbose=True)
#         # pose estimation visualization
#         for ret, img_name in zip(pe_results, path_batch):
#             if not os.path.exists(os.path.join(output_root)):
#                 os.makedirs(os.path.join(output_root))
#             visual_filename = os.path.join(output_root, os.path.basename(img_name).replace(".", "-detpose."))
            
#             ret.plot()
#             ret.save(filename=visual_filename)

#             n_instances = len(ret)
#             boxes_xyxyn_np = ret.boxes.xyxyn.cpu().numpy()
#             boxes_conf_np = ret.boxes.conf.cpu().numpy()
#             keypoints_xyn_np = ret.keypoints.xyn.cpu().numpy()
#             keypoints_conf_np = ret.keypoints.conf.cpu().numpy()

#             # meta info
#             instance_info = [
#                 {
#                     "box_xyxyn": boxes_xyxyn_np[i].tolist(),
#                     "box_conf": float(boxes_conf_np[i]),
#                     "keypoints_xyn": keypoints_xyn_np[i].tolist(),
#                     "keypoints_conf": keypoints_conf_np[i].tolist(),
#                 }
#                 for i in range(n_instances)
#             ]

#             json_filename = os.path.splitext(visual_filename)[0] + '.json'
#             with open(json_filename, 'w') as f:
#                 json.dump({"instance_info": instance_info}, f, indent=2)

# if __name__ == '__main__':
#     input_root = "/data/images/pose_s0/2026-01-30_pa"
#     output_root= "/data/images/pose_s0/2026-01-30_pa_yolo26x_17kpt"
#     predict_box_and_keypoints(input_root, output_root)

import argparse
import os
import json
import torch
from tqdm import tqdm
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")

def init_worker(gpu_id, pe_checkpoint):
    """初始化工作进程，加载模型到指定GPU"""
    global pose_estimater
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    pose_estimater = YOLO(pe_checkpoint).to(device)
    # 设置当前进程只使用指定的GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)


def _is_image_file(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTENSIONS)


def _resolve_from_list(raw_path: str, input_root: str):
    raw_path = raw_path.strip()
    if not raw_path:
        return None
    if os.path.isabs(raw_path) and os.path.isfile(raw_path):
        return raw_path
    candidate = os.path.join(input_root, raw_path)
    if os.path.isfile(candidate):
        return candidate
    fallback = os.path.join(input_root, os.path.basename(raw_path))
    if os.path.isfile(fallback):
        return fallback
    return None


def collect_input_paths(input_root: str, list_file: str = ""):
    if not list_file:
        return sorted(
            [
                os.path.join(input_root, f)
                for f in os.listdir(input_root)
                if _is_image_file(f)
            ]
        ), 0

    not_found = 0
    paths = []
    seen = set()
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            resolved = _resolve_from_list(line, input_root)
            if resolved is None:
                not_found += 1
                continue
            if not _is_image_file(resolved):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return sorted(paths), not_found


def _build_visibility_from_conf(keypoints_conf, threshold=0.25):
    return [1 if i < 5 or float(conf) > threshold else -1 for i, conf in enumerate(keypoints_conf)]


def process_single_batch(path_batch, output_root, pe_checkpoint, gpu_id=0, visibility_threshold=0.25):
    """处理单个批次"""
    # 每个进程初始化自己的模型（如果不使用init_worker）
    if 'pose_estimater' not in globals():
        init_worker(gpu_id, pe_checkpoint)

    results = []

    try:
        # 预测
        pe_results = pose_estimater.predict(source=path_batch, imgsz=1024, verbose=False)

        for ret, img_name in zip(pe_results, path_batch):
            base_name = os.path.basename(img_name)
            ext = os.path.splitext(base_name)[1].lower()

            # 构造输出路径
            visual_name = base_name #.replace(ext, "-detpose" + ext)
            visual_filename = os.path.join(output_root, visual_name)
            json_filename = os.path.splitext(visual_filename)[0] + '.json'

            # 检查是否已存在
            if os.path.exists(json_filename):
                results.append(f"跳过: {base_name}")
                continue

            # 确保输出目录存在
            os.makedirs(output_root, exist_ok=True)

            # 可视化和保存
            ret.plot()
            ret.save(filename=visual_filename)

            # 提取数据
            n_instances = len(ret)
            if n_instances > 0:
                boxes_xyxyn_np = ret.boxes.xyxyn.cpu().numpy()
                boxes_conf_np = ret.boxes.conf.cpu().numpy()
                keypoints_xyn_np = ret.keypoints.xyn.cpu().numpy()
                keypoints_conf_np = ret.keypoints.conf.cpu().numpy()

                instance_info = [
                    {
                        "box_xyxyn": boxes_xyxyn_np[i].tolist(),
                        "box_conf": float(boxes_conf_np[i]),
                        "keypoints_xyn": keypoints_xyn_np[i].tolist(),
                        "keypoints_conf": keypoints_conf_np[i].tolist(),
                        "visibility": _build_visibility_from_conf(
                            keypoints_conf_np[i].tolist(),
                            visibility_threshold,
                        ),
                    }
                    for i in range(n_instances)
                ]
            else:
                instance_info = []

            # 保存JSON
            with open(json_filename, 'w') as f:
                json.dump({"instance_info": instance_info}, f, indent=2)

            results.append(f"完成: {base_name} ({n_instances} instances)")

    except Exception as e:
        results.append(f"错误处理批次 {path_batch}: {str(e)}")

    return results

def predict_box_and_keypoints_multi(
    input_root, 
    output_root, 
    batch_size=8, 
    num_workers=None,  # 默认使用所有GPU，或CPU核心数
    pe_checkpoint="<YOLO_POSE_CHECKPOINT>",
    gpu_ids=None,  # 指定使用的GPU，如 [0,1,2,3]，None则自动检测
    list_file="",  # 可选：每行一个图像路径/文件名
    visibility_threshold=0.25,
):
    """
    多进程姿态估计处理

    Args:
        input_root: 输入图像目录
        output_root: 输出目录
        batch_size: 每个批次的图像数量
        num_workers: 进程数，None则自动设置为GPU数量
        pe_checkpoint: YOLO模型路径
        gpu_ids: 使用的GPU ID列表，None则使用所有可用GPU
    """

    # 数据准备
    input_img_path, not_found = collect_input_paths(input_root=input_root, list_file=list_file)
    if list_file:
        print(f"清单文件: {list_file}，未命中本地文件: {not_found}")

    # 过滤已处理文件
    pending_img_path = []
    for img_path in input_img_path:
        base_name = os.path.basename(img_path)
        ext = os.path.splitext(base_name)[1].lower()
        json_name = base_name.replace(ext, ".json")
        json_path = os.path.join(output_root, json_name)

        if not os.path.exists(json_path):
            pending_img_path.append(img_path)

    skipped_count = len(input_img_path) - len(pending_img_path)
    print(f"总共 {len(input_img_path)} 个文件，跳过 {skipped_count} 个已存在，待处理 {len(pending_img_path)} 个")

    if len(pending_img_path) == 0:
        print("所有文件已处理完毕！")
        return

    # 确定GPU配置
    if gpu_ids is None:
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [0]  # CPU模式

    if num_workers is None:
        num_workers = len(gpu_ids)

    print(f"使用 {num_workers} 个进程，GPU: {gpu_ids}")

    # 分组批次
    grouped_list = [pending_img_path[i:i+batch_size] for i in range(0, len(pending_img_path), batch_size)]
    total_batches = len(grouped_list)
    print(f"总批次数: {total_batches} (每批 {batch_size} 张)")

    # 为每个批次分配GPU（轮询分配）
    batch_gpu_pairs = [(batch, gpu_ids[i % len(gpu_ids)]) for i, batch in enumerate(grouped_list)]

    # 多进程处理
    # 方法1: 使用ProcessPoolExecutor（推荐，更简洁）
    results_summary = []

    # 设置启动方法为spawn，避免CUDA多进程问题
    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_batch = {
            executor.submit(
                process_single_batch, 
                batch, 
                output_root, 
                pe_checkpoint, 
                gpu_id,
                visibility_threshold,
            ): (i, batch) 
            for i, (batch, gpu_id) in enumerate(batch_gpu_pairs)
        }

        # 使用tqdm显示进度
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                try:
                    result = future.result()
                    results_summary.extend(result)
                    pbar.update(1)
                    # 可选：显示当前批次信息
                    if len(result) > 0:
                        pbar.set_postfix({"latest": result[0][:30]})
                except Exception as e:
                    print(f"批次 {batch_idx} 处理失败: {e}")
                    pbar.update(1)

    # 打印统计
    success_count = sum(1 for r in results_summary if r.startswith("完成"))
    skip_count = sum(1 for r in results_summary if r.startswith("跳过"))
    error_count = sum(1 for r in results_summary if r.startswith("错误"))

    print(f"\n处理完成: 成功 {success_count}, 跳过 {skip_count}, 错误 {error_count}")


# 备选方案：使用torch.multiprocessing（更适合PyTorch）
def predict_box_and_keypoints_torch_mp(
    input_root, 
    output_root, 
    batch_size=8, 
    num_workers=None,
    pe_checkpoint="<YOLO_POSE_CHECKPOINT>",
    gpu_ids=None,
    list_file="",
    visibility_threshold=0.25,
):
    """使用torch.multiprocessing的版本，某些情况下更稳定"""
    import torch.multiprocessing as tmp

    # 数据准备（同上）
    input_img_path, not_found = collect_input_paths(input_root=input_root, list_file=list_file)
    if list_file:
        print(f"清单文件: {list_file}，未命中本地文件: {not_found}")

    pending_img_path = []
    for img_path in input_img_path:
        base_name = os.path.basename(img_path)
        ext = os.path.splitext(base_name)[1].lower()
        json_name = base_name.replace(ext, ".json")
        json_path = os.path.join(output_root, json_name)
        if not os.path.exists(json_path):
            pending_img_path.append(img_path)

    if len(pending_img_path) == 0:
        print("所有文件已处理完毕！")
        return

    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    if num_workers is None:
        num_workers = len(gpu_ids)

    grouped_list = [pending_img_path[i:i+batch_size] for i in range(0, len(pending_img_path), batch_size)]
    batch_gpu_pairs = [(batch, gpu_ids[i % len(gpu_ids)]) for i, batch in enumerate(grouped_list)]

    # 使用torch multiprocessing
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(0, pe_checkpoint)) as pool:
        # 使用starmap或imap
        results = list(tqdm(
            pool.imap(
                lambda x: process_single_batch(x[0], output_root, pe_checkpoint, x[1], visibility_threshold), 
                batch_gpu_pairs
            ),
            total=len(batch_gpu_pairs),
            desc="Processing"
        ))

    print("处理完成")


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26 pose E2E infer with optional list-file")
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--list-file", default="", help="Optional txt file, one image path/name per line")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pe-checkpoint", default=os.getenv("SUBJECT_YOLO_CHECKPOINT", ""))
    p.add_argument("--gpu-ids", default="0,1", help='Comma separated GPU ids, e.g. "0,1"; empty means auto')
    p.add_argument("--visibility-threshold", type=float, default=0.25, help="keypoint conf 大于该阈值时 visibility=1，否则为 -1")
    return p.parse_args()


def parse_gpu_ids(raw: str):
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


if __name__ == '__main__':
    args = parse_args()
    predict_box_and_keypoints_multi(
        input_root=args.input_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pe_checkpoint=args.pe_checkpoint,
        gpu_ids=parse_gpu_ids(args.gpu_ids),
        list_file=args.list_file,
        visibility_threshold=args.visibility_threshold,
    )
