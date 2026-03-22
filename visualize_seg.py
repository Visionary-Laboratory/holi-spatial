#!/usr/bin/env python3
"""
分割结果可视化脚本：按图片画出分割 mask 与图例。

用法（三个参数顺序固定）：
  python visualize_seg.py <图片目录> <分割JSON路径> <输出目录>

示例：
  python visualize_seg.py ./images/ ./masks/xxx.json ./out/

输出：<输出目录>/seg/ 下为分割图，<输出目录>/tubiao/ 下为图例。
依赖：pip install numpy pillow pycocotools-mask
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as coco_mask
except ImportError:
    raise ImportError("请安装 pycocotools: pip install pycocotools-mask")

# 固定颜色表：约50种鲜艳、区分度高的颜色，序号 0,1,2,... 对应固定颜色
# 格式 (R, G, B)，便于与 numpy 数组配合
COLOR_TABLE = [
    (255, 0, 0),      # 0 红
    (0, 255, 0),       # 1 绿
    (0, 0, 255),       # 2 蓝
    (255, 255, 0),     # 3 黄
    (255, 0, 255),     # 4 品红
    (0, 255, 255),     # 5 青
    (255, 128, 0),     # 6 橙
    (128, 0, 255),     # 7 紫
    (255, 0, 128),     # 8 玫红
    (0, 255, 128),     # 9 春绿
    (128, 255, 0),     # 10 黄绿
    (0, 128, 255),     # 11 天蓝
    (255, 128, 128),   # 12 浅红
    (128, 255, 128),   # 13 浅绿
    (128, 128, 255),   # 14 浅蓝
    (255, 255, 128),   # 15 浅黄
    (255, 128, 255),   # 16 浅品红
    (128, 255, 255),   # 17 浅青
    (255, 165, 0),     # 18 橙黄
    (255, 69, 0),      # 19 橙红
    (50, 205, 50),     # 20 酸橙绿
    (0, 206, 209),     # 21 深青
    (255, 105, 180),   # 22 热粉
    (147, 112, 219),   # 23 中紫
    (60, 179, 113),    # 24 中海绿
    (30, 144, 255),    # 25 道奇蓝
    (255, 215, 0),     # 26 金
    (218, 112, 214),   # 27 兰花粉
    (70, 130, 180),    # 28 钢蓝
    (255, 99, 71),     # 29 番茄红
    (124, 252, 0),     # 30 草绿
    (0, 191, 255),     # 31 深天蓝
    (238, 130, 238),   # 32 紫罗兰
    (154, 205, 50),    # 33 黄绿
    (255, 20, 147),    # 34 深粉
    (72, 61, 139),     # 35 暗蓝紫
    (34, 139, 34),     # 36 森林绿
    (255, 140, 0),     # 37 深橙
    (210, 105, 30),    # 38 巧克力
    (0, 100, 0),       # 39 深绿
    (178, 34, 34),     # 40 砖红
    (75, 0, 130),      # 41 靛蓝
    (255, 69, 58),     # 42 朱红
    (0, 250, 154),     # 43 中春绿
    (186, 85, 211),    # 44 中兰
    (199, 21, 133),    # 45 中紫红
    (255, 160, 122),   # 46 浅鲑
    (64, 224, 208),    # 47 青绿
    (240, 230, 140),   # 48 卡其
    (220, 20, 60),     # 49 猩红
]
NUM_COLORS = len(COLOR_TABLE)


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_to_items(data):
    """
    将 JSON 数据按图片名聚合为: image_name -> [(label, mask_rle), ...]
    兼容两种格式：
      1) 每个 record 下 items 里含有多个不同 image（我们当前的 scene 级 JSON）
      2) 每个 record 只对应一张图，scene 作为图片名（旧格式）
    """
    image_items = defaultdict(list)
    for record in data:
        items = record.get("items", [])
        if not items:
            continue
        for item in items:
            # 优先使用 item 自己的 image 字段
            image_name = item.get("image") or item.get("image_name")
            if not image_name:
                # 回退：用 record 的 scene 字段
                image_name = record.get("scene", "")
                if image_name and not os.path.splitext(image_name)[1]:
                    image_name = image_name + ".png"
            if not image_name:
                continue
            label = item.get("label", "unknown")
            mask_rle = item.get("mask_rle")
            if mask_rle is None:
                continue
            image_items[image_name].append((label, mask_rle))
    return dict(image_items)


def decode_rle(mask_rle, h, w, _log_fail=True):
    """
    将 RLE 解码为二值 mask (H, W)，值为 0 或 1。
    传入画布 h,w 以便解码失败时尝试 size 互换（有的数据存的是 [W,H]）。
    """
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if size is None or counts is None:
        return None
    # 保证传给 pycocotools 的是标准 dict，且 size 为 [height, width]
    rle = {"size": [int(size[0]), int(size[1])], "counts": counts}
    err = None
    try:
        mask = coco_mask.decode(rle)
    except Exception as e:
        err = e
        try:
            rle = {"size": [int(size[1]), int(size[0])], "counts": counts}
            mask = coco_mask.decode(rle)
            err = None
        except Exception as e2:
            if _log_fail:
                import sys
                print(f"[RLE 解码失败] {type(e).__name__}: {e}", file=sys.stderr)
            return None
    if mask is None or mask.size == 0:
        return None
    return np.ascontiguousarray(mask)


def build_global_label_to_color_index(image_items):
    """
    根据所有图片中的 label 建立全局 label -> 颜色索引 映射，
    保证同一 label 在不同图片中颜色一致。
    返回 (label_to_idx, global_labels)，global_labels 为排序后的全体 label 列表。
    """
    all_labels = set()
    for items in image_items.values():
        for label, _ in items:
            if label != "__background__":
                all_labels.add(label)
    global_labels = sorted(all_labels)
    label_to_idx = {lb: i for i, lb in enumerate(global_labels)}
    return label_to_idx, global_labels


def draw_seg_for_image(image_path, items, color_table, out_seg_path, label_to_idx=None):
    """
    根据 items [(label, mask_rle), ...] 画一张分割图，保存到 out_seg_path。
    无 label 的区域显示原图，有 mask 的区域叠加上色。
    label_to_idx: 全局 label -> 颜色索引，若提供则同一 label 在所有图中颜色一致。
    """
    if not items:
        return None

    # 确定画布尺寸：用第一块 mask 的 size
    first_rle = items[0][1]
    h, w = first_rle["size"][0], first_rle["size"][1]

    # 用原图做底图，无 mask 处直接显示原图
    try:
        orig = np.array(Image.open(image_path).convert("RGB"))
        if orig.shape[0] != h or orig.shape[1] != w:
            orig = np.array(Image.fromarray(orig).resize((w, h), Image.BICUBIC))
        seg_canvas = orig.copy()
    except Exception:
        seg_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 本图出现的 label 去重并保持顺序（用于图例）；颜色用全局 label_to_idx
    order = []
    for label, _ in items:
        if label == "__background__":
            continue
        if label not in order:
            order.append(label)

    if not order:
        # 仅有 __background__ 时直接存原图
        Image.fromarray(seg_canvas).save(out_seg_path)
        return []

    decode_fail_count = 0
    log_once = [True]  # 只打印一次解码错误，避免刷屏
    for label, mask_rle in items:
        if label == "__background__":
            continue
        if label_to_idx is not None:
            idx = label_to_idx.get(label, 0)
        else:
            idx = order.index(label)  # 兼容未传 label_to_idx 时的按图内顺序
        color = color_table[idx % len(color_table)]
        mask = decode_rle(mask_rle, h, w, _log_fail=log_once[0])
        if mask is None:
            decode_fail_count += 1
            if log_once[0]:
                log_once[0] = False
            continue
        if mask.ndim == 3:
            mask = mask.squeeze()
        if mask.shape[0] != h or mask.shape[1] != w:
            # 尺寸不一致时尝试 resize（简单 nearest）
            from PIL import Image as PILImage
            mask = np.array(
                PILImage.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
            )
        mask_bool = mask.astype(bool)
        seg_canvas[mask_bool, 0] = color[0]
        seg_canvas[mask_bool, 1] = color[1]
        seg_canvas[mask_bool, 2] = color[2]

    Image.fromarray(seg_canvas).save(out_seg_path)
    return order  # 返回该图出现的 label 顺序，用于画图例


def draw_legend(labels, color_table, out_legend_path, patch_size=24, font_scale=0.5, label_to_idx=None):
    """
    画图例：每行一个颜色块 + 对应 label。
    label_to_idx 若提供，则按全局颜色索引绘制，保证与分割图一致。
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    n = len(labels)
    # 图例：色块高度 patch_size，文字行高约 patch_size，左边距留出色块
    line_height = max(patch_size, 28)
    img_h = n * line_height + 20
    img_w = 400
    legend_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    for i, label in enumerate(labels):
        y0 = 10 + i * line_height
        if label_to_idx is not None:
            idx = label_to_idx.get(label, i)
        else:
            idx = i
        color = color_table[idx % len(color_table)]
        # 色块
        y1 = y0 + patch_size
        legend_img[y0:y1, 10 : 10 + patch_size, 0] = color[0]
        legend_img[y0:y1, 10 : 10 + patch_size, 1] = color[1]
        legend_img[y0:y1, 10 : 10 + patch_size, 2] = color[2]
        # 文字
        text_x = 10 + patch_size + 10
        text_y = y0 + (patch_size + 14) // 2
        if has_cv2:
            cv2.putText(
                legend_img, str(label), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA
            )
        else:
            # 无 cv2 时用 PIL 画文字
            from PIL import ImageDraw, ImageFont
            pil_img = Image.fromarray(legend_img)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
            draw.text((text_x, y0), str(label), fill=(0, 0, 0), font=font)
            legend_img = np.array(pil_img)

    # 上面我们是按 (R, G, B) 写入颜色表到 legend_img 中的，
    # 因此这里直接按 RGB 存盘即可；若使用 cv2 画文字，其黑色文字对通道顺序无影响。
    Image.fromarray(legend_img).save(out_legend_path)


def main():
    parser = argparse.ArgumentParser(description="分割结果可视化：生成分割图与图例")
    parser.add_argument("images_dir", type=str, help="图片所在目录")
    parser.add_argument("json_path", type=str, help="分割结果 JSON 路径")
    parser.add_argument("output_dir", type=str, help="结果保存根目录（seg 与 tubiao 子目录会自动创建）")
    args = parser.parse_args()

    images_dir = args.images_dir.rstrip("/")
    output_dir = args.output_dir.rstrip("/")
    seg_dir = os.path.join(output_dir, "seg")
    tubiao_dir = os.path.join(output_dir, "tubiao")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(tubiao_dir, exist_ok=True)

    data = load_json(args.json_path)
    image_items = build_image_to_items(data)

    # 若 JSON 是 list of dict 且每个 dict 是单图多 segment 的格式（如 2dseg）
    if not image_items and isinstance(data, list):
        for record in data:
            scene = record.get("scene", "")
            items = record.get("items", [])
            if not items:
                continue
            # 用第一项的 image 作为该组图片名
            image_name = items[0].get("image") or items[0].get("image_name")
            if image_name:
                image_items[image_name] = [
                    (it.get("label", "unknown"), it.get("mask_rle"))
                    for it in items
                    if it.get("mask_rle") is not None
                ]

    if not image_items:
        print("未在 JSON 中找到任何 (image, label, mask_rle) 数据，请检查格式。")
        return

    # 先读取所有图片的 label，建立全局 label -> 颜色索引，保证同一 label 在所有图中颜色一致
    label_to_idx, global_labels = build_global_label_to_color_index(image_items)
    if global_labels:
        print(f"全局 label 共 {len(global_labels)} 个，已统一颜色映射。")

    # 遍历每张在 JSON 中出现的图
    for image_name, items in image_items.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"跳过（图片不存在）: {image_path}")
            continue

        base_name = os.path.splitext(image_name)[0]
        ext = os.path.splitext(image_name)[1] or ".png"
        out_seg_path = os.path.join(seg_dir, base_name + ext)
        out_legend_path = os.path.join(tubiao_dir, base_name + ext)

        order = draw_seg_for_image(
            image_path, items, COLOR_TABLE, out_seg_path, label_to_idx=label_to_idx
        )
        if order is not None:
            draw_legend(order, COLOR_TABLE, out_legend_path, label_to_idx=label_to_idx)
            print(f"已生成: {out_seg_path}, {out_legend_path}")


if __name__ == "__main__":
    main()
