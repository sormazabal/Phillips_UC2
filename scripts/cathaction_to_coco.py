"""Convert CathAction's segmentation_human_train (img/ + binary mask/ pairs) into
COCO-format train/val/test annotation files, reusing the original image files in
place (no copying). Boxes are derived from the binary masks via contour detection
-- CathAction ships no bounding-box annotations for this split.

Mask inspection (see plan) confirmed masks are single-channel binary (0/255), not
per-class -- so this yields one category, "device" (catheter+guidewire merged),
not two. Splits are by video prefix (the part of the filename before "_img-") so
frames from the same sequence never leak across train/val/test.
"""
import argparse
import glob
import json
import os
import random

import cv2
import numpy as np


def frames_by_video(img_dir):
    videos = {}
    for path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        file_name = os.path.basename(path)
        prefix = file_name.split("_img-")[0]
        videos.setdefault(prefix, []).append(file_name)
    return videos


def split_videos(videos, val_frac=0.1, test_frac=0.1, seed=0):
    prefixes = sorted(videos.keys())
    random.Random(seed).shuffle(prefixes)
    n = len(prefixes)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    return {
        "val": prefixes[:n_val],
        "test": prefixes[n_val:n_val + n_test],
        "train": prefixes[n_val + n_test:],
    }


def mask_to_annotations(mask_path, image_id, next_ann_id, min_area=20):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or len(contour) < 3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        polygon = contour.reshape(-1, 2).astype(np.float64).flatten().tolist()

        annotations.append({
            "id": next_ann_id(),
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [polygon],
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(area),
            "iscrowd": 0,
        })
    return annotations


def build_split(video_prefixes, videos, img_dir, mask_dir):
    images, annotations = [], []
    next_image_id = iter(range(1, 10 ** 8))
    next_ann_id_counter = iter(range(1, 10 ** 8))
    next_ann_id = lambda: next(next_ann_id_counter)

    for prefix in video_prefixes:
        for file_name in videos[prefix]:
            image_id = next(next_image_id)
            img_path = os.path.join(img_dir, file_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            images.append({
                "id": image_id,
                "file_name": file_name,
                "height": height,
                "width": width,
            })

            mask_name = os.path.splitext(file_name)[0] + "_mask.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                annotations.extend(mask_to_annotations(mask_path, image_id, next_ann_id))

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "device"}],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert CathAction masks to COCO train/val/test json")
    parser.add_argument("--root", default="CathAction/segmentation_human_train/human_dataset_train")
    parser.add_argument("--out_dir", default="CathAction/coco/annotations")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    img_dir = os.path.join(args.root, "img")
    mask_dir = os.path.join(args.root, "mask")

    videos = frames_by_video(img_dir)
    print(f"{len(videos)} video sequences, {sum(len(v) for v in videos.values())} frames")

    splits = split_videos(videos, args.val_frac, args.test_frac, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for split_name, prefixes in splits.items():
        coco = build_split(prefixes, videos, img_dir, mask_dir)
        out_path = os.path.join(args.out_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f)
        n_boxes = len(coco["annotations"])
        print(f"{split_name}: {len(prefixes)} videos, {len(coco['images'])} images, {n_boxes} boxes -> {out_path}")


if __name__ == "__main__":
    main()
