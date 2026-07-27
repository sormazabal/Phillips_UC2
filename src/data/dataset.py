import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split: str = "train", img_size: tuple = (512, 512)):
    """
    Get albumentations data transformation pipeline.
    Applies standard ImageNet normalization matching HF Vision Transformer / SegFormer requirements.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['bbox_labels'],
                min_visibility=0.1
            )
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['bbox_labels'],
                min_visibility=0.1
            )
        )


class ArcadeDataset(Dataset):
    """
    PyTorch Dataset for ARCADE XCA dataset.
    Combines vessel segmentation (from syntax annotations) and stenosis object detection (from stenosis annotations).
    """
    def __init__(self, data_dir: str, split: str = "train", img_size: tuple = (512, 512), transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.transforms = transforms if transforms is not None else get_transforms(split, img_size)

        # Paths
        syntax_img_dir = os.path.join(data_dir, "syntax", split, "images")
        syntax_ann_file = os.path.join(data_dir, "syntax", split, "annotations", f"{split}.json")

        stenosis_ann_file = os.path.join(data_dir, "stenosis", split, "annotations", f"{split}.json")

        self.img_dir = syntax_img_dir

        # Load syntax annotations (Vessel tree segmentation)
        with open(syntax_ann_file, "r") as f:
            syntax_coco = json.load(f)

        # Load stenosis annotations (Stenosis object detection)
        if os.path.exists(stenosis_ann_file):
            with open(stenosis_ann_file, "r") as f:
                stenosis_coco = json.load(f)
        else:
            stenosis_coco = {"annotations": []}

        # Index images by image_id
        self.images = {img["id"]: img for img in syntax_coco.get("images", [])}
        self.image_ids = list(self.images.keys())

        # Group segmentation annotations by image_id
        self.seg_anns = {}
        for ann in syntax_coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.seg_anns:
                self.seg_anns[img_id] = []
            self.seg_anns[img_id].append(ann)

        # Group stenosis detection annotations by image_id
        self.det_anns = {}
        for ann in stenosis_coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.det_anns:
                self.det_anns[img_id] = []
            self.det_anns[img_id].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def _polygon_to_mask(self, polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            if isinstance(poly, list):
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        return mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]

        # Load image
        img_path = os.path.join(self.img_dir, file_name)
        if not os.path.exists(img_path):
            # Fallback if image path has different relative directory
            img_path = os.path.join(self.data_dir, "stenosis", self.split, "images", file_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate segmentation mask
        seg_mask = np.zeros((height, width), dtype=np.float32)
        if img_id in self.seg_anns:
            for ann in self.seg_anns[img_id]:
                if "segmentation" in ann:
                    poly_mask = self._polygon_to_mask(ann["segmentation"], height, width)
                    seg_mask = np.maximum(seg_mask, poly_mask)

        # Generate bounding boxes [x_min, y_min, x_max, y_max] in Pascal VOC format
        bboxes = []
        bbox_labels = []
        if img_id in self.det_anns:
            for ann in self.det_anns[img_id]:
                if "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                    x_min = max(0, x)
                    y_min = max(0, y)
                    x_max = min(width, x + w)
                    y_max = min(height, y + h)

                    # Ensure valid box dimensions
                    if x_max > x_min and y_max > y_min:
                        bboxes.append([x_min, y_min, x_max, y_max])
                        bbox_labels.append(1)  # Class 1: coronary_stenosis

        # Apply Albumentations
        transformed = self.transforms(
            image=image,
            mask=seg_mask,
            bboxes=bboxes,
            bbox_labels=bbox_labels
        )

        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"].unsqueeze(0)  # Shape [1, H, W]

        transformed_bboxes = transformed["bboxes"]
        transformed_labels = transformed["bbox_labels"]

        if len(transformed_bboxes) > 0:
            boxes_tensor = torch.tensor(transformed_bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(transformed_labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": transformed_image,
            "mask": transformed_mask,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": img_id,
            "file_name": file_name
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    file_names = [item["file_name"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "images": images,
        "masks": masks,
        "boxes": boxes,
        "labels": labels,
        "file_names": file_names,
        "image_ids": image_ids
    }
