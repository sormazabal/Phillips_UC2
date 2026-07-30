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

    syntax/<split>/images and stenosis/<split>/images are DIFFERENT physical frames that
    happen to reuse the same file_names/ids per split. So this dataset does NOT join the two
    COCO sources by image id. Instead each source is indexed against its own images list and
    its own annotations, and the two resulting sample sets are concatenated: every sample is
    tagged task="seg" (from syntax, real mask, no boxes) or task="det" (from det_ann_subset,
    real boxes, no mask). Never mix images from one source with labels from the other.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: tuple = (512, 512),
        transforms=None,
        det_ann_subset: str = "stenosis",
        det_categories: list = None,
        img_dir: str = None,
        seg_ann_file: str = None,
        det_ann_file: str = None,
        det_img_dir: str = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.transforms = transforms if transforms is not None else get_transforms(split, img_size)

        # Paths (overridable for non-ARCADE COCO sources; defaults preserve ARCADE layout).
        # Overrides may contain a "{split}" placeholder (e.g. one converter script's output
        # reused across train/val/test) -- .format() is a no-op on paths without one.
        syntax_img_dir = (img_dir.format(split=split) if img_dir else os.path.join(data_dir, "syntax", split, "images"))
        syntax_ann_file = (seg_ann_file.format(split=split) if seg_ann_file
                           else os.path.join(data_dir, "syntax", split, "annotations", f"{split}.json"))
        det_img_dir = (det_img_dir.format(split=split) if det_img_dir
                       else os.path.join(data_dir, det_ann_subset, split, "images"))
        det_ann_file = (det_ann_file.format(split=split) if det_ann_file
                        else os.path.join(data_dir, det_ann_subset, split, "annotations", f"{split}.json"))

        # Load segmentation annotations (vessel tree, from the syntax subset)
        with open(syntax_ann_file, "r") as f:
            syntax_coco = json.load(f)

        # Load detection annotations (from det_ann_subset, e.g. "stenosis" or "syntax")
        if os.path.exists(det_ann_file):
            with open(det_ann_file, "r") as f:
                det_coco = json.load(f)
        else:
            det_coco = {"images": [], "annotations": [], "categories": []}

        # Map COCO category_id -> contiguous label (1..K). If det_categories is not
        # given, every detection annotation is Class 1 (today's single-class behavior).
        if det_categories:
            name_to_id = {c["name"]: c["id"] for c in det_coco.get("categories", [])}
            self.cat2label = {name_to_id[name]: i + 1 for i, name in enumerate(det_categories) if name in name_to_id}
        else:
            self.cat2label = None

        # Group segmentation annotations by their own image_id (syntax's id space)
        seg_anns = {}
        for ann in syntax_coco.get("annotations", []):
            seg_anns.setdefault(ann["image_id"], []).append(ann)

        # Group detection annotations by their own image_id (det subset's id space)
        det_anns = {}
        for ann in det_coco.get("annotations", []):
            det_anns.setdefault(ann["image_id"], []).append(ann)

        # Two independent, self-consistent sample lists: each entry pairs an image
        # with annotations from the SAME COCO source's images/annotations arrays.
        self.samples = []
        for img in syntax_coco.get("images", []):
            self.samples.append({
                "task": "seg",
                "img_dir": syntax_img_dir,
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
                "anns": seg_anns.get(img["id"], []),
            })
        for img in det_coco.get("images", []):
            self.samples.append({
                "task": "det",
                "img_dir": det_img_dir,
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
                "anns": det_anns.get(img["id"], []),
            })

    def __len__(self):
        return len(self.samples)

    def _polygon_to_mask(self, polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            if isinstance(poly, list):
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        return mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_name = sample["file_name"]
        height = sample["height"]
        width = sample["width"]

        # Load image from this sample's own directory (never the other task's)
        img_path = os.path.join(sample["img_dir"], file_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        seg_mask = np.zeros((height, width), dtype=np.float32)
        bboxes = []
        bbox_labels = []
        has_seg = False
        has_det = False

        if sample["task"] == "seg":
            has_seg = True
            for ann in sample["anns"]:
                if "segmentation" in ann:
                    poly_mask = self._polygon_to_mask(ann["segmentation"], height, width)
                    seg_mask = np.maximum(seg_mask, poly_mask)
        else:
            has_det = True
            for ann in sample["anns"]:
                if "bbox" not in ann:
                    continue
                if self.cat2label is not None:
                    label = self.cat2label.get(ann["category_id"])
                    if label is None:
                        continue  # category not in det_categories, skip annotation
                else:
                    label = 1

                x, y, w, h = ann["bbox"]
                x_min = max(0, x)
                y_min = max(0, y)
                x_max = min(width, x + w)
                y_max = min(height, y + h)

                # Ensure valid box dimensions
                if x_max > x_min and y_max > y_min:
                    bboxes.append([x_min, y_min, x_max, y_max])
                    bbox_labels.append(label)

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
            "has_seg": has_seg,
            "has_det": has_det,
            "image_id": sample["image_id"],
            "file_name": file_name,
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    file_names = [item["file_name"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    has_seg = torch.tensor([item["has_seg"] for item in batch], dtype=torch.bool)
    has_det = torch.tensor([item["has_det"] for item in batch], dtype=torch.bool)

    return {
        "images": images,
        "masks": masks,
        "boxes": boxes,
        "labels": labels,
        "file_names": file_names,
        "image_ids": image_ids,
        "has_seg": has_seg,
        "has_det": has_det,
    }


if __name__ == "__main__":
    # Self-check: no cross-join between syntax and stenosis sources for a given split.
    import hashlib

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Arcade")
    split = "test"

    with open(os.path.join(data_dir, "syntax", split, "annotations", f"{split}.json")) as f:
        n_syntax = len(json.load(f)["images"])
    with open(os.path.join(data_dir, "stenosis", split, "annotations", f"{split}.json")) as f:
        n_stenosis = len(json.load(f)["images"])

    ds = ArcadeDataset(data_dir=data_dir, split=split)
    assert len(ds) == n_syntax + n_stenosis, f"expected {n_syntax + n_stenosis}, got {len(ds)}"
    print(f"OK: len(dataset) == {n_syntax} syntax + {n_stenosis} stenosis == {len(ds)}")

    def md5(path):
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    seen_seg = seen_det = 0
    for sample in ds.samples:
        img_path = os.path.join(sample["img_dir"], sample["file_name"])
        expected_dir = "syntax" if sample["task"] == "seg" else "stenosis"
        assert expected_dir in sample["img_dir"].replace("\\", "/"), sample["img_dir"]
        assert md5(img_path) == md5(os.path.join(data_dir, expected_dir, split, "images", sample["file_name"]))
        if sample["task"] == "seg":
            seen_seg += 1
        else:
            seen_det += 1
    assert seen_seg == n_syntax and seen_det == n_stenosis
    print(f"OK: every seg sample loads from syntax/, every det sample loads from stenosis/ ({seen_seg} + {seen_det})")

    for i in range(len(ds)):
        item = ds[i]
        assert item["has_seg"] != item["has_det"], "sample must have exactly one of has_seg/has_det"
    print("OK: every sample has exactly one of has_seg/has_det set")
