import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ArcadeDataset, collate_fn
from src.models.hf_dual_net import HFDualNet
from src.models.lightning_module import ArcadeLightningModule


def dice_score(logits, targets, smooth=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + smooth) / (union + smooth)).tolist()


def match_detections(det_out, boxes, labels, img_size, conf_thresh, iou_thresh=0.3):
    """Grid-cell detection match: a GT box counts as detected if the head's
    predicted box in the grid cell containing the GT center clears conf_thresh
    and overlaps the GT box by >= iou_thresh. Mirrors the grid target-assignment
    the model is trained with (src/utils/losses.py BBoxLoss)."""
    B, C, H, W = det_out.shape
    det_out = det_out.view(B, 6, H, W)
    pred_boxes = torch.sigmoid(det_out[:, :4, :, :])
    pred_conf = torch.sigmoid(det_out[:, 4, :, :])

    tp = fp = fn = 0
    for i in range(B):
        gt_boxes = boxes[i]
        hit_cells = set()
        for box in gt_boxes:
            x_min, y_min, x_max, y_max = (box / img_size[1]).tolist()
            grid_x = min(int(((x_min + x_max) / 2.0) * W), W - 1)
            grid_y = min(int(((y_min + y_max) / 2.0) * H), H - 1)

            conf = pred_conf[i, grid_y, grid_x].item()
            if conf <= conf_thresh:
                fn += 1
                continue

            px_min, py_min, px_max, py_max = pred_boxes[i, :, grid_y, grid_x].tolist()
            inter_w = max(0.0, min(x_max, px_max) - max(x_min, px_min))
            inter_h = max(0.0, min(y_max, py_max) - max(y_min, py_min))
            inter = inter_w * inter_h
            union = (x_max - x_min) * (y_max - y_min) + (px_max - px_min) * (py_max - py_min) - inter
            iou = inter / union if union > 0 else 0.0

            if iou >= iou_thresh:
                tp += 1
            else:
                fn += 1
            hit_cells.add((grid_y, grid_x))

        # Any other high-confidence cell with no matching GT is a false positive
        pos = (pred_conf[i] > conf_thresh).nonzero().tolist()
        for r, c in pos:
            if (r, c) not in hit_cells:
                fp += 1

    return tp, fp, fn


@torch.no_grad()
def evaluate(config, checkpoint_path=None, conf_thresh=0.5, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path:
        model = ArcadeLightningModule.load_from_checkpoint(checkpoint_path, config=config).model
    else:
        # Zero-shot: pretrained backbone, randomly initialized task heads.
        model_cfg = config.get("model", {})
        model = HFDualNet(
            backbone_name=model_cfg.get("backbone", "nvidia/mit-b3"),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            num_seg_classes=model_cfg.get("num_seg_classes", 1),
            num_det_classes=model_cfg.get("num_det_classes", 1),
            backbone_kwargs=model_cfg.get("backbone_kwargs"),
        )
    model.to(device).eval()

    img_size = tuple(config.get("model", {}).get("img_size", [512, 512]))
    dataset = ArcadeDataset(data_dir=config["data"]["data_dir"], split=split, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=config["data"].get("batch_size", 8),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    dice_scores = []
    tp = fp = fn = 0
    for batch in loader:
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        outputs = model(images)

        dice_scores.extend(dice_score(outputs["seg_logits"], masks))
        b_tp, b_fp, b_fn = match_detections(outputs["det_out"], batch["boxes"], batch["labels"], img_size, conf_thresh)
        tp += b_tp
        fp += b_fp
        fn += b_fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "split": split,
        "num_images": len(dataset),
        "mean_dice_score": sum(dice_scores) / len(dice_scores) if dice_scores else 0.0,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "conf_threshold": conf_thresh,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARCADE dual-task model (zero-shot or from a checkpoint)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional .ckpt; omit for zero-shot pretrained backbone")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = evaluate(config, checkpoint_path=args.checkpoint, conf_thresh=args.conf_thresh, split=args.split)
    print(json.dumps(results, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
