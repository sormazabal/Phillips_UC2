import os
import sys
import json
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import nms

from src.models.hf_dual_net import HFDualNet
from src.models.lightning_module import ArcadeLightningModule


def preprocess_image(image_path: str, img_size: tuple = (512, 512)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0)  # Shape [1, 3, H, W]
    return input_tensor, (orig_h, orig_w)


DEFAULT_DET_CLASSES = ["coronary_stenosis"]


def load_model(checkpoint_path: str = None, backbone_name: str = "nvidia/mit-b5", det_classes: list = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_classes = det_classes or DEFAULT_DET_CLASSES

    if checkpoint_path and os.path.exists(checkpoint_path):
        # ponytail: strict=False — only lightning_model.model is used here, so a loss_fn
        # buffer mismatch (e.g. pos_weight added after older checkpoints were saved) is harmless.
        lightning_model = ArcadeLightningModule.load_from_checkpoint(checkpoint_path, strict=False)
        model = lightning_model.model
        det_classes = lightning_model.config.get("model", {}).get("det_classes", det_classes)
    else:
        model = HFDualNet(backbone_name=backbone_name, freeze_backbone=False, num_det_classes=len(det_classes))

    model.to(device)
    model.eval()
    model.det_classes = det_classes
    return model, device


def decode_detections(det_out, orig_h: int, orig_w: int, conf_threshold: float = 0.5, det_classes: list = None,
                       iou_threshold: float = 0.5):
    det_classes = det_classes or DEFAULT_DET_CLASSES
    num_classes = len(det_classes)
    B, C, H, W = det_out.shape

    pred_boxes = torch.sigmoid(det_out[0, :4, :, :])  # [4, 7, 7]
    pred_conf = torch.sigmoid(det_out[0, 4, :, :])   # [7, 7]
    pred_cls = det_out[0, 5:5 + num_classes, :, :]   # [K, 7, 7]

    devices = []
    instance_count = 1

    # Filter detections above confidence threshold
    high_conf_indices = torch.nonzero(pred_conf > conf_threshold)

    # Suppress duplicate boxes on the same object (adjacent grid cells often both
    # cross conf_threshold for one physical device) before building device entries.
    if high_conf_indices.shape[0] > 0:
        rows, cols = high_conf_indices[:, 0], high_conf_indices[:, 1]
        scores = pred_conf[rows, cols]
        boxes_px = torch.stack([
            pred_boxes[0, rows, cols] * orig_w,
            pred_boxes[1, rows, cols] * orig_h,
            pred_boxes[2, rows, cols] * orig_w,
            pred_boxes[3, rows, cols] * orig_h,
        ], dim=1)
        keep = nms(boxes_px, scores, iou_threshold)
        high_conf_indices = high_conf_indices[keep]

    for idx in high_conf_indices:
        r, c = idx[0].item(), idx[1].item()
        confidence = float(pred_conf[r, c].item())
        device_class = det_classes[int(torch.argmax(pred_cls[:, r, c]).item())]

        box = pred_boxes[:, r, c]
        x_min = float(box[0].item() * orig_w)
        y_min = float(box[1].item() * orig_h)
        x_max = float(box[2].item() * orig_w)
        y_max = float(box[3].item() * orig_h)

        # Ensure valid coordinates
        x_min = max(0.0, round(x_min, 2))
        y_min = max(0.0, round(y_min, 2))
        x_max = min(float(orig_w), round(x_max, 2))
        y_max = min(float(orig_h), round(y_max, 2))

        device_entry = {
            "device_class": device_class,
            "instance_id": f"target_{instance_count:02d}",
            "bounding_box": [x_min, y_min, x_max, y_max],
            "detection_confidence": round(confidence, 4)
        }

        # Severity is a confidence/area heuristic, not a learned label — ARCADE has no
        # severity ground truth. Only meaningful for the stenosis class.
        if device_class == "coronary_stenosis":
            box_area = (x_max - x_min) * (y_max - y_min)
            img_area = orig_h * orig_w
            area_ratio = box_area / (img_area + 1e-6)
            if confidence > 0.85 or area_ratio > 0.05:
                severity = "high"
            elif confidence > 0.65:
                severity = "moderate"
            else:
                severity = "low"
            device_entry["severity"] = severity
        else:
            device_entry["device_state"] = None

        devices.append(device_entry)
        instance_count += 1

    return devices


def run_inference(
    image_path: str,
    checkpoint_path: str = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    backbone_name: str = "nvidia/mit-b5",
    img_size: tuple = (512, 512),
):
    model, device = load_model(checkpoint_path, backbone_name)

    # Preprocess
    input_tensor, (orig_h, orig_w) = preprocess_image(image_path, img_size)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    devices = decode_detections(outputs["det_out"], orig_h, orig_w, conf_threshold, model.det_classes, iou_threshold)

    # Format JSON output according to target schema
    frame_id = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now(timezone.utc).isoformat()

    result_json = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "devices": devices
    }

    return result_json


def main():
    parser = argparse.ArgumentParser(description="Inference script for Arcade XCA Dual-Task Model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input XCA image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold for stenosis detection")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for NMS (suppresses overlapping duplicate boxes)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save output JSON file")
    args = parser.parse_args()

    result = run_inference(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        conf_threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh
    )

    json_str = json.dumps(result, indent=2)
    print(json_str)

    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(json_str)
        print(f"\nSaved inference output to {args.output_json}")


if __name__ == "__main__":
    main()
