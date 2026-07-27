import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross Entropy (BCE) and Dice Loss for Semantic Segmentation.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, 1, H, W]
        # targets: [B, 1, H, W]
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # BCE Loss
        bce_loss = F.binary_cross_entropy(probs_flat, targets_flat)

        # Dice Loss
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)

        return bce_loss + dice_loss


class BBoxLoss(nn.Module):
    """
    Smooth L1 / GIoU Loss for bounding box regression and confidence BCE loss.
    """
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, det_out, targets_boxes, targets_labels):
        # det_out: [B, out_dim, 7, 7]
        # For simplicity in grid-based anchor-free target matching:
        # We compute objectness loss across grid cells and bbox regression for positive cells
        B, C, H, W = det_out.shape
        det_out = det_out.view(B, 6, H, W)  # 4 bbox + 1 conf + 1 class

        pred_boxes = torch.sigmoid(det_out[:, :4, :, :])
        pred_conf = det_out[:, 4, :, :]
        pred_cls = det_out[:, 5, :, :]

        target_conf = torch.zeros_like(pred_conf)
        target_boxes = torch.zeros_like(pred_boxes)
        target_cls = torch.zeros_like(pred_cls)

        # Construct grid targets from ground truth bounding boxes
        for i in range(B):
            boxes = targets_boxes[i]
            if len(boxes) > 0:
                for box in boxes:
                    # Normalize box [x_min, y_min, x_max, y_max] to [0, 1] assuming image size 512
                    x_min, y_min, x_max, y_max = box / 512.0
                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0

                    grid_x = min(int(center_x * W), W - 1)
                    grid_y = min(int(center_y * H), H - 1)

                    target_conf[i, grid_y, grid_x] = 1.0
                    target_boxes[i, :, grid_y, grid_x] = torch.tensor([x_min, y_min, x_max, y_max], device=det_out.device)
                    target_cls[i, grid_y, grid_x] = 1.0

        pos_mask = (target_conf > 0.5)

        # Confidence Loss
        conf_loss = self.bce(pred_conf, target_conf)

        # Classification Loss
        cls_loss = self.bce(pred_cls[pos_mask], target_cls[pos_mask]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=det_out.device)

        # BBox Regression Loss
        if pos_mask.sum() > 0:
            reg_loss = self.smooth_l1(pred_boxes.permute(0, 2, 3, 1)[pos_mask], target_boxes.permute(0, 2, 3, 1)[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=det_out.device)

        return reg_loss, conf_loss + cls_loss


class CombinedDualLoss(nn.Module):
    """
    Combined Total Loss for Dual-Task Learning:
    Total_Loss = lambda_seg * Dice_BCE_Loss + lambda_det * BBox_Regression_Loss + lambda_cls * Classification_Loss
    """
    def __init__(self, lambda_seg: float = 1.0, lambda_det: float = 1.0, lambda_cls: float = 1.0):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_det = lambda_det
        self.lambda_cls = lambda_cls

        self.seg_loss_fn = DiceBCELoss()
        self.det_loss_fn = BBoxLoss()

    def forward(self, outputs, target_masks, target_boxes, target_labels):
        seg_logits = outputs["seg_logits"]
        det_out = outputs["det_out"]

        # 1. Segmentation Loss
        seg_loss = self.seg_loss_fn(seg_logits, target_masks)

        # 2. BBox Regression & Classification Loss
        bbox_reg_loss, bbox_cls_loss = self.det_loss_fn(det_out, target_boxes, target_labels)

        # Total Weighted Loss
        total_loss = (
            (self.lambda_seg * seg_loss) +
            (self.lambda_det * bbox_reg_loss) +
            (self.lambda_cls * bbox_cls_loss)
        )

        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "bbox_reg_loss": bbox_reg_loss,
            "bbox_cls_loss": bbox_cls_loss
        }
