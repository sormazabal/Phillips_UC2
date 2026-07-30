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
    Smooth L1 / GIoU Loss for bounding box regression, confidence BCE loss,
    and (when num_classes > 1) cross-entropy classification loss.
    """
    def __init__(self, num_classes: int = 1, img_size: float = 512.0, pos_weight: float = 45.0):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        # ponytail: fixed pos_weight tuned for the 7x7 (49-cell) grid, ~1 positive cell/image.
        # Recompute if grid size changes.
        self.conf_bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pos_weight))

    def forward(self, det_out, targets_boxes, targets_labels):
        # det_out: [B, 4 + 1 + num_classes, 7, 7]
        B, C, H, W = det_out.shape

        pred_boxes = torch.sigmoid(det_out[:, :4, :, :])
        pred_conf = det_out[:, 4, :, :]
        pred_cls = det_out[:, 5:5 + self.num_classes, :, :]  # [B, K, H, W]

        target_conf = torch.zeros_like(pred_conf)
        target_boxes = torch.zeros_like(pred_boxes)
        target_label = torch.zeros((B, H, W), dtype=torch.long, device=det_out.device)

        # Construct grid targets from ground truth bounding boxes
        for i in range(B):
            boxes = targets_boxes[i]
            labels = targets_labels[i]
            if len(boxes) > 0:
                for box, label in zip(boxes, labels):
                    # Normalize box [x_min, y_min, x_max, y_max] to [0, 1]
                    x_min, y_min, x_max, y_max = box / self.img_size
                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0

                    grid_x = min(int(center_x * W), W - 1)
                    grid_y = min(int(center_y * H), H - 1)

                    target_conf[i, grid_y, grid_x] = 1.0
                    target_boxes[i, :, grid_y, grid_x] = torch.tensor([x_min, y_min, x_max, y_max], device=det_out.device)
                    target_label[i, grid_y, grid_x] = int(label.item()) - 1  # labels are 1-indexed

        pos_mask = (target_conf > 0.5)

        # Confidence (objectness) Loss
        conf_loss = self.conf_bce(pred_conf, target_conf)

        # Classification Loss — cross-entropy over K classes at positive cells.
        # ponytail: with num_classes == 1 there's nothing to classify (objectness
        # already carries presence), so cls_loss is a no-op zero.
        if self.num_classes > 1 and pos_mask.sum() > 0:
            cls_logits = pred_cls.permute(0, 2, 3, 1)[pos_mask]  # [N_pos, K]
            cls_loss = self.ce(cls_logits, target_label[pos_mask])
        else:
            cls_loss = torch.tensor(0.0, device=det_out.device)

        # BBox Regression Loss
        if pos_mask.sum() > 0:
            reg_loss = self.smooth_l1(pred_boxes.permute(0, 2, 3, 1)[pos_mask], target_boxes.permute(0, 2, 3, 1)[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=det_out.device)

        return reg_loss, conf_loss, cls_loss


class CombinedDualLoss(nn.Module):
    """
    Combined Total Loss for Dual-Task Learning:
    Total_Loss = lambda_seg * Dice_BCE_Loss + lambda_det * BBox_Regression_Loss + lambda_cls * Classification_Loss
    """
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_det: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_conf: float = 1.0,
        num_classes: int = 1,
        img_size: float = 512.0,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_det = lambda_det
        self.lambda_cls = lambda_cls
        self.lambda_conf = lambda_conf

        self.seg_loss_fn = DiceBCELoss()
        self.det_loss_fn = BBoxLoss(num_classes=num_classes, img_size=img_size)

    def forward(self, outputs, target_masks, target_boxes, target_labels):
        seg_logits = outputs["seg_logits"]
        det_out = outputs["det_out"]

        # 1. Segmentation Loss
        seg_loss = self.seg_loss_fn(seg_logits, target_masks)

        # 2. BBox Regression, Objectness & Classification Loss
        bbox_reg_loss, bbox_conf_loss, bbox_cls_loss = self.det_loss_fn(det_out, target_boxes, target_labels)

        # Total Weighted Loss
        total_loss = (
            (self.lambda_seg * seg_loss) +
            (self.lambda_det * bbox_reg_loss) +
            (self.lambda_conf * bbox_conf_loss) +
            (self.lambda_cls * bbox_cls_loss)
        )

        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "bbox_reg_loss": bbox_reg_loss,
            "bbox_conf_loss": bbox_conf_loss,
            "bbox_cls_loss": bbox_cls_loss
        }


if __name__ == "__main__":
    # Self-check: pos_weight must actually change the confidence loss (not a no-op),
    # given a single positive cell out of 49 like real training targets.
    det_out = torch.randn(2, 6, 7, 7)
    boxes = [torch.tensor([[100.0, 100.0, 200.0, 200.0]]), torch.tensor([[50.0, 50.0, 150.0, 150.0]])]
    labels = [torch.tensor([1]), torch.tensor([1])]

    weighted_conf = BBoxLoss(pos_weight=45.0).forward(det_out, boxes, labels)[1]
    unweighted_conf = BBoxLoss(pos_weight=1.0).forward(det_out, boxes, labels)[1]
    assert weighted_conf.item() != unweighted_conf.item(), "pos_weight had no effect on conf_loss"
    print("OK: pos_weight changes conf_loss as expected", weighted_conf.item(), "vs", unweighted_conf.item())

    # Self-check: K=1 has no classification signal (cls_loss is a no-op zero).
    reg1, conf1, cls1 = BBoxLoss(num_classes=1).forward(det_out, boxes, labels)
    assert cls1.item() == 0.0, "K=1 cls_loss should be a no-op zero"
    print("OK: K=1 cls_loss is zero")

    # Self-check: K=3 classification loss must actually respond to the labels —
    # logits favoring the correct class score lower cls_loss than logits favoring a wrong one.
    K = 3
    boxes3 = [torch.tensor([[100.0, 100.0, 200.0, 200.0]]), torch.tensor([[50.0, 50.0, 150.0, 150.0]])]
    labels3 = [torch.tensor([1]), torch.tensor([3])]  # image 0 -> class 1, image 1 -> class 3
    loss_fn3 = BBoxLoss(num_classes=K)

    det_out_correct = torch.zeros(2, 4 + 1 + K, 7, 7)
    # Grid cells for the two boxes above (7x7 grid, 512 img size): centers (150,150)->cell(2,2), (100,100)->cell(1,1)
    det_out_correct[0, 5 + 0, 2, 2] = 10.0  # favors class 1 at image 0's positive cell
    det_out_correct[1, 5 + 2, 1, 1] = 10.0  # favors class 3 at image 1's positive cell

    det_out_wrong = torch.zeros(2, 4 + 1 + K, 7, 7)
    det_out_wrong[0, 5 + 1, 2, 2] = 10.0  # favors the wrong class at image 0
    det_out_wrong[1, 5 + 0, 1, 1] = 10.0  # favors the wrong class at image 1

    cls_correct = loss_fn3.forward(det_out_correct, boxes3, labels3)[2]
    cls_wrong = loss_fn3.forward(det_out_wrong, boxes3, labels3)[2]
    assert cls_correct.item() < cls_wrong.item(), "cls_loss did not favor correct-class logits"
    assert det_out_correct.shape == (2, 8, 7, 7)
    print("OK: K=3 cls_loss favors correct-class logits", cls_correct.item(), "vs", cls_wrong.item())
