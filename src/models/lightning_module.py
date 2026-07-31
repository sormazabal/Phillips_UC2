import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models.hf_dual_net import HFDualNet
from src.utils.losses import CombinedDualLoss


class ArcadeLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Arcade XCA Dual-Task (Vessel Segmentation + Stenosis Detection).
    Handles training, validation steps, optimization, and metric logging.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})

        det_classes = model_cfg.get("det_classes", ["coronary_stenosis"])
        img_size = model_cfg.get("img_size", [512, 512])

        # Dual-Head Model
        self.model = HFDualNet(
            backbone_name=model_cfg.get("backbone", "nvidia/mit-b3"),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            num_seg_classes=model_cfg.get("num_seg_classes", 1),
            num_det_classes=len(det_classes),
            backbone_kwargs=model_cfg.get("backbone_kwargs")
        )

        # Loss Function
        self.loss_fn = CombinedDualLoss(
            lambda_seg=train_cfg.get("lambda_seg", 1.0),
            lambda_det=train_cfg.get("lambda_det", 1.0),
            lambda_cls=train_cfg.get("lambda_cls", 1.0),
            lambda_conf=train_cfg.get("lambda_conf", 1.0),
            num_classes=len(det_classes),
            img_size=float(img_size[1]),
            seg_pos_weight=train_cfg.get("seg_pos_weight"),
        )

    def forward(self, x):
        return self.model(x)

    def _compute_dice(self, logits, targets, valid_mask=None, smooth=1e-6):
        if valid_mask is not None:
            if valid_mask.sum() == 0:
                return None
            logits = logits[valid_mask]
            targets = targets[valid_mask]
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()

    @staticmethod
    def _is_oom(exc: RuntimeError) -> bool:
        return "out of memory" in str(exc).lower()

    def training_step(self, batch, batch_idx):
        try:
            images = batch["images"]
            masks = batch["masks"]
            boxes = batch["boxes"]
            labels = batch["labels"]
            has_seg = batch["has_seg"]
            has_det = batch["has_det"]

            outputs = self(images)
            losses = self.loss_fn(outputs, masks, boxes, labels, has_seg=has_seg, has_det=has_det)

            total_loss = losses["total_loss"]
            seg_dice = self._compute_dice(outputs["seg_logits"], masks, valid_mask=has_seg)

            # Logging metrics
            self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/seg_loss", losses["seg_loss"], on_step=True, on_epoch=True)
            self.log("train/bbox_reg_loss", losses["bbox_reg_loss"], on_step=True, on_epoch=True)
            self.log("train/bbox_conf_loss", losses["bbox_conf_loss"], on_step=True, on_epoch=True)
            self.log("train/bbox_cls_loss", losses["bbox_cls_loss"], on_step=True, on_epoch=True)
            if seg_dice is not None:
                self.log("train/dice_score", seg_dice, on_step=True, on_epoch=True, prog_bar=True)

            return total_loss
        except RuntimeError as e:
            if not self._is_oom(e):
                raise
            # ponytail: single-batch OOM on an unattended run must not kill the whole
            # job -- drop this batch, free the fragment, keep training. Recurring OOMs
            # still show up as repeated warnings in the log for follow-up.
            torch.cuda.empty_cache()
            self.print(f"[OOM] skipped train batch {batch_idx}: {e}")
            return None

    def validation_step(self, batch, batch_idx):
        try:
            images = batch["images"]
            masks = batch["masks"]
            boxes = batch["boxes"]
            labels = batch["labels"]
            has_seg = batch["has_seg"]
            has_det = batch["has_det"]

            outputs = self(images)
            losses = self.loss_fn(outputs, masks, boxes, labels, has_seg=has_seg, has_det=has_det)

            total_loss = losses["total_loss"]
            seg_dice = self._compute_dice(outputs["seg_logits"], masks, valid_mask=has_seg)

            # Logging metrics
            self.log("val/total_loss", total_loss, on_epoch=True, prog_bar=True)
            self.log("val/seg_loss", losses["seg_loss"], on_epoch=True)
            self.log("val/bbox_reg_loss", losses["bbox_reg_loss"], on_epoch=True)
            self.log("val/bbox_conf_loss", losses["bbox_conf_loss"], on_epoch=True)
            self.log("val/bbox_cls_loss", losses["bbox_cls_loss"], on_epoch=True)
            if seg_dice is not None:
                self.log("val/dice_score", seg_dice, on_epoch=True, prog_bar=True)

            return total_loss
        except RuntimeError as e:
            if not self._is_oom(e):
                raise
            torch.cuda.empty_cache()
            self.print(f"[OOM] skipped val batch {batch_idx}: {e}")
            return None

    def configure_optimizers(self):
        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("learning_rate", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-2))
        # self.trainer.max_epochs reflects a CLI --max_epochs override; the config value
        # is only a fallback for when this module is built without a Trainer attached.
        max_epochs = self.trainer.max_epochs if self.trainer is not None else train_cfg.get("max_epochs", 50)

        # Fine-tuning optimizer: different learning rates for backbone vs heads if unfrozen
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


if __name__ == "__main__":
    # Self-check: the OOM-skip guard must only catch actual OOM errors and let
    # every other RuntimeError propagate -- a broad except here would hide real bugs.
    assert ArcadeLightningModule._is_oom(RuntimeError("CUDA out of memory. Tried to allocate..."))
    assert ArcadeLightningModule._is_oom(RuntimeError("CUDA error: out of memory"))
    assert not ArcadeLightningModule._is_oom(RuntimeError("size mismatch"))
    assert not ArcadeLightningModule._is_oom(RuntimeError("index out of bounds"))
    print("OK: _is_oom distinguishes OOM RuntimeErrors from other RuntimeErrors")
