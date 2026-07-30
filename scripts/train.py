import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.dataset import ArcadeDataset, collate_fn
from src.models.lightning_module import ArcadeLightningModule


def main():
    parser = argparse.ArgumentParser(description="Train Arcade XCA Dual-Task Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to .ckpt to resume training from")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    model_cfg = config.get("model", {})
    det_classes = model_cfg.get("det_classes", ["coronary_stenosis"])
    det_categories = det_classes if len(det_classes) > 1 else None

    # Datasets & DataLoaders
    train_dataset = ArcadeDataset(
        data_dir=data_cfg.get("data_dir", "./Arcade"),
        split="train",
        img_size=tuple(model_cfg.get("img_size", [512, 512])),
        det_ann_subset=data_cfg.get("det_ann_subset", "stenosis"),
        det_categories=det_categories,
    )

    val_dataset = ArcadeDataset(
        data_dir=data_cfg.get("data_dir", "./Arcade"),
        split="val",
        img_size=tuple(model_cfg.get("img_size", [512, 512])),
        det_ann_subset=data_cfg.get("det_ann_subset", "stenosis"),
        det_categories=det_categories,
    )

    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn
    )

    # Lightning Module
    model = ArcadeLightningModule(config=config)

    # Logger & Callbacks
    logger = WandbLogger(
        project=wandb_cfg.get("project", "arcade-xca-dual-task"),
        entity=wandb_cfg.get("entity", None),
        id=wandb_cfg.get("run_id", None),
        resume="allow"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/dice_score",
        mode="max",
        filename="best-arcade-model-{epoch:02d}-{val/dice_score:.4f}",
        save_top_k=1
    )

    early_stopping = EarlyStopping(
        monitor="val/dice_score",
        mode="max",
        patience=10
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 50),
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices=1
    )

    # Train model
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
