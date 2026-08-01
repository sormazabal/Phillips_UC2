import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before `import torch` -- reduces allocator fragmentation, which is a common
# cause of OOM well before the GPU is actually full on long unfrozen-encoder runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
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
    parser.add_argument("--max_epochs", type=int, default=None, help="Override training.max_epochs from config")
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Fraction/count of train batches per epoch (Lightning passthrough; for quick smoke runs)")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Fraction/count of val batches per epoch (Lightning passthrough)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Where to write checkpoints (e.g. a Google Drive path in Colab, so they survive a runtime disconnect)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        # Headroom below 1.0 so the allocator raises a catchable OOM instead of the
        # driver/other processes on the box getting starved -- important on an unattended
        # remote run with no one there to intervene.
        torch.cuda.set_per_process_memory_fraction(0.85)

    data_cfg = config.get("data", {})
    if os.path.isdir("/kaggle/input"):
        # ponytail: hardcoded to this dataset's actual mount path; if the dataset is
        # re-uploaded under a different owner/slug, update this prefix
        for key, value in data_cfg.items():
            if isinstance(value, str) and value.startswith("./CathAction"):
                data_cfg[key] = value.replace(
                    "./CathAction", "/kaggle/input/datasets/sofiazowormazabal/cathaction", 1
                )
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    model_cfg = config.get("model", {})
    det_classes = model_cfg.get("det_classes", ["coronary_stenosis"])
    det_categories = det_classes if len(det_classes) > 1 else None

    # Optional overrides for non-ARCADE COCO sources (e.g. CathAction); a "{split}"
    # placeholder in any path is filled in per train/val/test by ArcadeDataset.
    source_overrides = {
        k: data_cfg[k] for k in ("img_dir", "seg_ann_file", "det_ann_file", "det_img_dir") if k in data_cfg
    }

    # Datasets & DataLoaders
    train_dataset = ArcadeDataset(
        data_dir=data_cfg.get("data_dir", "./Arcade"),
        split="train",
        img_size=tuple(model_cfg.get("img_size", [512, 512])),
        det_ann_subset=data_cfg.get("det_ann_subset", "stenosis"),
        det_categories=det_categories,
        **source_overrides,
    )

    val_dataset = ArcadeDataset(
        data_dir=data_cfg.get("data_dir", "./Arcade"),
        split="val",
        img_size=tuple(model_cfg.get("img_size", [512, 512])),
        det_ann_subset=data_cfg.get("det_ann_subset", "stenosis"),
        det_categories=det_categories,
        **source_overrides,
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
        # A "/" inside a Lightning filename placeholder is not substituted -- it's read as a
        # literal path separator, so the old "{val/dice_score:.4f}" template silently wrote
        # into a nested "val/" subdirectory instead of encoding the metric in the filename.
        dirpath=args.checkpoint_dir,
        monitor="val/dice_score",
        mode="max",
        filename="best-model-{epoch:02d}",
        save_top_k=1,
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor="val/dice_score",
        mode="max",
        patience=10
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs if args.max_epochs is not None else train_cfg.get("max_epochs", 50),
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=4,  # batch_size 2 x 4 = effective batch 8
        gradient_clip_val=1.0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
