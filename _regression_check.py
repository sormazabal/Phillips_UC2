"""Regression check: same checkpoint, same single-class config, compare
metrics to the README baseline (mean_dice ~0.052, detection F1 0.0).
Works around a pre-existing (pre-Phase-1) strict-load buffer mismatch by
loading with strict=False. Throwaway, delete after running."""
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ArcadeDataset, collate_fn
from src.models.lightning_module import ArcadeLightningModule
from scripts.evaluate import dice_score, match_detections, _prf1

with open("config.yaml") as f:
    config = yaml.safe_load(f)

lm = ArcadeLightningModule(config=config)
ckpt = torch.load("checkpoints/last_46.ckpt", map_location="cpu", weights_only=False)
missing, unexpected = lm.load_state_dict(ckpt["state_dict"], strict=False)
print("missing:", missing, "unexpected:", unexpected)
model = lm.model
model.eval()

det_classes = config["model"]["det_classes"]
img_size = tuple(config["model"]["img_size"])
dataset = ArcadeDataset(
    data_dir=config["data"]["data_dir"], split="test", img_size=img_size,
    det_ann_subset=config["data"].get("det_ann_subset", "stenosis"),
)
loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

dice_scores = []
totals = {1: [0, 0, 0]}
with torch.no_grad():
    for batch in loader:
        outputs = model(batch["images"])
        dice_scores.extend(dice_score(outputs["seg_logits"], batch["masks"]))
        stats = match_detections(outputs["det_out"], batch["boxes"], batch["labels"], img_size, 0.5, len(det_classes))
        for label, (tp, fp, fn) in stats.items():
            totals[label][0] += tp
            totals[label][1] += fp
            totals[label][2] += fn

p, r, f1 = _prf1(*totals[1])
print("num_images", len(dataset))
print("mean_dice_score", sum(dice_scores) / len(dice_scores))
print("detection_precision", p, "recall", r, "f1", f1)
