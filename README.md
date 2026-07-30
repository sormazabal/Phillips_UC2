# ARCADE XCA Dual-Task Medical Imaging Pipeline

PyTorch & Hugging Face machine learning repository for processing X-ray Coronary Angiography (XCA) images from the ARCADE dataset.

This pipeline performs a dual medical imaging task simultaneously:
1. **Semantic Segmentation:** Complete coronary vessel tree segmentation.
2. **Object Detection & Localization:** Multi-class detection of coronary stenosis and, as the class list grows, other cardiac interventional devices (catheters, guidewires, stents, ...) for interventional balloon/stent target planning.

## Summary:

A fine-tuned SegFormer backbone jointly segments the full coronary vessel tree and detects/localizes/classifies an arbitrary number of object classes with severity scoring, from one X-ray angiography frame. The detection head, loss, dataset loader, inference decoder and viewer are all class-count-agnostic — adding a device class is a `config.yaml` change (`model.det_classes`), not a code change, as long as you have COCO-format boxes for it.

**Scope note:** this repo currently trains only on ARCADE (vessel segments + stenosis). It does not yet detect real interventional devices (stents, balloons, catheters, guidewires, ...) — that requires a device-annotated dataset. See [Extending to device detection](#-extending-to-device-detection) below for what's available and how to plug one in.

## Client pain points:

## Potential Client: 
---

## 🏗️ Model Architecture

Rather than training a model from scratch, this repository leverages pre-trained Vision Transformers from Hugging Face Hub (e.g., `nvidia/mit-b3` SegFormer) to accelerate convergence on smaller medical datasets.

- **Encoder (Backbone):** Hugging Face pre-trained `SegformerModel` / Vision Transformer (`nvidia/mit-b5`).
- **Head 1 (Segmentation):** SegFormer MLP decoder fusing multi-scale feature maps (`c1`, `c2`, `c3`, `c4`) to output binary vessel tree masks.
- **Head 2 (Detection/Localization/Classification):** Anchor-free detection head on a 7×7 grid predicting, per cell, bounding boxes `[x_min, y_min, x_max, y_max]`, an objectness confidence score, and a class logit per entry in `model.det_classes`. One object per grid cell. Severity is a post-hoc confidence/area heuristic applied only to the `coronary_stenosis` class — it is not a learned label.

---

## 📂 Repository Structure

```text
.
├── .gitignore
├── config.yaml
├── requirements.txt
├── README.md
├── scripts/
│   ├── train.py          # PyTorch Lightning training script with WandB logging
│   ├── inference.py      # Inference script formatted to target JSON schema
│   └── evaluate.py       # Dataset-wide metrics (Dice, detection P/R/F1); zero-shot or from a checkpoint
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py    # PyTorch Dataset parser for ARCADE syntax & stenosis annotations
    ├── models/
    │   ├── __init__.py
    │   ├── hf_dual_net.py        # Dual-head network with HF SegFormer/ViT backbone
    │   └── lightning_module.py   # PyTorch Lightning wrapper
    └── utils/
        ├── __init__.py
        └── losses.py     # Combined Dice-BCE segmentation and BBox regression loss
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.9+
- PyTorch & CUDA support (optional for GPU acceleration)

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Kaggle API Setup (Windows / PowerShell)

Generate an API token at [kaggle.com/settings/api](https://www.kaggle.com/settings/api) ("Generate New Token"), then set it for your PowerShell session:

```powershell
$env:KAGGLE_API_TOKEN = "your_token_here"
```

This only lasts for the current terminal session. To persist it across sessions/reboots:

```powershell
[System.Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "your_token_here", "User")
```

(open a new terminal for this to take effect)

Verify it works with:

```powershell
kaggle datasets list
```

**Never commit your token.** Don't put it in `.env`, `.gitignore`d or not — treat any token that's touched a repo folder as compromised and regenerate it.

---

## ⚙️ Configuration (`config.yaml`)

All training parameters, dataset directories, loss weights, and model choices are centralized in `config.yaml`:

```yaml
model:
  backbone: "nvidia/mit-b5"
  freeze_backbone: true
  num_seg_classes: 1
  det_classes: ["coronary_stenosis"]   # num_det_classes = len(det_classes)
  img_size: [512, 512]

data:
  data_dir: "./Arcade"
  det_ann_subset: "stenosis"   # which ARCADE subset ("stenosis" or "syntax") supplies detection boxes
  batch_size: 8
  num_workers: 4

training:
  max_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  lambda_seg: 1.0
  lambda_det: 1.0
  lambda_conf: 1.0
  lambda_cls: 1.0

wandb:
  project: "arcade-xca-dual-task"
  entity: null
```

`model.det_classes` is the single source of truth for the number of detection classes — the model, loss, and inference decoder all derive `num_det_classes`/channel layout from `len(det_classes)`. `lambda_conf` weights the objectness (presence) loss separately from `lambda_cls`, which weights the classification loss and is a no-op when there's only one class.

To point detection at a different COCO annotation source (e.g. the ARCADE `syntax` subset's 25 SYNTAX segment categories, or an external device dataset), set `data.det_ann_subset` and list matching category names in `model.det_classes`; `ArcadeDataset` also accepts explicit `img_dir` / `seg_ann_file` / `det_ann_file` overrides for non-ARCADE layouts.

---

## 🏋️ Training

To launch the training pipeline with PyTorch Lightning and Weights & Biases experiment tracking:

```bash
python scripts/train.py --config config.yaml
```

The trainer will automatically:
- Log training & validation metrics (Dice score, Total loss, Bounding Box loss) to WandB.
- Save top-performing checkpoints based on validation Dice score.
- Apply early stopping after 10 non-improving epochs.

---

## 🔮 Inference & JSON Output Schema

Run inference on any XCA image:

```bash
python scripts/inference.py --image_path Arcade\stenosis\test\images\1.png --checkpoint path/to/best_checkpoint.ckpt --output_json output.json
```

### Strict Output JSON Schema

The inference script outputs structured predictions:

```json
{
  "frame_id": "image_name",
  "timestamp": "2026-07-28T12:00:00+00:00",
  "devices": [
    {
      "device_class": "coronary_stenosis",
      "instance_id": "target_01",
      "bounding_box": [120.5, 230.1, 165.2, 280.4],
      "severity": "high",
      "detection_confidence": 0.94
    }
  ]
}
```

`device_class` is decoded from `model.det_classes` (argmax over the class logits), not a hard-coded string. `severity` is only emitted for the `coronary_stenosis` class; other classes get `"device_state": null` instead, matching the shape of the target schema in the [Philips device-detection brief](#-extending-to-device-detection) (which also expects `landmarks`, `device_state`, and `tracking_confidence` — none of which this pipeline produces yet).

---

## 🧪 Zero-Shot Evaluation & Metrics

To measure how the pre-trained backbone performs **without any fine-tuning** (frozen `nvidia/mit-b3`/`mit-b5` backbone + randomly initialized task heads), run `evaluate.py` on the test split with no `--checkpoint`:

```bash
python scripts/evaluate.py --config config.yaml --split test
```

Once you have a trained checkpoint, pass it in to get post-training metrics on the same split:

```bash
python scripts/evaluate.py --config config.yaml --split test --checkpoint path/to/best_checkpoint.ckpt
```
python scripts/evaluate.py --config config.yaml --split test --checkpoint checkpoints\last_46.ckpt

This reports, over the whole split:

- **`mean_dice_score`** — average segmentation Dice score across all test images.
- **`detection_precision` / `detection_recall` / `detection_f1`** — stenosis detection quality, matching each ground-truth box to the model's prediction in the same grid cell (IoU ≥ 0.3, confidence ≥ `--conf_thresh`).

```json
{
  "split": "test",
  "num_images": 300,
  "mean_dice_score": 0.052,
  "detection_precision": 0.0,
  "detection_recall": 0.0,
  "detection_f1": 0.0,
  "conf_threshold": 0.5
}
```
Zero-shot example
```
  "split": "test",
  "num_images": 300,
  "mean_dice_score": 0.060636836380387346,
  "detection_precision": 0.0,
  "detection_recall": 0.0,
  "detection_f1": 0.0,
  "conf_threshold": 0.5
  ``` 


Zero-shot numbers will be low (the task heads are untrained) — this is a baseline to compare against once the model has been fine-tuned. Add `--output_json results.json` to save the report.

---

## 🖥️ Interactive Viewer

A Streamlit app for loading a checkpoint, running inference, and visualizing results in the browser:
- **Segmentation:** pixel-level vessel mask overlay + boundary contours.
- **Localization:** stenosis bounding boxes, center landmarks, severity/confidence labels.

```bash
streamlit run app.py
```

Upload an XCA image (or point it at a path under `Arcade/`), pick a checkpoint from `checkpoints/`
(or run zero-shot), and adjust the detection/mask thresholds from the sidebar.

---

## 📊 Target Metrics

- **Vessel Segmentation:** Dice Score $\ge 0.90$
- **Stenosis Localization:** Mean Average Precision (mAP) $> 0.80$
