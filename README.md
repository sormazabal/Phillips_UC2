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
    │   └── dataset.py    # PyTorch Dataset parser for ARCADE (or any COCO) syntax/detection annotations, multi-class
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
- **`detection_precision` / `detection_recall` / `detection_f1`** — macro-averaged across `model.det_classes`. Matches each ground-truth box to the model's prediction in the same grid cell, requiring both IoU ≥ 0.3 and a correct predicted class (confidence ≥ `--conf_thresh`).
- **`detection_per_class`** — the same precision/recall/F1/tp/fp/fn broken out per class name.

```json
{
  "split": "test",
  "num_images": 300,
  "mean_dice_score": 0.6885528723398845,
  "detection_precision": 0.03225806451612903,
  "detection_recall": 0.0025906735751295338,
  "detection_f1": 0.004796163069544365,
  "detection_per_class": {
    "coronary_stenosis": {"precision": 0.032, "recall": 0.0026, "f1": 0.0048, "tp": 1, "fp": 30, "fn": 385}
  },
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


Detection F1 is near zero because of the syntax/stenosis image-id mismatch described in [Extending to device detection](#-extending-to-device-detection) — every training sample pairs a `syntax` image with an unrelated frame's `stenosis` boxes. Segmentation (Dice) is trained on correctly-aligned data and reaches ~0.69 on this checkpoint. Zero-shot numbers will be low (the task heads are untrained) — this is a baseline to compare against once the model has been fine-tuned. Add `--output_json results.json` to save the report.

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

Note: `evaluate.py`'s detection metric is a same-grid-cell match, not true mAP (no NMS, no IoU-threshold sweep) — treat it as a proxy, not a claim against the mAP target above.

---

## 🩺 Extending to device detection

A Philips brief ("AI-Based Detection and Tracking of Cardiac Interventional Devices") asks for detection, classification, localization, segmentation and temporal tracking of cardiac interventional devices: coronary stents, balloon catheters, guide catheters, guidewires, atherectomy devices, ablation/mapping catheters, pacemaker leads, and prosthetic/transcatheter valves — with per-device landmarks (tips, markers), a device-state label (inflated/deployed/crimped, etc.), and cross-frame tracking IDs.

This pipeline's detection head, loss, dataset loader and inference decoder are multi-class-native (see Configuration above), so adding a device class is a config change given COCO-format boxes for it. **No single public dataset covers the brief's full device list.** What exists:

| Dataset | Device classes | Annotations | License | Notes |
|---|---|---|---|---|
| [CathAction](https://huggingface.co/datasets/airvlab/CathAction) ([arXiv 2408.13126](https://arxiv.org/abs/2408.13126)) | catheter, guidewire | ~25k segmentation masks (derive boxes via connected components), collision boxes, action labels | CC-BY-NC-SA-4.0 | Best available multi-device set with masks. Split into 4 independent zips — `segmentation_human_train.zip` alone is 0.14 GB and enough for a PoC; skip the 41.8 GB action zip and 4.35 GB collision zip (wrong label space). |
| [Guide3D](https://airvlab.github.io/guide3d/) ([arXiv 2410.22224](https://arxiv.org/html/2410.22224v1)) | guidewire (2 types) | curve/segmentation, bi-planar | non-commercial | Useful for guidewire tip landmarks later. |
| [AngioCAD](https://www.sciencedirect.com/science/article/abs/pii/S0169260726001331) | none (stenosis) | per-artery lesion labels, temporal video | public | Best starting point for the tracking half of the brief — it preserves frame sequences. |
| ARCADE (this repo) | none (25 SYNTAX segments + stenosis) | COCO boxes/polygons | CC-BY | Used to prove the multi-class refactor (see below); no real devices. |

**Licensing:** CathAction/Guide3D are CC-BY-NC-SA — fine for research/PoC, not for a shipped Philips product. Production would need Philips' own annotated DICOM studies.

**Out of scope in this repo:** temporal tracking, tip/marker keypoints, device-state classification, uncertainty estimation, DICOM sequence ingestion, and real-time latency work — none of the datasets above provide the frame-sequence or state annotations needed, so these remain unimplemented pending a Philips-provided or purpose-built dataset.

**Trying the multi-class refactor on another ARCADE subset:** point detection at ARCADE's `syntax` annotations (25 real SYNTAX segment categories) instead of `stenosis`:

```yaml
model:
  det_classes: ["1","2","3","4","5","6","7","8","9","9a","10","10a","11","12","12a","13","14","14a","15","16","16a","16b","16c","12b","14b"]
data:
  det_ann_subset: "syntax"
```

**Known data issue — fixed:** `Arcade/syntax/<split>/images/N.png` and `Arcade/stenosis/<split>/images/N.png` are *different frames* that share the same `file_name`s and image ids 1–1000. `ArcadeDataset` used to index images from the `syntax` file and join detection boxes from `stenosis` onto those same ids, so every training sample paired a syntax image with an unrelated frame's stenosis boxes — the cause of the near-zero detection metrics above. `ArcadeDataset` now builds two independent, self-consistent sample sets — segmentation samples from `syntax` (own images + own polygons) and detection samples from `det_ann_subset` (own images + own boxes) — concatenated into one dataset, each sample tagged `has_seg`/`has_det` so the loss only trains the task it actually has ground truth for. No image is ever paired with another source's labels. Existing checkpoints (e.g. `last_46.ckpt`) were trained before this fix — their detection head learned from mismatched pairs and needs retraining; the segmentation head/backbone remain valid.
