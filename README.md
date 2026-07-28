# ARCADE XCA Dual-Task Medical Imaging Pipeline

Production-ready PyTorch & Hugging Face machine learning repository for processing X-ray Coronary Angiography (XCA) images from the ARCADE dataset.

This pipeline performs a dual medical imaging task simultaneously:
1. **Semantic Segmentation:** Complete coronary vessel tree segmentation.
2. **Object Detection & Localization:** Localizing and classifying coronary stenosis (narrowed arteries) for interventional balloon/stent target planning.

---

## 🏗️ Model Architecture

Rather than training a model from scratch, this repository leverages pre-trained Vision Transformers from Hugging Face Hub (e.g., `nvidia/mit-b3` SegFormer) to accelerate convergence on smaller medical datasets.

- **Encoder (Backbone):** Hugging Face pre-trained `SegformerModel` / Vision Transformer (`nvidia/mit-b3`).
- **Head 1 (Segmentation):** SegFormer MLP decoder fusing multi-scale feature maps (`c1`, `c2`, `c3`, `c4`) to output binary vessel tree masks.
- **Head 2 (Detection/Localization):** Anchor-free detection head predicting bounding boxes `[x_min, y_min, x_max, y_max]`, detection confidence scores, and severity classifications.

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
│   └── inference.py      # Inference script formatted to target JSON schema
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

---

## ⚙️ Configuration (`config.yaml`)

All training parameters, dataset directories, loss weights, and model choices are centralized in `config.yaml`:

```yaml
model:
  backbone: "nvidia/mit-b3"
  freeze_backbone: true
  num_seg_classes: 1
  num_det_classes: 1
  img_size: [512, 512]

data:
  data_dir: "./Arcade"
  batch_size: 8
  num_workers: 4

training:
  max_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  lambda_seg: 1.0
  lambda_det: 1.0
  lambda_cls: 1.0

wandb:
  project: "arcade-xca-dual-task"
  entity: null
```

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
python scripts/inference.py --image_path path/to/xca_image.png --checkpoint path/to/best_checkpoint.ckpt --output_json output.json
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

---

## 📊 Target Metrics

- **Vessel Segmentation:** Dice Score $\ge 0.90$
- **Stenosis Localization:** Mean Average Precision (mAP) $> 0.80$
