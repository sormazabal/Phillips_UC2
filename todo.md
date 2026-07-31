# CathAction training fix — GPU machine steps

## 1. Smoke test (run first, ~a couple minutes)

```bash
WANDB_MODE=offline python scripts/train.py --config config_cathaction.yaml --max_epochs 1 --limit_train_batches 20 --limit_val_batches 5
```

Drop `WANDB_MODE=offline` if `wandb login` is already set up on that machine.

Check the output for:

- No OOM / crash.
- Model summary table shows **`Trainable params` ≈ 24.0 M** (not 2.0 M) — confirms the backbone unfreeze took effect.

## 2. Full run (only after the smoke test passes)

```bash
WANDB_MODE=offline python scripts/train.py --config config_cathaction.yaml
```

Baseline to beat: **val dice 0.4214** (peak-threshold dice measured on the old frozen-backbone checkpoint).

## 3. Final evaluation

```bash
python scripts/evaluate.py --config config_cathaction.yaml --checkpoint <best_ckpt_path> --split val --seg_thresh 0.2 --output_json results_val.json
```
