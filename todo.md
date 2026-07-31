Smoke-test on GPU machine to confirm ~24M trainable params and VRAM fit
Self-check passes. Four layers of OOM defense now in config_cathaction.yaml, train.py, hf_dual_net.py, and lightning_module.py:

Smaller footprint by construction: batch_size: 2 with accumulate_grad_batches=4 in the trainer (same effective batch of 8, half the peak activation memory per step).
Gradient checkpointing on the encoder, enabled automatically whenever the backbone is unfrozen (hf_dual_net.py) — this is the single biggest lever for fitting an unfrozen transformer encoder on 8GB.

Allocator tuning: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set before import torch, plus set_per_process_memory_fraction(0.85) (was 0.9) so the allocator fails a step before it starves the rest of the GPU/driver on that machine.
Non-fatal recovery: training_step/validation_step now catch CUDA-OOM specifically (verified by the self-check to not swallow other RuntimeErrors), call empty_cache(), print a [OOM] skipped batch N line, and return None so Lightning skips just that batch instead of crashing the whole run.
None of this is a 100% guarantee — if the box also runs other GPU jobs, memory_fraction won't protect you — but a single-batch OOM will no longer end the overnight run, and it's now much less likely to occur in the first place. Whenever you get to the other machine, run the smoke test from before; the only thing left to verify is that it doesn't OOM and shows Trainable params ≈ 24M.

