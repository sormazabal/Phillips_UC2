import glob, os, shutil, subprocess, sys

#DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/Arcade/checkpoints'  # ponytail: adjust if checkpoints live elsewhere on Drive
DRIVE_CHECKPOINT_DIR = 'checkpoints' 
os.makedirs('checkpoints', exist_ok=True)

ckpts = sorted(glob.glob(f'{DRIVE_CHECKPOINT_DIR}/*.ckpt'), key=os.path.getmtime)
if not ckpts:
    raise FileNotFoundError(f"No .ckpt files found in {DRIVE_CHECKPOINT_DIR}")

latest_ckpt = ckpts[-1]
local_ckpt = os.path.join('checkpoints', os.path.basename(latest_ckpt))
if os.path.abspath(latest_ckpt) != os.path.abspath(local_ckpt):
    shutil.copy(latest_ckpt, local_ckpt)
print(f"Loaded latest checkpoint: {latest_ckpt} -> {local_ckpt}")

# Continue training from local_ckpt, logging to same wandb run
# (set wandb.run_id in config.yaml to the run at
#  https://wandb.ai/idssp/arcade-xca-dual-task?nw=nwusersoa2100 so it resumes the same run instead of starting a new one)
config_path = os.environ.get("TRAIN_CONFIG", "config.yaml")
subprocess.run(
    [sys.executable, "scripts/train.py", "--config", config_path, "--resume", local_ckpt],
    check=True,
)