
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "arpit180302/reef-watch-checkpoints"

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

CHECKPOINT_FILES = [
    "tf_efficientnetv2_s.in21k_ft_in1k_fold0.pt",
    "convnext_tiny.fb_in22k_ft_in1k_fold0.pt",
]


def ensure_checkpoints():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or None

    paths = {}

    for filename in CHECKPOINT_FILES:
        destination = CHECKPOINT_DIR / filename

        if destination.exists() and destination.stat().st_size > 0:
            paths[filename] = str(destination)
            continue

        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=token,
            local_dir=str(CHECKPOINT_DIR),
        )

        paths[filename] = downloaded_path

    return paths
