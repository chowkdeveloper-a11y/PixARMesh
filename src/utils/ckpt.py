import os
from transformers.trainer_utils import get_last_checkpoint as hf_get_last_checkpoint


def get_last_checkpoint(output_dir: str) -> str | None:
    if not os.path.exists(output_dir):
        return
    last_ckpt = hf_get_last_checkpoint(output_dir)
    return last_ckpt
