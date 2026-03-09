import json
import os
import logging
from transformers import (
    TrainerCallback,
    TrainerState,
)


class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = os.path.join(log_file_path, "log.jsonl")

    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        if logs is None:
            return
        _ = logs.pop("total_flos", None)
        if state.is_world_process_zero:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                **logs,
            }

            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")


def get_logger(filename, level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    fh = logging.FileHandler(str(filename))
    fh.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
