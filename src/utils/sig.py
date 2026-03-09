import signal
import threading

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

_STOP_REQUESTED = threading.Event()


def _sigusr1_handler(signum, frame):
    _STOP_REQUESTED.set()


def install_sigusr1_handler():
    signal.signal(signal.SIGUSR1, _sigusr1_handler)


class SaveAndStopOnSignalCallback(TrainerCallback):
    def __init__(self):
        self.signal_received = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        return self._maybe_save_and_stop(args, state, control, **kwargs)

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        return self._maybe_save_and_stop(args, state, control, **kwargs)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        return self._maybe_save_and_stop(args, state, control, **kwargs)

    def _maybe_save_and_stop(self, args, state, control, **kwargs):
        if _STOP_REQUESTED.is_set() and not self.signal_received:
            self.signal_received = True
            control.should_save = True
            control.should_training_stop = True
            control.should_epoch_stop = True
            control.should_evaluate = False

        return control
