"""Sophisticated early stopping with EMA smoothing and warmup protection."""

from math import inf
from typing import Any


class EarlyStopping:
    """
    Multi-metric early stopping with exponential moving average smoothing
    and a warmup guard that prevents stopping before min_epochs.

    Parameters
    ----------
    patience : int
        Consecutive evaluations without improvement before stopping.
    min_delta : float
        Minimum absolute change in the smoothed metric to count as improvement.
    min_epochs : int
        Warmup period — patience counter is frozen until epoch >= min_epochs.
    ema_alpha : float
        EMA smoothing factor. 0 = no smoothing (use raw value), 1 = no memory.
    monitor : str
        "f1" (maximize) or "loss" (minimize).

    Usage
    -----
        stopper = EarlyStopping(patience=15, min_epochs=20, monitor="f1")
        for epoch, val_f1 in training_loop:
            if stopper.step(val_f1, epoch=epoch):
                break
            if stopper.improved:
                save_best_checkpoint()
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        min_epochs: int = 10,
        ema_alpha: float = 0.3,
        monitor: str = "f1",
    ) -> None:
        if monitor not in ("f1", "loss"):
            raise ValueError(f"monitor must be 'f1' or 'loss', got {monitor!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.ema_alpha = ema_alpha
        self.monitor = monitor

        self.best: float = -inf if monitor == "f1" else inf
        self.smoothed: float | None = None
        self.counter: int = 0
        self.improved: bool = False
        self.best_epoch: int = 0

    def step(self, val_metric: float, epoch: int = 0) -> bool:
        """
        Advance one evaluation step.

        Returns True if training should stop.
        Sets self.improved=True whenever a new best is found.
        """
        if self.smoothed is None:
            self.smoothed = val_metric
        else:
            self.smoothed = self.ema_alpha * val_metric + (1.0 - self.ema_alpha) * self.smoothed

        if epoch < self.min_epochs:
            self.improved = False
            return False

        if self.monitor == "f1":
            is_better = self.smoothed > self.best + self.min_delta
        else:
            is_better = self.smoothed < self.best - self.min_delta

        if is_better:
            self.best = self.smoothed
            self.counter = 0
            self.improved = True
            self.best_epoch = epoch
        else:
            self.counter += 1
            self.improved = False

        return self.counter >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "smoothed": self.smoothed,
            "counter": self.counter,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = state["best"]
        self.smoothed = state.get("smoothed")
        self.counter = state["counter"]
        self.best_epoch = state.get("best_epoch", 0)
