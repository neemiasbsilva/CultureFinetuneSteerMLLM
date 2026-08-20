"""Pin the training-halt contract of the EMA early stopper.

The stopper decides when a run ends and which epoch gets labelled "best", so its
mistakes never surface as errors — they surface as a checkpoint trained for the
wrong number of epochs.  A flipped comparison would turn the loss monitor into a
maximiser, an off-by-one in the patience counter would cut every run short or
never end one, and a warmup guard that leaks would let a single noisy first
evaluation kill a run before the model has seen the data.  Resume is equally
silent: a state_dict that drops the counter restarts patience from zero.
"""

from __future__ import annotations

from itertools import pairwise
from math import inf
from typing import NamedTuple

import pytest

from src.training.early_stopping import EarlyStopping


class _StepTrace(NamedTuple):
    stop: bool
    improved: bool
    counter: int
    best: float
    best_epoch: int


def _trace(stopper: EarlyStopping, schedule: list[tuple[int, float]]) -> list[_StepTrace]:
    history: list[_StepTrace] = []
    for epoch, value in schedule:
        stop = stopper.step(value, epoch=epoch)
        history.append(
            _StepTrace(stop, stopper.improved, stopper.counter, stopper.best, stopper.best_epoch)
        )
    return history


@pytest.mark.parametrize("monitor", ["accuracy", "kappa", "F1", "", "loss "])
def test_unknown_monitor_is_rejected_at_construction(monitor: str) -> None:
    with pytest.raises(ValueError, match="monitor must be 'f1' or 'loss'"):
        EarlyStopping(monitor=monitor)


@pytest.mark.parametrize(
    ("monitor", "initial_best"),
    [("f1", -inf), ("loss", inf)],
)
def test_initial_best_is_the_worst_possible_value_for_the_monitor(
    monitor: str, initial_best: float
) -> None:
    stopper = EarlyStopping(monitor=monitor)
    assert stopper.best == initial_best
    assert stopper.smoothed is None
    assert stopper.counter == 0
    assert stopper.improved is False
    assert stopper.best_epoch == 0


@pytest.mark.parametrize(
    ("monitor", "worsening_metrics", "initial_best"),
    [
        ("f1", [0.9, 0.7, 0.5, 0.3, 0.1], -inf),
        ("loss", [0.1, 0.3, 0.5, 0.7, 0.9], inf),
    ],
)
def test_warmup_never_stops_and_never_flags_improvement_below_min_epochs(
    monitor: str, worsening_metrics: list[float], initial_best: float
) -> None:
    stopper = EarlyStopping(patience=1, min_delta=0.0, min_epochs=5, ema_alpha=1.0, monitor=monitor)
    history = _trace(stopper, list(enumerate(worsening_metrics)))

    assert [entry.stop for entry in history] == [False] * 5
    assert [entry.improved for entry in history] == [False] * 5
    assert [entry.counter for entry in history] == [0] * 5
    assert [entry.best for entry in history] == [initial_best] * 5
    assert [entry.best_epoch for entry in history] == [0] * 5


@pytest.mark.parametrize("monitor", ["f1", "loss"])
def test_first_step_after_warmup_always_records_a_best(monitor: str) -> None:
    stopper = EarlyStopping(patience=1, min_delta=0.0, min_epochs=3, ema_alpha=1.0, monitor=monitor)
    for epoch in range(3):
        assert stopper.step(0.42, epoch=epoch) is False

    assert stopper.step(0.42, epoch=3) is False
    assert stopper.improved is True
    assert stopper.best == pytest.approx(0.42)
    assert stopper.best_epoch == 3
    assert stopper.counter == 0


def test_smoothing_still_accumulates_during_warmup() -> None:
    stopper = EarlyStopping(min_epochs=10, ema_alpha=0.5, monitor="f1")
    stopper.step(1.0, epoch=0)
    assert stopper.smoothed == pytest.approx(1.0)
    stopper.step(0.0, epoch=1)
    assert stopper.smoothed == pytest.approx(0.5)
    assert stopper.best == -inf


@pytest.mark.parametrize("patience", [1, 2, 3, 5])
def test_stop_fires_exactly_on_the_patience_th_consecutive_non_improving_step(
    patience: int,
) -> None:
    stopper = EarlyStopping(
        patience=patience, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="f1"
    )
    assert stopper.step(0.8, epoch=0) is False

    stops = [stopper.step(0.1, epoch=epoch) for epoch in range(1, patience + 1)]
    assert stops == [False] * (patience - 1) + [True]
    assert stopper.counter == patience


def test_an_improvement_resets_the_patience_counter_before_the_stop_fires() -> None:
    stopper = EarlyStopping(patience=3, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="f1")
    schedule = [(0, 0.50), (1, 0.10), (2, 0.10), (3, 0.90), (4, 0.10), (5, 0.10)]
    history = _trace(stopper, schedule)

    assert [entry.counter for entry in history] == [0, 1, 2, 0, 1, 2]
    assert not any(entry.stop for entry in history)
    assert stopper.best == pytest.approx(0.90)
    assert stopper.best_epoch == 3


@pytest.mark.parametrize(
    ("monitor", "metrics", "expected_improved"),
    [
        ("loss", [2.0, 1.0, 1.5, 0.5, 0.5], [True, True, False, True, False]),
        ("f1", [0.2, 0.6, 0.4, 0.9, 0.9], [True, True, False, True, False]),
    ],
)
def test_monitor_direction_decides_which_metric_moves_count_as_improvement(
    monitor: str, metrics: list[float], expected_improved: list[bool]
) -> None:
    stopper = EarlyStopping(
        patience=100, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor=monitor
    )
    history = _trace(stopper, list(enumerate(metrics)))
    assert [entry.improved for entry in history] == expected_improved


@pytest.mark.parametrize(
    ("monitor", "metrics"),
    [
        ("f1", [0.4, 0.9, 0.1, 0.95, 0.2, 0.3]),
        ("loss", [0.9, 0.4, 1.8, 0.35, 1.2, 1.0]),
    ],
)
def test_best_only_ever_moves_in_the_improving_direction(
    monitor: str, metrics: list[float]
) -> None:
    stopper = EarlyStopping(
        patience=100, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor=monitor
    )
    bests = [entry.best for entry in _trace(stopper, list(enumerate(metrics)))]
    pairs = list(pairwise(bests))
    if monitor == "f1":
        assert all(later >= earlier for earlier, later in pairs)
    else:
        assert all(later <= earlier for earlier, later in pairs)


def test_ema_alpha_of_one_makes_the_smoothed_value_equal_the_raw_value() -> None:
    stopper = EarlyStopping(ema_alpha=1.0, min_epochs=0, monitor="f1")
    for epoch, value in enumerate([0.13, 0.87, 0.42, 0.42, 0.99]):
        stopper.step(value, epoch=epoch)
        assert stopper.smoothed == pytest.approx(value)


def test_intermediate_ema_alpha_weights_the_new_observation_by_alpha() -> None:
    stopper = EarlyStopping(ema_alpha=0.25, min_epochs=0, min_delta=0.0, monitor="f1")
    stopper.step(0.8, epoch=0)
    assert stopper.smoothed == pytest.approx(0.8)
    stopper.step(0.4, epoch=1)
    assert stopper.smoothed == pytest.approx(0.7)
    stopper.step(0.4, epoch=2)
    assert stopper.smoothed == pytest.approx(0.625)


def test_ema_alpha_of_zero_freezes_the_smoothed_value_at_the_first_observation() -> None:
    stopper = EarlyStopping(ema_alpha=0.0, min_epochs=0, min_delta=0.0, patience=100, monitor="f1")
    stopper.step(0.3, epoch=0)
    for epoch, value in enumerate([0.99, 0.01, 0.55], start=1):
        stopper.step(value, epoch=epoch)
        assert stopper.smoothed == pytest.approx(0.3)
    assert stopper.best == pytest.approx(0.3)
    assert stopper.best_epoch == 0


def test_min_delta_suppresses_improvements_that_are_smaller_than_the_threshold() -> None:
    stopper = EarlyStopping(patience=100, min_delta=0.01, min_epochs=0, ema_alpha=1.0, monitor="f1")
    stopper.step(0.50, epoch=0)
    stopper.step(0.505, epoch=1)
    improved_on_small_gain = stopper.improved
    best_after_small_gain = stopper.best
    stopper.step(0.52, epoch=2)

    assert improved_on_small_gain is False
    assert best_after_small_gain == pytest.approx(0.50)
    assert stopper.improved is True
    assert stopper.best == pytest.approx(0.52)


def test_bookkeeping_across_a_scripted_run_with_warmup_min_delta_and_patience() -> None:
    stopper = EarlyStopping(patience=4, min_delta=0.01, min_epochs=2, ema_alpha=1.0, monitor="f1")
    schedule = [
        (0, 0.10),
        (1, 0.90),
        (2, 0.50),
        (3, 0.505),
        (4, 0.60),
        (5, 0.59),
        (6, 0.58),
        (7, 0.57),
        (8, 0.56),
    ]
    history = _trace(stopper, schedule)

    assert [entry.stop for entry in history] == [False] * 8 + [True]
    assert [entry.improved for entry in history] == [
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert [entry.counter for entry in history] == [0, 0, 0, 1, 0, 1, 2, 3, 4]
    assert [entry.best_epoch for entry in history] == [0, 0, 2, 2, 4, 4, 4, 4, 4]
    assert stopper.best == pytest.approx(0.60)


def test_state_dict_exposes_exactly_the_resumable_fields() -> None:
    stopper = EarlyStopping(patience=3, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="loss")
    stopper.step(1.0, epoch=0)
    stopper.step(2.0, epoch=1)

    state = stopper.state_dict()
    assert set(state) == {"best", "smoothed", "counter", "best_epoch"}
    assert state["best"] == pytest.approx(1.0)
    assert state["smoothed"] == pytest.approx(2.0)
    assert state["counter"] == 1
    assert state["best_epoch"] == 0


def test_load_state_dict_round_trips_counter_and_best_values() -> None:
    original = EarlyStopping(patience=3, min_delta=0.0, min_epochs=0, ema_alpha=0.5, monitor="f1")
    for epoch, value in enumerate([0.4, 0.9, 0.2, 0.2]):
        original.step(value, epoch=epoch)

    restored = EarlyStopping(patience=3, min_delta=0.0, min_epochs=0, ema_alpha=0.5, monitor="f1")
    restored.load_state_dict(original.state_dict())

    assert restored.best == pytest.approx(original.best)
    assert restored.smoothed == pytest.approx(original.smoothed)
    assert restored.counter == original.counter
    assert restored.best_epoch == original.best_epoch
    assert restored.state_dict() == original.state_dict()


def test_a_resumed_stopper_stops_at_the_same_step_as_an_uninterrupted_one() -> None:
    schedule = [(epoch, 0.9 - 0.05 * epoch) for epoch in range(8)]

    uninterrupted = EarlyStopping(
        patience=3, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="f1"
    )
    uninterrupted_stops = [entry.stop for entry in _trace(uninterrupted, schedule)]

    first_half = EarlyStopping(patience=3, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="f1")
    early_stops = [entry.stop for entry in _trace(first_half, schedule[:2])]
    resumed = EarlyStopping(patience=3, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="f1")
    resumed.load_state_dict(first_half.state_dict())
    late_stops = [entry.stop for entry in _trace(resumed, schedule[2:])]

    assert early_stops + late_stops == uninterrupted_stops
    assert resumed.best == pytest.approx(uninterrupted.best)
    assert resumed.counter == uninterrupted.counter


def test_load_state_dict_tolerates_legacy_states_without_smoothing_fields() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="loss")
    stopper.load_state_dict({"best": 0.25, "counter": 1})

    assert stopper.best == pytest.approx(0.25)
    assert stopper.smoothed is None
    assert stopper.counter == 1
    assert stopper.best_epoch == 0

    assert stopper.step(0.30, epoch=4) is True
    assert stopper.improved is False


def test_load_state_dict_requires_the_best_and_counter_keys() -> None:
    stopper = EarlyStopping(monitor="f1")
    with pytest.raises(KeyError, match="counter"):
        stopper.load_state_dict({"best": 0.5})
    with pytest.raises(KeyError, match="best"):
        stopper.load_state_dict({"counter": 1})
