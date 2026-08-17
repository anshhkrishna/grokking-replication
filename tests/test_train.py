import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_dataset
from train import train_model


def test_split_not_mutated_by_training():
    """The leakage check: `train_model` takes the already-split train/test
    token arrays as input and must never rewrite them, so the split stays the
    single, frozen one `make_dataset` produced -- not something a training run
    (or a later weight-decay/seed in a sweep) could redraw or leak between.
    """
    ds = make_dataset(p=13, train_frac=0.5, seed=0)
    train_before = ds["train"].copy()
    test_before = ds["test"].copy()
    train_model(
        ds["train"], ds["test"], num_steps=20, weight_decay=0.3, seed_model=0,
        seed_batch=1, vocab_size=ds["vocab_size"], seq_len=ds["train"].shape[1],
        log_every=10, print_log=False,
    )
    assert np.array_equal(ds["train"], train_before)
    assert np.array_equal(ds["test"], test_before)


def test_same_seeds_reproduce_same_log():
    ds = make_dataset(p=13, train_frac=0.5, seed=0)
    kwargs = dict(
        num_steps=20, weight_decay=0.3, seed_model=0, seed_batch=1,
        vocab_size=ds["vocab_size"], seq_len=ds["train"].shape[1], log_every=10,
        print_log=False,
    )
    _, rows_a = train_model(ds["train"], ds["test"], **kwargs)
    _, rows_b = train_model(ds["train"], ds["test"], **kwargs)
    assert rows_a == rows_b


def test_early_stop_threshold_stops_at_first_logged_row_that_clears_it():
    ds = make_dataset(p=13, train_frac=0.5, seed=0)
    _, rows = train_model(
        ds["train"], ds["test"], num_steps=100, weight_decay=0.0, seed_model=0,
        seed_batch=1, vocab_size=ds["vocab_size"], seq_len=ds["train"].shape[1],
        log_every=10, print_log=False, early_stop_threshold=0.0,
    )
    # Accuracy is always >= 0.0, so the very first logged row already clears
    # the threshold and training should stop there rather than running all
    # 100 steps.
    assert len(rows) == 1
    assert rows[0][0] == 1


def test_early_stop_threshold_none_runs_full_budget():
    ds = make_dataset(p=13, train_frac=0.5, seed=0)
    _, rows = train_model(
        ds["train"], ds["test"], num_steps=30, weight_decay=0.0, seed_model=0,
        seed_batch=1, vocab_size=ds["vocab_size"], seq_len=ds["train"].shape[1],
        log_every=10, print_log=False, early_stop_threshold=None,
    )
    assert rows[-1][0] == 30


def test_weight_decay_improves_final_test_accuracy_across_seeds():
    """A fast, small-scale (p=13) stand-in for the full claim measured for real
    in `results/rigor.log`: at the same step budget, nonzero weight decay
    reaches meaningfully higher test accuracy than zero weight decay. The full-
    scale (p=97) multi-seed sweep in `results/rigor.log` shows the same
    direction but, at the step budget the 10-minute CPU limit allows, only 1 of
    9 seed/weight-decay runs actually crosses the strict 99% "generalized"
    threshold -- so the literal threshold-and-shrinking-delay pattern from the
    claim does not hold cleanly within budget. What does hold, at both this
    small scale and the full scale, is the weaker but real pattern this test
    checks: every nonzero-weight-decay run's final test accuracy beats every
    zero-weight-decay run's, for every seed tried.
    """
    ds = make_dataset(p=13, train_frac=0.7, seed=0)
    kwargs = dict(
        num_steps=2000, n_layers=2, d_model=16, d_mlp=64, batch_size=64,
        vocab_size=ds["vocab_size"], seq_len=ds["train"].shape[1], lr=3e-3,
        log_every=500, print_log=False,
    )
    seeds = [0, 1, 2]
    zero_wd_acc = []
    high_wd_acc = []
    for seed in seeds:
        _, rows = train_model(
            ds["train"], ds["test"], weight_decay=0.0, seed_model=seed,
            seed_batch=seed + 1, **kwargs,
        )
        zero_wd_acc.append(rows[-1][3])
        _, rows = train_model(
            ds["train"], ds["test"], weight_decay=1.0, seed_model=seed,
            seed_batch=seed + 1, **kwargs,
        )
        high_wd_acc.append(rows[-1][3])

    assert min(high_wd_acc) > max(zero_wd_acc)
