"""Weight-decay sweep: runs the full training loop from `train.py` once per
weight-decay value, all other settings held identical to the zero-decay
baseline, and reports the step at which train and test accuracy each first
cross a fixed threshold.
"""
import numpy as np

from data import make_dataset
from train import (
    BATCH_SEED,
    BATCH_SIZE,
    D_MLP,
    D_MODEL,
    LEARNING_RATE,
    LOG_EVERY,
    MODEL_SEED,
    N_LAYERS,
    NUM_TRAIN_STEPS,
    train_model,
)

WEIGHT_DECAYS = [0.0, 0.1, 0.3, 1.0]
GEN_THRESHOLD = 0.99


def first_crossing(log_rows, col, threshold):
    """Return the step at which `col` (2 for train_acc, 3 for test_acc) first
    reaches `threshold`, or None if it never does within the logged steps.
    """
    for row in log_rows:
        if row[col] >= threshold:
            return row[0]
    return None


def run_sweep(train_tokens, test_tokens, vocab_size, seq_len):
    results = []
    for wd in WEIGHT_DECAYS:
        print(f"=== weight_decay={wd:.4g} ===")
        _, log_rows = train_model(
            train_tokens, test_tokens, num_steps=NUM_TRAIN_STEPS, weight_decay=wd,
            seed_model=MODEL_SEED, seed_batch=BATCH_SEED, n_layers=N_LAYERS,
            d_model=D_MODEL, d_mlp=D_MLP, batch_size=BATCH_SIZE, vocab_size=vocab_size,
            seq_len=seq_len, lr=LEARNING_RATE, log_every=LOG_EVERY,
        )
        train_step = first_crossing(log_rows, 2, GEN_THRESHOLD)
        test_step = first_crossing(log_rows, 3, GEN_THRESHOLD)
        train_str = train_step if train_step is not None else "not reached"
        test_str = test_step if test_step is not None else "not reached"
        print(f"weight_decay={wd:.4g}  train>={GEN_THRESHOLD:.2f} at step {train_str}  "
              f"test>={GEN_THRESHOLD:.2f} at step {test_str}")
        results.append((wd, train_step, test_step))
    return results


def main():
    ds = make_dataset()
    print(f"experiment: p={ds['p']} vocab_size={ds['vocab_size']} train={ds['train'].shape} "
          f"test={ds['test'].shape} weight_decays={WEIGHT_DECAYS} threshold={GEN_THRESHOLD}")
    print(f"model: n_layers={N_LAYERS} d_model={D_MODEL} d_mlp={D_MLP} batch_size={BATCH_SIZE} "
          f"lr={LEARNING_RATE} num_steps={NUM_TRAIN_STEPS} model_seed={MODEL_SEED} batch_seed={BATCH_SEED}")

    results = run_sweep(ds["train"], ds["test"], ds["vocab_size"], ds["train"].shape[1])

    print("=== summary ===")
    for wd, train_step, test_step in results:
        train_str = train_step if train_step is not None else "not reached"
        test_str = test_step if test_step is not None else "not reached"
        gap = (test_step - train_step) if (train_step is not None and test_step is not None) else None
        gap_str = gap if gap is not None else "n/a"
        print(f"weight_decay={wd:.4g}  generalization_step={test_str}  "
              f"memorization_step={train_str}  gap={gap_str}")


if __name__ == "__main__":
    main()
