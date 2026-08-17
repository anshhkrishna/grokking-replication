"""Multi-seed rigor pass: repeats the weight-decay sweep across several
(model, batch) seed pairs, all sharing the single, fixed train/test split from
`data.py`, and reports the generalization step's mean and spread per weight-decay
value plus the memorization-to-generalization gap, alongside the final test
accuracy each run reaches within the step budget (a metric that stays
well-defined even for seeds that never cross the generalization threshold).
"""
import numpy as np

from data import make_dataset
from experiment import GEN_THRESHOLD, first_crossing
from train import (
    BATCH_SIZE,
    D_MLP,
    D_MODEL,
    LEARNING_RATE,
    LOG_EVERY,
    N_LAYERS,
    train_model,
)

WEIGHT_DECAYS = [0.0, 0.3, 1.0]
SEEDS = [0, 1, 2]
NUM_TRAIN_STEPS = 2500


def run_seed(train_tokens, test_tokens, vocab_size, seq_len, seed):
    """One full weight-decay sweep at a given seed. `seed` sets both the model
    init seed and the batch-sampling seed (offset by one, matching the
    MODEL_SEED=0 / BATCH_SEED=1 convention from `train.py`). The train/test
    split itself is not reseeded here: it is built once by the caller from a
    fixed `data.py` seed and reused unchanged across every seed and weight
    decay value in the sweep.
    """
    results = []
    for wd in WEIGHT_DECAYS:
        print(f"=== seed={seed} weight_decay={wd:.4g} ===")
        _, log_rows = train_model(
            train_tokens, test_tokens, num_steps=NUM_TRAIN_STEPS, weight_decay=wd,
            seed_model=seed, seed_batch=seed + 1, n_layers=N_LAYERS, d_model=D_MODEL,
            d_mlp=D_MLP, batch_size=BATCH_SIZE, vocab_size=vocab_size, seq_len=seq_len,
            lr=LEARNING_RATE, log_every=LOG_EVERY, early_stop_threshold=GEN_THRESHOLD,
        )
        train_step = first_crossing(log_rows, 2, GEN_THRESHOLD)
        test_step = first_crossing(log_rows, 3, GEN_THRESHOLD)
        gap = (test_step - train_step) if (train_step is not None and test_step is not None) else None
        final_train_acc, final_test_acc = log_rows[-1][2], log_rows[-1][3]
        train_str = train_step if train_step is not None else "not reached"
        test_str = test_step if test_step is not None else "not reached"
        gap_str = gap if gap is not None else "n/a"
        print(f"seed={seed}  weight_decay={wd:.4g}  generalization_step={test_str}  "
              f"memorization_step={train_str}  gap={gap_str}  "
              f"final_train_acc={final_train_acc:.4f}  final_test_acc={final_test_acc:.4f}")
        results.append((seed, wd, train_step, test_step, gap, final_train_acc, final_test_acc))
    return results


def summarize(all_results):
    """Per weight-decay value: how many seeds reached the threshold, the
    mean/std of generalization step and gap across the seeds that did (a
    weight-decay value where no seed reaches threshold reports these as "n/a"
    rather than a number computed from zero samples), and the mean/std of
    final test accuracy across every seed regardless of whether it crossed
    the threshold.
    """
    summary = []
    for wd in WEIGHT_DECAYS:
        rows = [r for r in all_results if r[1] == wd]
        reached = [r for r in rows if r[3] is not None]
        n_reached = len(reached)
        n_total = len(rows)
        if n_reached > 0:
            gen_steps = np.array([r[3] for r in reached], dtype=float)
            gaps = np.array([r[4] for r in reached], dtype=float)
            gen_mean, gen_std = float(gen_steps.mean()), float(gen_steps.std())
            gap_mean, gap_std = float(gaps.mean()), float(gaps.std())
        else:
            gen_mean = gen_std = gap_mean = gap_std = None
        final_test_accs = np.array([r[6] for r in rows], dtype=float)
        final_test_mean, final_test_std = float(final_test_accs.mean()), float(final_test_accs.std())
        summary.append((wd, n_reached, n_total, gen_mean, gen_std, gap_mean, gap_std,
                         final_test_mean, final_test_std))
    return summary


def main():
    ds = make_dataset()
    print(f"rigor: p={ds['p']} vocab_size={ds['vocab_size']} train={ds['train'].shape} "
          f"test={ds['test'].shape} weight_decays={WEIGHT_DECAYS} seeds={SEEDS} "
          f"threshold={GEN_THRESHOLD} num_steps={NUM_TRAIN_STEPS}")
    print(f"model: n_layers={N_LAYERS} d_model={D_MODEL} d_mlp={D_MLP} batch_size={BATCH_SIZE} "
          f"lr={LEARNING_RATE}")
    print("split generated once by make_dataset() above; every seed and weight-decay "
          "run below reuses ds['train']/ds['test'] unchanged.")

    all_results = []
    for seed in SEEDS:
        all_results.extend(run_seed(ds["train"], ds["test"], ds["vocab_size"], ds["train"].shape[1], seed))

    print("=== summary (mean +/- std over seeds that reached threshold; "
          "final_test_acc is mean +/- std over all seeds) ===")
    for (wd, n_reached, n_total, gen_mean, gen_std, gap_mean, gap_std,
         final_test_mean, final_test_std) in summarize(all_results):
        if gen_mean is None:
            print(f"weight_decay={wd:.4g}  reached={n_reached}/{n_total}  "
                  f"generalization_step=n/a  gap=n/a  "
                  f"final_test_acc={final_test_mean:.4f}+/-{final_test_std:.4f}")
        else:
            print(f"weight_decay={wd:.4g}  reached={n_reached}/{n_total}  "
                  f"generalization_step={gen_mean:.1f}+/-{gen_std:.1f}  "
                  f"gap={gap_mean:.1f}+/-{gap_std:.1f}  "
                  f"final_test_acc={final_test_mean:.4f}+/-{final_test_std:.4f}")


if __name__ == "__main__":
    main()
