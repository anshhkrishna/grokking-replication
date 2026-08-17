"""Modular addition dataset: every (a, b, (a + b) mod p) triple, tokenized and split.

Generated entirely in-process from a prime `p` — no external source, no download.
"""
import numpy as np

P = 97
TRAIN_FRAC = 0.5


def all_triples(p):
    """Return (a, b, c) arrays covering every (a, b, (a + b) mod p) pair, p*p total."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a = a.reshape(-1)
    b = b.reshape(-1)
    c = (a + b) % p
    return a, b, c


def tokenize(a, b, c, p):
    """Stack into (N, 4) rows [a, b, EQ, c]; EQ is the extra token id p, vocab_size = p + 1."""
    eq = np.full_like(a, p)
    return np.stack([a, b, eq, c], axis=1)


def split_indices(n, train_frac, seed):
    """A fixed random permutation split into disjoint (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    return perm[:n_train], perm[n_train:]


def make_dataset(p=P, train_frac=TRAIN_FRAC, seed=0):
    """Build the full tokenized dataset and split it once into train/test.

    Returns a dict with `train` and `test` arrays of shape (*, 4) holding
    [a, b, EQ, c] rows, plus `vocab_size`, `eq_token`, `p`, `train_frac`, `seed`.
    """
    a, b, c = all_triples(p)
    tokens = tokenize(a, b, c, p)
    train_idx, test_idx = split_indices(tokens.shape[0], train_frac, seed)
    return {
        "train": tokens[train_idx],
        "test": tokens[test_idx],
        "vocab_size": p + 1,
        "eq_token": p,
        "p": p,
        "train_frac": train_frac,
        "seed": seed,
    }


if __name__ == "__main__":
    ds = make_dataset()
    print(f"p={ds['p']} vocab_size={ds['vocab_size']} train={ds['train'].shape} test={ds['test'].shape}")
    print(f"sample train row: {ds['train'][0].tolist()}")
