import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_dataset, all_triples, P, TRAIN_FRAC


def test_vocab_and_shape():
    ds = make_dataset(p=13, train_frac=0.5, seed=0)
    assert ds["vocab_size"] == 14
    assert ds["eq_token"] == 13
    assert ds["train"].shape[1] == 4
    assert ds["test"].shape[1] == 4


def test_eq_token_column():
    ds = make_dataset(p=13, train_frac=0.5, seed=0)
    assert np.all(ds["train"][:, 2] == ds["eq_token"])
    assert np.all(ds["test"][:, 2] == ds["eq_token"])


def test_labels_match_modular_addition():
    ds = make_dataset(p=13, train_frac=0.5, seed=0)
    rows = np.concatenate([ds["train"], ds["test"]], axis=0)
    a, b, eq, c = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    assert np.array_equal(c, (a + b) % ds["p"])


def test_train_test_disjoint_and_covers_all_pairs():
    p = 11
    ds = make_dataset(p=p, train_frac=0.5, seed=0)
    train_pairs = {tuple(row[:2]) for row in ds["train"]}
    test_pairs = {tuple(row[:2]) for row in ds["test"]}
    assert train_pairs.isdisjoint(test_pairs)
    all_pairs = {(a, b) for a in range(p) for b in range(p)}
    assert train_pairs | test_pairs == all_pairs
    assert len(train_pairs) + len(test_pairs) == p * p


def test_split_size_matches_train_frac():
    p = 20
    ds = make_dataset(p=p, train_frac=0.5, seed=0)
    assert ds["train"].shape[0] == p * p // 2
    assert ds["test"].shape[0] == p * p - p * p // 2


def test_same_seed_reproduces_same_split():
    a = make_dataset(p=17, train_frac=0.5, seed=7)
    b = make_dataset(p=17, train_frac=0.5, seed=7)
    assert np.array_equal(a["train"], b["train"])
    assert np.array_equal(a["test"], b["test"])


def test_different_seed_gives_different_split():
    a = make_dataset(p=17, train_frac=0.5, seed=1)
    b = make_dataset(p=17, train_frac=0.5, seed=2)
    assert not np.array_equal(a["train"], b["train"])


def test_all_triples_count():
    p = 9
    a, b, c = all_triples(p)
    assert a.shape[0] == p * p
    assert np.array_equal(c, (a + b) % p)


def test_default_config():
    assert P == 97
    assert TRAIN_FRAC == 0.5
