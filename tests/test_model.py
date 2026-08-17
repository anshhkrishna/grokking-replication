"""Checks the hand-derived backward pass against numerical gradients from finite
differences, plus basic sanity properties of the forward pass.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import (
    backward, final_position_accuracy, final_position_loss, forward, init_params,
)

N_LAYERS = 2
D_MODEL = 6
D_MLP = 10
SEQ_LEN = 4
VOCAB_SIZE = 7
BATCH_SIZE = 3
FD_EPS = 1e-5
SAMPLES_PER_ARRAY = 3
REL_ERROR_TOL = 1e-4


def _param_arrays(params, grads):
    """Flatten the nested params/grads structure into (param_array, grad_array,
    label) triples, so every weight in the model can be checked the same way
    regardless of where it lives in the structure.
    """
    pairs = [
        (params["W_E"], grads["W_E"], "W_E"),
        (params["W_pos"], grads["W_pos"], "W_pos"),
        (params["W_U"], grads["W_U"], "W_U"),
    ]
    for i, (p_layer, g_layer) in enumerate(zip(params["layers"], grads["layers"])):
        for key in p_layer:
            pairs.append((p_layer[key], g_layer[key], f"layers[{i}].{key}"))
    return pairs


def test_backward_matches_finite_differences():
    params = init_params(N_LAYERS, D_MODEL, D_MLP, VOCAB_SIZE, SEQ_LEN, seed=42)
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))

    logits, cache = forward(params, tokens, cache=True)
    base_loss = final_position_loss(logits, tokens)
    assert np.isfinite(base_loss)

    grads = backward(params, tokens, cache)

    def loss_of(p):
        return final_position_loss(forward(p, tokens, cache=False), tokens)

    idx_rng = np.random.default_rng(3)
    max_rel_error = 0.0
    for arr, grad_arr, label in _param_arrays(params, grads):
        flat_indices = idx_rng.choice(arr.size, size=min(SAMPLES_PER_ARRAY, arr.size), replace=False)
        for flat in flat_indices:
            multi_index = np.unravel_index(flat, arr.shape)
            original = arr[multi_index]

            arr[multi_index] = original + FD_EPS
            loss_plus = loss_of(params)
            arr[multi_index] = original - FD_EPS
            loss_minus = loss_of(params)
            arr[multi_index] = original

            numeric_grad = (loss_plus - loss_minus) / (2 * FD_EPS)
            analytic_grad = grad_arr[multi_index]
            denom = max(abs(numeric_grad), abs(analytic_grad), 1e-8)
            rel_error = abs(numeric_grad - analytic_grad) / denom
            max_rel_error = max(max_rel_error, rel_error)

    assert max_rel_error < REL_ERROR_TOL, f"max relative error {max_rel_error} exceeds {REL_ERROR_TOL}"


def test_forward_output_shape():
    params = init_params(N_LAYERS, D_MODEL, D_MLP, VOCAB_SIZE, SEQ_LEN, seed=1)
    tokens = np.zeros((BATCH_SIZE, SEQ_LEN), dtype=int)
    logits = forward(params, tokens, cache=False)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert np.isfinite(logits).all()


def test_causal_mask_blocks_future_tokens():
    """Changing the last token (c) must not change the logits used to predict it
    (position seq_len - 2, the EQ token): a causal model's prediction for c can
    depend only on [a, b, EQ], never on c itself.
    """
    params = init_params(N_LAYERS, D_MODEL, D_MLP, VOCAB_SIZE, SEQ_LEN, seed=2)
    rng = np.random.default_rng(9)
    tokens = rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    tokens_perturbed = tokens.copy()
    tokens_perturbed[:, -1] = (tokens_perturbed[:, -1] + 1) % VOCAB_SIZE

    logits = forward(params, tokens, cache=False)
    logits_perturbed = forward(params, tokens_perturbed, cache=False)
    np.testing.assert_allclose(logits[:, -2, :], logits_perturbed[:, -2, :])


def test_final_position_accuracy_matches_argmax():
    params = init_params(N_LAYERS, D_MODEL, D_MLP, VOCAB_SIZE, SEQ_LEN, seed=5)
    rng = np.random.default_rng(11)
    tokens = rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    logits = forward(params, tokens, cache=False)
    acc = final_position_accuracy(logits, tokens)
    pred = logits[:, -2, :].argmax(axis=-1)
    expected = float((pred == tokens[:, -1]).mean())
    assert acc == pytest.approx(expected)
