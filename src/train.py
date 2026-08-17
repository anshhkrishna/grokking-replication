"""Training loop for the modular-addition transformer: decoupled-weight-decay Adam
(AdamW), written from scratch, trained on random minibatches drawn from the fixed
train split from `data.py`. Train and test accuracy are evaluated on the full
splits at regular step intervals (not on the minibatch, which would be a noisier
estimate).
"""
import numpy as np

from data import make_dataset
from model import backward, final_position_accuracy, final_position_loss, forward, init_params

N_LAYERS = 2
D_MODEL = 24
D_MLP = 96
BATCH_SIZE = 512
LEARNING_RATE = 2e-3
NUM_TRAIN_STEPS = 6000
LOG_EVERY = 200
WEIGHT_DECAY = 0.0

MODEL_SEED = 0
BATCH_SEED = 1

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


def _zeros_like_params(params):
    return {
        "W_E": np.zeros_like(params["W_E"]),
        "W_pos": np.zeros_like(params["W_pos"]),
        "W_U": np.zeros_like(params["W_U"]),
        "layers": [{k: np.zeros_like(v) for k, v in layer.items()} for layer in params["layers"]],
    }


def _adamw_update(arr, g, m, v, t, lr, weight_decay):
    """Decoupled weight decay (Loshchilov & Hutter): the decay term shrinks the
    weight directly, separate from the Adam moment estimates of the gradient.
    """
    m[:] = ADAM_BETA1 * m + (1 - ADAM_BETA1) * g
    v[:] = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (g * g)
    m_hat = m / (1 - ADAM_BETA1**t)
    v_hat = v / (1 - ADAM_BETA2**t)
    arr -= lr * (m_hat / (np.sqrt(v_hat) + ADAM_EPS) + weight_decay * arr)


def adamw_step(params, grads, m, v, t, lr, weight_decay):
    t = t + 1
    for key in ("W_E", "W_pos", "W_U"):
        _adamw_update(params[key], grads[key], m[key], v[key], t, lr, weight_decay)
    for layer, gl, ml, vl in zip(params["layers"], grads["layers"], m["layers"], v["layers"]):
        for key in layer:
            _adamw_update(layer[key], gl[key], ml[key], vl[key], t, lr, weight_decay)
    return t


def train_model(train_tokens, test_tokens, num_steps, weight_decay, seed_model,
                 seed_batch=0, n_layers=N_LAYERS, d_model=D_MODEL, d_mlp=D_MLP,
                 batch_size=BATCH_SIZE, vocab_size=None, seq_len=None,
                 lr=LEARNING_RATE, log_every=LOG_EVERY, print_log=True,
                 early_stop_threshold=None):
    """Minibatch AdamW training: every step draws a fresh random `batch_size`
    subset of `train_tokens` (with replacement across steps, without replacement
    within a step) for the gradient update. Every `log_every` steps, evaluates
    accuracy on the full train and test splits. Returns (params, log_rows), where
    each log row is (step, minibatch_train_loss, full_train_acc, full_test_acc).

    If `early_stop_threshold` is set, training stops as soon as a logged row has
    both train and test accuracy at or above it, since both crossing steps are
    already captured in `log_rows` at that point and further steps would only
    change the model's state after the metric of interest was already decided.
    """
    if vocab_size is None or seq_len is None:
        raise ValueError("vocab_size and seq_len must be provided")

    params = init_params(n_layers, d_model, d_mlp, vocab_size, seq_len, seed=seed_model)
    m = _zeros_like_params(params)
    v = _zeros_like_params(params)
    t = 0
    batch_rng = np.random.default_rng(seed_batch)
    n_train = train_tokens.shape[0]
    effective_batch = min(batch_size, n_train)

    log_rows = []
    for step in range(1, num_steps + 1):
        batch_idx = batch_rng.choice(n_train, size=effective_batch, replace=False)
        batch = train_tokens[batch_idx]

        logits, cache = forward(params, batch, cache=True)
        batch_loss = final_position_loss(logits, batch)
        grads = backward(params, batch, cache)
        t = adamw_step(params, grads, m, v, t, lr, weight_decay)

        if step == 1 or step % log_every == 0 or step == num_steps:
            train_logits = forward(params, train_tokens, cache=False)
            train_acc = final_position_accuracy(train_logits, train_tokens)
            test_logits = forward(params, test_tokens, cache=False)
            test_acc = final_position_accuracy(test_logits, test_tokens)
            log_rows.append((step, float(batch_loss), train_acc, test_acc))
            if print_log:
                print(f"step {step:5d}/{num_steps}  wd={weight_decay:.4g}  "
                      f"batch_loss={batch_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
            if (early_stop_threshold is not None and train_acc >= early_stop_threshold
                    and test_acc >= early_stop_threshold):
                break

    return params, log_rows


def main():
    ds = make_dataset()
    print(f"baseline: p={ds['p']} vocab_size={ds['vocab_size']} train={ds['train'].shape} "
          f"test={ds['test'].shape} weight_decay={WEIGHT_DECAY}")
    print(f"model: n_layers={N_LAYERS} d_model={D_MODEL} d_mlp={D_MLP} batch_size={BATCH_SIZE} "
          f"lr={LEARNING_RATE} num_steps={NUM_TRAIN_STEPS} model_seed={MODEL_SEED} batch_seed={BATCH_SEED}")
    train_model(
        ds["train"], ds["test"], num_steps=NUM_TRAIN_STEPS, weight_decay=WEIGHT_DECAY,
        seed_model=MODEL_SEED, seed_batch=BATCH_SEED, vocab_size=ds["vocab_size"],
        seq_len=ds["train"].shape[1], log_every=LOG_EVERY,
    )


if __name__ == "__main__":
    main()
