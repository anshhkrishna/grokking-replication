"""Decoder-only transformer for the [a, b, EQ, c] modular-addition sequences: a
couple of causal self-attention + MLP blocks, learned position embeddings, single
head per layer, no normalization layers. Forward and backward passes are
hand-derived and hand-written in NumPy; nothing here depends on an autograd
framework.
"""
import numpy as np

_CAUSAL_MASK_CACHE = {}


def _causal_mask(seq_len):
    if seq_len not in _CAUSAL_MASK_CACHE:
        _CAUSAL_MASK_CACHE[seq_len] = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return _CAUSAL_MASK_CACHE[seq_len]


def init_params(n_layers, d_model, d_mlp, vocab_size, seq_len, seed):
    """Random parameters for an n_layers-layer transformer. Attention weights are
    initialized from N(0, 1/d_model); MLP weights from N(0, 1/d_mlp) on the output
    projection and N(0, 1/d_model) on the input projection, with zero biases.
    """
    rng = np.random.default_rng(seed)
    attn_scale = 1.0 / np.sqrt(d_model)
    mlp_in_scale = 1.0 / np.sqrt(d_model)
    mlp_out_scale = 1.0 / np.sqrt(d_mlp)

    def mat(rows, cols, scale):
        return rng.normal(0.0, scale, size=(rows, cols))

    params = {
        "W_E": mat(vocab_size, d_model, attn_scale),
        "W_pos": mat(seq_len, d_model, attn_scale),
        "W_U": mat(d_model, vocab_size, attn_scale),
        "layers": [
            {
                "W_Q": mat(d_model, d_model, attn_scale),
                "W_K": mat(d_model, d_model, attn_scale),
                "W_V": mat(d_model, d_model, attn_scale),
                "W_O": mat(d_model, d_model, attn_scale),
                "W_in": mat(d_model, d_mlp, mlp_in_scale),
                "b_in": np.zeros(d_mlp),
                "W_out": mat(d_mlp, d_model, mlp_out_scale),
                "b_out": np.zeros(d_model),
            }
            for _ in range(n_layers)
        ],
    }
    return params


def forward(params, tokens, cache=False):
    """Run the model on a (batch, seq_len) array of token ids. Returns logits of
    shape (batch, seq_len, vocab_size). Each layer is a residual attention block
    followed by a residual ReLU MLP block, both causally masked by construction
    (attention is masked; the MLP acts per-position). With cache=True, also
    returns the intermediate activations `backward` needs.
    """
    batch_size, seq_len = tokens.shape
    d_model = params["W_E"].shape[1]
    mask = _causal_mask(seq_len)

    X = params["W_E"][tokens] + params["W_pos"][None, :seq_len, :]
    layer_cache = []
    for layer in params["layers"]:
        Q = X @ layer["W_Q"]
        K = X @ layer["W_K"]
        V = X @ layer["W_V"]
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_model)
        scores = np.where(mask, -np.inf, scores)
        shifted = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        A = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        Z = A @ V
        O = Z @ layer["W_O"]
        X_attn = X + O

        H_pre = X_attn @ layer["W_in"] + layer["b_in"]
        H = np.maximum(H_pre, 0.0)
        M = H @ layer["W_out"] + layer["b_out"]
        X_next = X_attn + M

        if cache:
            layer_cache.append({
                "X": X, "Q": Q, "K": K, "V": V, "A": A, "Z": Z,
                "X_attn": X_attn, "H_pre": H_pre, "H": H,
            })
        X = X_next

    logits = X @ params["W_U"]
    if cache:
        return logits, {"X_final": X, "layers": layer_cache}
    return logits


def final_position_loss(logits, tokens):
    """Cross-entropy loss on predicting the last token (c) from the second-to-last
    position's logits (the EQ token, at index seq_len - 2). This is the "final-
    position prediction" the task actually cares about, unlike a full next-token
    loss over every position.
    """
    pred_logits = logits[:, -2, :]
    targets = tokens[:, -1]
    shifted = pred_logits - pred_logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    b = pred_logits.shape[0]
    nll = -log_probs[np.arange(b), targets]
    return nll.mean()


def final_position_accuracy(logits, tokens):
    """Fraction of rows where argmax(logits[:, -2, :]) equals the target token
    tokens[:, -1].
    """
    pred = logits[:, -2, :].argmax(axis=-1)
    return float((pred == tokens[:, -1]).mean())


def _contract_bs(x, y):
    """sum_{b,s} x[b,s,:] outer y[b,s,:] -> (x.shape[-1], y.shape[-1]), via a
    reshape to 2D so the reduction runs as one BLAS matmul instead of an
    element-by-element einsum loop.
    """
    d, e = x.shape[-1], y.shape[-1]
    return x.reshape(-1, d).T @ y.reshape(-1, e)


def backward(params, tokens, cache):
    """Gradient of final_position_loss(forward(params, tokens, cache=True), tokens)
    with respect to every array in `params`, returned in the same nested structure.
    Derived by hand: softmax-cross-entropy backward at the single predicted
    position, then per layer (in reverse), the MLP block (ReLU, two linear
    layers), then the attention block (softmax backward, then the linear Q/K/V/O
    projections), accumulated through the residual stream.
    """
    batch_size, seq_len = tokens.shape
    d_model = params["W_E"].shape[1]
    mask = _causal_mask(seq_len)

    X_final = cache["X_final"]
    logits = X_final @ params["W_U"]
    pred_logits = logits[:, -2, :]
    targets = tokens[:, -1]
    b, v = pred_logits.shape

    shifted = pred_logits - pred_logits.max(axis=-1, keepdims=True)
    exp_shift = np.exp(shifted)
    probs = exp_shift / exp_shift.sum(axis=-1, keepdims=True)
    onehot = np.zeros_like(probs)
    onehot[np.arange(b), targets] = 1.0
    dpred_logits = (probs - onehot) / b

    dlogits = np.zeros_like(logits)
    dlogits[:, -2, :] = dpred_logits

    grads = {"W_U": _contract_bs(X_final, dlogits)}
    dX = dlogits @ params["W_U"].T

    grads["layers"] = [None] * len(params["layers"])
    for l in reversed(range(len(params["layers"]))):
        layer = params["layers"][l]
        lc = cache["layers"][l]

        # MLP block backward: dX splits into the residual path and the path
        # through H = relu(X_attn @ W_in + b_in), M = H @ W_out + b_out.
        dM = dX
        dWout = _contract_bs(lc["H"], dM)
        dbout = dM.sum(axis=(0, 1))
        dH = dM @ layer["W_out"].T
        dH_pre = dH * (lc["H_pre"] > 0)
        dWin = _contract_bs(lc["X_attn"], dH_pre)
        dbin = dH_pre.sum(axis=(0, 1))
        dX_attn_from_mlp = dH_pre @ layer["W_in"].T
        dX_attn = dX + dX_attn_from_mlp

        # Attention block backward.
        dO = dX_attn
        dWO = _contract_bs(lc["Z"], dO)
        dZ = dO @ layer["W_O"].T

        # dA[b,s,t] = sum_d dZ[b,s,d] * V[b,t,d] = dZ[b] @ V[b].T (batched).
        dA = dZ @ lc["V"].transpose(0, 2, 1)
        # dV[b,t,d] = sum_s A[b,s,t] * dZ[b,s,d] = A[b].T @ dZ[b] (batched).
        dV = lc["A"].transpose(0, 2, 1) @ dZ

        sum_term = np.sum(lc["A"] * dA, axis=-1, keepdims=True)
        dscores = lc["A"] * (dA - sum_term)
        dscores = np.where(mask, 0.0, dscores)
        dscores = dscores / np.sqrt(d_model)

        dQ = dscores @ lc["K"]
        # dK[b,t,d] = sum_s dscores[b,s,t] * Q[b,s,d] = dscores[b].T @ Q[b] (batched).
        dK = dscores.transpose(0, 2, 1) @ lc["Q"]

        dWQ = _contract_bs(lc["X"], dQ)
        dWK = _contract_bs(lc["X"], dK)
        dWV = _contract_bs(lc["X"], dV)

        dX_from_attn = dQ @ layer["W_Q"].T + dK @ layer["W_K"].T + dV @ layer["W_V"].T
        dX = dX_attn + dX_from_attn

        grads["layers"][l] = {
            "W_Q": dWQ, "W_K": dWK, "W_V": dWV, "W_O": dWO,
            "W_in": dWin, "b_in": dbin, "W_out": dWout, "b_out": dbout,
        }

    grads["W_pos"] = np.zeros_like(params["W_pos"])
    grads["W_pos"][:seq_len] = dX.sum(axis=0)
    grads["W_E"] = np.zeros_like(params["W_E"])
    np.add.at(grads["W_E"], tokens, dX)

    return grads
