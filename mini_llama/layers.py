import numpy as np


def rms_norm(x, weight, eps=1e-5):
    # Normalize each vector by its root mean square, then scale by the learned weight.
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight


def linear(x, weight, bias=None):
    # GGUF tensors are stored with the first dimension corresponding
    # to the output/input convention used by the model.
    # The model code calls this as x @ weight.T.
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def silu(x):
    # Use a numerically stable SiLU implementation to avoid overflow in exp().
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = x[~pos] * exp_x / (1.0 + exp_x)
    return out
