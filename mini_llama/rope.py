import numpy as np


def apply_rope(x, position, theta=10000.0):
    # x: [n_heads, head_dim]
    n_heads, head_dim = x.shape
    half = head_dim // 2

    freq = 1.0 / (
        theta ** (np.arange(0, half, dtype=np.float32) / half)
    )

    angles = position * freq
    cos = np.cos(angles)
    sin = np.sin(angles)

    a = x[:, :half]
    b = x[:, half:2 * half]

    y = np.empty_like(x)
    y[:, :half] = a * cos[None, :] - b * sin[None, :]
    y[:, half:2 * half] = a * sin[None, :] + b * cos[None, :]

    if head_dim > 2 * half:
        y[:, 2 * half:] = x[:, 2 * half:]

    return y
