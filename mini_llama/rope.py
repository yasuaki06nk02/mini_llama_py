import numpy as np


def apply_rope(
    x,
    position,
    theta=10000.0,
    mode="split",
):
    """
    x:
        [n_heads, head_dim]

    mode:
        "split"      : current mini-llama implementation
        "interleaved": adjacent pair implementation
    """

    n_heads, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError(
            f"RoPE head_dim must be even, got {head_dim}"
        )

    half = head_dim // 2

    freq = 1.0 / (
        theta ** (
            np.arange(
                half,
                dtype=np.float32,
            ) / half
        )
    )

    angles = position * freq

    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)

    if mode == "split":

        a = x[:, :half]
        b = x[:, half:]

        y = np.empty_like(x)

        y[:, :half] = (
            a * cos[None, :]
            - b * sin[None, :]
        )

        y[:, half:] = (
            a * sin[None, :]
            + b * cos[None, :]
        )

        return y

    elif mode == "interleaved":

        a = x[:, 0::2]
        b = x[:, 1::2]

        y = np.empty_like(x)

        y[:, 0::2] = (
            a * cos[None, :]
            - b * sin[None, :]
        )

        y[:, 1::2] = (
            a * sin[None, :]
            + b * cos[None, :]
        )

        return y

    else:
        raise ValueError(
            f"Unknown RoPE mode: {mode}"
        )