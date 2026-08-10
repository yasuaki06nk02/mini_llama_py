import numpy as np

from .gguf import GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0


def _block_count(shape, block_size):
    n = int(np.prod(shape))
    if n % block_size != 0:
        raise ValueError(f"tensor has {n} elements, not divisible by {block_size}")
    return n // block_size


def tensor_nbytes(shape, ggml_type):
    n = int(np.prod(shape))

    if ggml_type == GGML_TYPE_F32:
        return n * 4
    if ggml_type == GGML_TYPE_F16:
        return n * 2
    if ggml_type == GGML_TYPE_Q4_0:
        return _block_count(shape, 32) * 18
    if ggml_type == GGML_TYPE_Q8_0:
        return _block_count(shape, 32) * 34

    raise NotImplementedError(f"GGML type {ggml_type} is not implemented")


def dequant_q4_0(data, n):
    # Q4_0 block:
    #   fp16 scale
    #   16 bytes containing 32 unsigned 4-bit values
    raw = np.frombuffer(data, dtype=np.uint8)
    blocks = raw.reshape(-1, 18)

    scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = blocks[:, 2:]

    out = np.empty((len(blocks), 32), dtype=np.float32)

    lo = qs & 0x0F
    hi = qs >> 4

    out[:, 0:16] = (lo.astype(np.float32) - 8.0) * scales[:, None]
    out[:, 16:32] = (hi.astype(np.float32) - 8.0) * scales[:, None]

    return out.reshape(-1)[:n]


def dequant_q8_0(data, n):
    # Q8_0 block:
    #   fp16 scale
    #   32 signed int8 values
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 34)

    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = raw[:, 2:].view(np.int8).astype(np.float32)

    return (qs * scales[:, None]).reshape(-1)[:n]


def load_tensor(reader, info):
    n = int(np.prod(info.shape))
    nbytes = tensor_nbytes(info.shape, info.ggml_type)
    data = reader.tensor_bytes(info, nbytes)

    if info.ggml_type == GGML_TYPE_F32:
        x = np.frombuffer(data, dtype="<f4", count=n).copy()

    elif info.ggml_type == GGML_TYPE_F16:
        x = np.frombuffer(data, dtype="<f2", count=n).astype(np.float32)

    elif info.ggml_type == GGML_TYPE_Q4_0:
        x = dequant_q4_0(data, n)

    elif info.ggml_type == GGML_TYPE_Q8_0:
        x = dequant_q8_0(data, n)

    else:
        raise NotImplementedError(info.ggml_type)

    return x.reshape(info.shape)
