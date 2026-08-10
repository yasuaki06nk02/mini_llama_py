import numpy as np

from .gguf import GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0


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
    if ggml_type == GGML_TYPE_Q5_0:
        return _block_count(shape, 32) * 22
    if ggml_type == GGML_TYPE_Q8_0:
        return _block_count(shape, 32) * 34
    if ggml_type == GGML_TYPE_Q4_K:
        return _block_count(shape, 256) * 144
    if ggml_type == GGML_TYPE_Q6_K:
        return _block_count(shape, 256) * 210

    raise NotImplementedError(f"GGML type {ggml_type} is not implemented")


def dequant_q4_0(data, n):
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 18)

    d = raw[:, :2].copy().view(np.float16).astype(np.float32)
    qs = raw[:, 2:].reshape((-1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((-1, 32)).astype(np.int8) - np.int8(8)

    return (d * qs.astype(np.float32)).reshape(-1)[:n]


def dequant_q8_0(data, n):
    # Q8_0 block:
    #   fp16 scale
    #   32 signed int8 values
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 34)

    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = raw[:, 2:].view(np.int8).astype(np.float32)

    return (qs * scales[:, None]).reshape(-1)[:n]


def dequant_q5_0(data, n):
    # Q5_0 block:
    #   fp16 scale
    #   4 bytes high bits
    #   16 bytes containing 32 unsigned 4-bit values
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 22)

    d = raw[:, :2].copy().view(np.float16).astype(np.float32)
    qh = raw[:, 2:6].view(np.uint32)
    qs = raw[:, 6:]

    qh = qh.reshape((raw.shape[0], 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = qs.reshape((raw.shape[0], -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((raw.shape[0], -1))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)

    return (d * q.astype(np.float32)).reshape(-1)[:n]


def dequant_q4_k(data, n):
    # Q4_K block:
    #   fp16 d
    #   fp16 dmin
    #   12 bytes packed scales/mins
    #   128 bytes 4-bit quants
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 144)

    d, rest = np.hsplit(raw, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [12])

    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    scales = scales.view(np.uint8)
    scales = scales.reshape((raw.shape[0], 3, 4))
    dsc, msc, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([dsc & 0x3F, (m_d & 0x0F) | ((dsc >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([msc & 0x3F, (m_d >> 4) | ((msc >> 2) & 0x30)], axis=-1)
    sc = sc.reshape((raw.shape[0], 8))
    mn = mn.reshape((raw.shape[0], 8))

    d = (d * sc.astype(np.float32)).reshape((raw.shape[0], -1, 1))
    dm = (dmin * mn.astype(np.float32)).reshape((raw.shape[0], -1, 1))

    qs = qs.reshape((raw.shape[0], -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((raw.shape[0], -1, 32)).astype(np.float32)

    return (d * qs - dm).reshape((raw.shape[0], 256)).reshape(-1)[:n]


def dequant_q6_k(data, n):
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 210)

    ql, rest = np.hsplit(raw, [128])
    qh, rest = np.hsplit(rest, [64])
    scales, d = np.hsplit(rest, [16])

    scales = scales.view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((raw.shape[0], 16, 1))

    ql = ql.reshape((raw.shape[0], -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((raw.shape[0], -1, 32))
    qh = qh.reshape((raw.shape[0], -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((raw.shape[0], -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((raw.shape[0], 16, -1)).astype(np.float32)

    return (d * q).reshape((raw.shape[0], 256))[:,:].reshape(-1)[:n]


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

    elif info.ggml_type == GGML_TYPE_Q5_0:
        x = dequant_q5_0(data, n)

    elif info.ggml_type == GGML_TYPE_Q8_0:
        x = dequant_q8_0(data, n)
    elif info.ggml_type == GGML_TYPE_Q4_K:
        x = dequant_q4_k(data, n)
    elif info.ggml_type == GGML_TYPE_Q6_K:
        x = dequant_q6_k(data, n)

    else:
        raise NotImplementedError(info.ggml_type)

    return x.reshape(info.shape)
