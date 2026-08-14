import numpy as np

from .layers import linear, rms_norm
from .rope import apply_rope


class KVCache:
    def __init__(self):
        # Store keys and values for all previous tokens in the sequence.
        self.k = []
        self.v = []

    def append(self, k, v):
        self.k.append(k)
        self.v.append(v)

    def arrays(self):
        return np.stack(self.k, axis=0), np.stack(self.v, axis=0)


def attention(
    x,
    wq,
    bq,
    wk,
    bk,
    wv,
    bv,
    wo,
    position,
    cache,
    n_heads,
    n_kv_heads,
    head_dim,
    rope_theta,
    rope_mode="split",
    q_norm=None,
    k_norm=None,
    rms_eps=1e-6,
):
    
    #print("RoPE theta:",rope_theta)

    # Project the current hidden state into Q, K, and V vectors.
    q = linear(x, wq, bq).reshape(n_heads, head_dim)
    k = linear(x, wk, bk).reshape(n_kv_heads, head_dim)
    v = linear(x, wv, bv).reshape(n_kv_heads, head_dim)

    # Qwen3 applies RMSNorm to Q and K before RoPE.
    if q_norm is not None:
        #print("Q RMSNorm: ON")
        q = rms_norm(q, q_norm,rms_eps)


    if k_norm is not None:
        #print("K RMSNorm: ON")
        k = rms_norm(k, k_norm,rms_eps)


    # Apply RoPE before caching so future tokens can attend to these positions.
    q = apply_rope(q, position, rope_theta, rope_mode,)
    k = apply_rope(k, position, rope_theta, rope_mode,)

    # Append the current token to the KV cache and fetch the full history.
    cache.append(k, v)
    keys, values = cache.arrays()

    # GQA: repeat KV heads to match Q heads.
    group = n_heads // n_kv_heads
    key_heads = np.repeat(keys, group, axis=1)
    value_heads = np.repeat(values, group, axis=1)

    # q: [H,D]
    # key_heads: [T,H,D]
    scores = np.einsum("hd,thd->ht", q, key_heads).astype(np.float32, copy=False)
    scores *= 1.0 / np.sqrt(head_dim)

    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores).astype(np.float32, copy=False)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    out = np.einsum("ht,thd->hd", probs, value_heads)
    out = out.reshape(n_heads * head_dim)

    return linear(out, wo)
