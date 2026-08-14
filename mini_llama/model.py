import numpy as np

from .attention import KVCache, attention
from .ffn import swiglu
from .layers import linear, rms_norm


class LlamaModel:
    def __init__(self, reader, tensor_loader):
        self.reader = reader
        self.load = tensor_loader

        meta = reader.metadata

        arch = meta.get("general.architecture", "llama")
        self.arch = arch
        # Map the GGUF architecture name to the metadata prefix used by its tensors.
        self.prefix = self._resolve_prefix(arch, meta)

        if self.prefix is None:
            raise ValueError(
                f"This educational implementation supports llama/qwen families only, got {arch}"
            )

        self.n_layers = int(meta[f"{self.prefix}.block_count"])
        self.embedding_dim = int(meta[f"{self.prefix}.embedding_length"])
        self.n_heads = int(meta[f"{self.prefix}.attention.head_count"])
        self.n_kv_heads = int(meta.get(f"{self.prefix}.attention.head_count_kv", self.n_heads))
        self.head_dim = int(meta.get(f"{self.prefix}.attention.key_length", self.embedding_dim // self.n_heads))
        self.ffn_dim = int(meta[f"{self.prefix}.feed_forward_length"])
        self.rope_theta = float(meta.get(f"{self.prefix}.rope.freq_base", 10000.0))
        self.rms_eps = float(meta.get(f"{self.prefix}.attention.layer_norm_rms_epsilon", 1e-5))

        if self.prefix == "qwen3":
            #self.rope_mode = "interleaved"
            self.rope_mode = "split"
        else:
            self.rope_mode = "split"
        print(f"Using RoPE mode: {self.rope_mode}")

        # One KV cache per transformer layer.
        self.cache = [KVCache() for _ in range(self.n_layers)]

        self.emb = self._tensor("token_embd.weight")
        self.norm = self._tensor("output_norm.weight")
        # Some Qwen models use tied input/output embeddings.
        if "output.weight" in reader.tensors:
            self.output = self._tensor("output.weight")
        else:
            self.output = self.emb

        # Collect all tensors needed for each transformer block.
        self.layers = []
        for i in range(self.n_layers):
            p = f"blk.{i}."
            self.layers.append({
                "attn_norm": self._tensor(p + "attn_norm.weight"),
                "wq": self._tensor(p + "attn_q.weight"),
                "bq": self._optional_tensor(p + "attn_q.bias"),
                "wk": self._tensor(p + "attn_k.weight"),
                "bk": self._optional_tensor(p + "attn_k.bias"),
                "wv": self._tensor(p + "attn_v.weight"),
                "bv": self._optional_tensor(p + "attn_v.bias"),
                "wo": self._tensor(p + "attn_output.weight"),
                "ffn_norm": self._tensor(p + "ffn_norm.weight"),
                "gate": self._tensor(p + "ffn_gate.weight"),
                "up": self._tensor(p + "ffn_up.weight"),
                "down": self._tensor(p + "ffn_down.weight"),
                # Qwen3
                "q_norm": self._optional_tensor(p + "attn_q_norm.weight"),
                "k_norm": self._optional_tensor(p + "attn_k_norm.weight"),
            })

            #if i == 0:
            #    print("q_norm:", "blk.0.attn_q_norm.weight" in reader.tensors)
            #    print("k_norm:", "blk.0.attn_k_norm.weight" in reader.tensors)


    @staticmethod
    def _resolve_prefix(arch, meta):
        if arch == "llama":
            return "llama"

        if arch.startswith("qwen"):
            if f"{arch}.block_count" in meta:
                return arch
            if "qwen2.block_count" in meta:
                return "qwen2"
            if "qwen3.block_count" in meta:
                return "qwen3"

        if "llama.block_count" in meta:
            return "llama"
        if "qwen2.block_count" in meta:
            return "qwen2"
        if "qwen3.block_count" in meta:
            return "qwen3"
        return None

    def _tensor(self, name):
        return self.load(self.reader, self.reader.tensor(name))

    def _optional_tensor(self, name):
        try:
            return self._tensor(name)
        except KeyError:
            return None

    def reset_cache(self):
        # Clear the cache before starting a fresh prompt.
        self.cache = [KVCache() for _ in range(self.n_layers)]

    def forward(self, token_id, position):
        # Look up the embedding vector for the current token ID.
        x = self.emb[int(token_id)].astype(np.float32)

        for i, layer in enumerate(self.layers):
            # Normalize the hidden state before attention.
            h = rms_norm(x, layer["attn_norm"], self.rms_eps)

            # Run self-attention with the cached history of previous tokens.
            a = attention(
                h,
                layer["wq"],
                layer["bq"],
                layer["wk"],
                layer["bk"],
                layer["wv"],
                layer["bv"],
                layer["wo"],
                position,
                self.cache[i],
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.rope_theta,
                self.rope_mode,
                layer["q_norm"],
                layer["k_norm"],
                self.rms_eps,
            )


            #if i == 0:
            #    print(f"Layer {i}:")
            #    print("Q shape:", layer["wq"].shape)
            #    print("K shape:", layer["wk"].shape)
            #    print("V shape:", layer["wv"].shape)

            #    print("Q norm shape:", None if layer["q_norm"] is None else layer["q_norm"].shape)
            #    print("K norm shape:", None if layer["k_norm"] is None else layer["k_norm"].shape)

            #    print("RMS epsilon:", self.rms_eps)
            #    print("RoPE theta:", self.rope_theta)

            x = x + a

            # Normalize again before the feed-forward network.
            h = rms_norm(x, layer["ffn_norm"], self.rms_eps)
            # SwiGLU is the model's MLP / feed-forward transformation.
            x = x + swiglu(
                h,
                layer["gate"],
                layer["up"],
                layer["down"],
            )

        x = rms_norm(x, self.norm, self.rms_eps)

        # Project the final hidden state to vocabulary-sized logits.
        logits = linear(x, self.output)
        return logits
