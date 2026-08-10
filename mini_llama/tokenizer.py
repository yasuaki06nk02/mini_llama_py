import re


class Tokenizer:
    def __init__(self, reader):
        self.reader = reader
        self.chat_template = None
        # Qwen GGUF models often store tokenizer behavior in metadata instead of tokenizer.model.
        self.add_bos = bool(reader.metadata.get("tokenizer.ggml.add_bos_token", True))
        self.add_eos = bool(reader.metadata.get("tokenizer.ggml.add_eos_token", False))

        if reader.metadata.get("tokenizer.ggml.tokens") is not None:
            self._init_ggml(reader)
        else:
            self._init_sentencepiece(reader)

    def _init_sentencepiece(self, reader):
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise RuntimeError(
                "sentencepiece is required for SentencePiece-based GGUF models: pip install sentencepiece"
            ) from e

        model = reader.metadata.get("tokenizer.model")
        if model is None:
            raise ValueError("GGUF does not contain tokenizer.model or tokenizer.ggml.tokens")

        self.sp = spm.SentencePieceProcessor()
        ok = self.sp.LoadFromSerializedProto(bytes(model))
        if not ok:
            raise RuntimeError("failed to load tokenizer.model")

        self.bos_id = self._meta_int(reader, "tokenizer.ggml.bos_token_id", self.sp.bos_id())
        self.eos_id = self._meta_int(reader, "tokenizer.ggml.eos_token_id", self.sp.eos_id())
        self.unknown_id = self._meta_int(reader, "tokenizer.ggml.unknown_token_id", self.sp.unk_id())
        self.stop_ids = {x for x in [self.eos_id] if x >= 0}
        template = reader.metadata.get("tokenizer.chat_template")
        self.chat_template = str(template) if template is not None else None
        self.mode = "sentencepiece"

    def _init_ggml(self, reader):
        # Build a byte-level BPE tokenizer from GGUF metadata.
        self.tokens = [str(x) for x in reader.metadata["tokenizer.ggml.tokens"]]
        self.merges = [str(x) for x in reader.metadata.get("tokenizer.ggml.merges", [])]
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.merge_ranks = {}
        for rank, merge in enumerate(self.merges):
            left, right = merge.split(" ", 1)
            self.merge_ranks[(left, right)] = rank

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.bos_id = self._meta_int(reader, "tokenizer.ggml.bos_token_id", -1)
        self.eos_id = self._meta_int(reader, "tokenizer.ggml.eos_token_id", -1)
        self.unknown_id = self._meta_int(reader, "tokenizer.ggml.unknown_token_id", -1)
        self.stop_ids = {x for x in [self.bos_id, self.eos_id] if x >= 0}
        template = reader.metadata.get("tokenizer.chat_template")
        self.chat_template = str(template) if template is not None else None
        self.mode = "ggml"

    @staticmethod
    def _meta_int(reader, key, default):
        value = reader.metadata.get(key)
        return int(value) if value is not None else int(default)

    @staticmethod
    def _bytes_to_unicode():
        # Match the byte-to-unicode mapping used by GPT-2 style tokenizers.
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    @staticmethod
    def _get_pairs(word):
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _bpe(self, token):
        # Merge token pairs according to the rank table until no better merge exists.
        word = tuple(token)
        if len(word) <= 1:
            return [token]

        pairs = self._get_pairs(word)
        while pairs:
            best = None
            best_rank = None
            for pair in pairs:
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best = pair
                    best_rank = rank
            if best is None:
                break

            first, second = best
            merged = first + second
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) <= 1:
                break
            pairs = self._get_pairs(word)
        return list(word)

    def _encode_piece(self, piece):
        # Convert text to UTF-8 bytes and then to printable unicode symbols.
        encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
        if encoded in self.token_to_id:
            return [self.token_to_id[encoded]]
        return [self.token_to_id[p] for p in self._bpe(encoded) if p in self.token_to_id]

    def encode(self, text, add_bos=True):
        # Convert input text into token IDs before the model sees it.
        if self.mode == "sentencepiece":
            ids = list(self.sp.EncodeAsIds(text))
            if add_bos and self.bos_id >= 0:
                ids.insert(0, self.bos_id)
            return ids

        pieces = re.findall(r"\s+|\S+", text)
        ids = []
        for piece in pieces:
            ids.extend(self._encode_piece(piece))
        if add_bos and self.add_bos and self.bos_id >= 0:
            ids.insert(0, self.bos_id)
        if self.add_eos and self.eos_id >= 0:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        # Convert generated token IDs back into human-readable text.
        if self.mode == "sentencepiece":
            return self.sp.DecodeIds([int(x) for x in ids])

        text = "".join(self.tokens[int(x)] for x in ids if 0 <= int(x) < len(self.tokens))
        byte_values = bytearray()
        for ch in text:
            if ch in self.byte_decoder:
                byte_values.append(self.byte_decoder[ch])
        return byte_values.decode("utf-8", errors="replace")

    def format_chat(self, prompt, system=None):
        # Wrap the raw prompt in Qwen-style ChatML so the model sees the expected format.
        system_text = system or "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return (
            f"<|im_start|>system\n{system_text}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
