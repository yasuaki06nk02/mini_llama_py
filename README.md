# mini-llama.py

`llama.cpp` 系の GGUF モデルを、できるだけ少ないコードで読み解くための学習用 Python 実装です。

このプロジェクトの目的は「速く動かすこと」ではなく、「モデルが推論するときに内部で何が起きているかを追えるようにすること」です。

たとえば、1回の生成はおおまかに次の流れで進みます。

1. GGUF ファイルを開く
2. 重みテンソルと tokenizer 情報を読む
3. 入力文を token ID に変換する
4. Embedding を引く
5. Transformer block を何層も通す
6. 最後の logits から次の token を選ぶ
7. 選んだ token を文字列に戻す

この実装は、その一連の流れを小さなファイルに分けて読めるようにしています。

## What This Project Supports

- GGUF v2 / v3 の読み込み
- F32 / F16
- Q4_0
- Q8_0
- SentencePiece 系 tokenizer
- GGML/GPT-2 系 tokenizer metadata
- Llama 系 Transformer
- Qwen / Qwen2 系 Transformer
- RMSNorm
- RoPE
- Grouped Query Attention
- KV cache
- Prefill / Decode
- Greedy sampling
- Temperature / top-k sampling

## What This Project Does Not Try To Do

これは教育用の最小実装なので、`llama.cpp` のような高速化はしていません。

以下はまだ限定的、または未対応です。

- K-quant の本格対応
- AVX / NEON / CUDA
- Flash Attention
- mmap
- speculative decoding
- GGUF の全 tokenizer 方式
- Llama / Qwen 以外の多くのアーキテクチャ

## Install

Python 3.10 以上を想定しています。

```bash
pip install -r requirements.txt
```

`sentencepiece` は SentencePiece tokenizer を使うモデル向けに必要です。
Qwen 系のように `tokenizer.ggml.tokens` を持つ GGUF では、メタデータから tokenizer を構築します。

## How To Run

基本の実行例:

```bash
python main.py model.gguf "The highest mountain in Japan is"
```

Qwen 系の instruct モデルを試す例:

```bash
python main.py qwen2.5.gguf "日本の首都はどこですか？" --chat --system "You are a helpful assistant."
```

`--chat` は、入力文を Qwen の ChatML 形式に整形してから推論するためのフラグです。
`--system` はその会話の前提となる system message を指定します。

デフォルトでは greedy decoding です。つまり、毎回もっとも確率が高い token を選びます。
必要なら `--temperature` や `--top-k` を変えて、少し多様な出力にもできます。

## Project Structure

### `main.py`

CLI の入口です。
GGUF ファイルを開き、tokenizer と model を作って、`generate()` を呼びます。

### `mini_llama/gguf.py`

GGUF ファイルのヘッダ、メタデータ、tensor 情報を読みます。

### `mini_llama/tensor.py`

GGUF 内の量子化 tensor を NumPy 配列に変換します。

### `mini_llama/tokenizer.py`

入力テキストを token ID に変換し、token ID を文字列に戻します。

Qwen 系では `tokenizer.ggml.tokens` と `tokenizer.ggml.merges` を使うので、`tokenizer.model` が無い GGUF でも動きます。

### `mini_llama/layers.py`

線形層、RMSNorm、SiLU などの基本演算です。

### `mini_llama/rope.py`

RoPE の回転位置埋め込みを担当します。

### `mini_llama/attention.py`

Q/K/V の計算、RoPE 適用、KV cache、attention 重みの計算を行います。

### `mini_llama/ffn.py`

FFN の SwiGLU 部分を実装しています。

### `mini_llama/model.py`

Transformer 全体の forward pass を組み立てます。

### `mini_llama/sampler.py`

logits から次の token を選びます。

### `mini_llama/generate.py`

prefill と decode を分けて、実際の生成ループを回します。

## How The Inference Flow Works

### 1. GGUF is opened

`GGUFReader` がファイルを読み、メタデータと tensor 情報を集めます。
ここにはモデルのアーキテクチャ名、層数、ヘッド数、tokenizer 情報などが入っています。

### 2. Tensors are loaded

`tensor.py` が重みを NumPy 配列として取り出します。
量子化されている場合は、必要に応じて float 配列へ戻します。

### 3. Text becomes token IDs

`tokenizer.py` が prompt を token ID の列に変換します。
Qwen 系では GGUF 内の metadata を使って tokenizer を再構築します。

### 4. Prefill runs first

prompt 全体を 1 token ずつ model に通します。
この段階で KV cache に「入力文の記憶」がたまります。

### 5. Decode runs after that

1 token ずつ次の token を選びます。
毎回、直前までの KV cache を使うので、prompt 全体を毎回再計算しません。

### 6. Sampling chooses the next token

`temperature=0` なら greedy です。
`temperature>0` なら確率的に次 token を選べます。

### 7. Tokens are decoded back to text

選ばれた token ID は `tokenizer.py` で文字列へ戻されます。

## Reading Order

初学者は次の順番で読むと分かりやすいです。

1. `mini_llama/gguf.py`
2. `mini_llama/tensor.py`
3. `mini_llama/tokenizer.py`
4. `mini_llama/layers.py`
5. `mini_llama/rope.py`
6. `mini_llama/attention.py`
7. `mini_llama/ffn.py`
8. `mini_llama/model.py`
9. `mini_llama/sampler.py`
10. `mini_llama/generate.py`
11. `main.py`

## Notes For Qwen Models

Qwen 系の GGUF では、`tokenizer.model` が存在しないことがあります。
その場合は以下の metadata を使います。

- `tokenizer.ggml.tokens`
- `tokenizer.ggml.merges`
- `tokenizer.ggml.model`
- `tokenizer.ggml.pre`
- `tokenizer.chat_template`

また、Qwen の instruct モデルは独自の会話形式を前提に学習されているため、`--chat` を使って ChatML 風の prompt に整形するのが自然です。

## Notes For Beginners

- `GGUF` はモデル本体の入れ物です
- `tokenizer` は文章を token ID に変えます
- `embedding` は token ID をベクトルに変えます
- `attention` は文脈のどこを見るかを決めます
- `FFN` は各 token の表現をさらに加工します
- `logits` は次の token 候補のスコアです
- `sampling` は次に出す token を決めます

## Why This Is Useful

`llama.cpp` の動作を見ているだけだと、内部の流れが一気に進みすぎて追いにくいことがあります。
この実装は速度よりも可読性を優先しているので、学びながら少しずつ改造するのに向いています。

## Example Workflow

1. 小さな GGUF を用意する
2. `python main.py ...` で動かす
3. `tokenizer.py` を見て tokenization を理解する
4. `model.py` を見て forward pass を理解する
5. `attention.py` を見て KV cache と RoPE を理解する
6. 少しずつ改造して挙動を観察する

## Troubleshooting

- 出力が文字化けする場合は、まず tokenizer の実装と GGUF の metadata を確認します
- 応答が途中で止まる場合は、stop token の扱いを見ます
- 数値警告が出る場合は、`layers.py` や `attention.py` の安定性を見ます
- モデルが読めない場合は、GGUF の architecture 名と tensor 名を確認します

