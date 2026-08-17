import argparse

from mini_llama.gguf import GGUFReader
from mini_llama.tensor import load_tensor
from mini_llama.tokenizer import Tokenizer
from mini_llama.model import LlamaModel
from mini_llama.generate import generate
from mini_llama.chat_loop import chat_loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")

    # 通常モードでは prompt を指定する。
    # --chat の場合は省略可能。
    parser.add_argument("prompt", nargs="?", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)

    # 1回だけチャットテンプレートを使って生成
    parser.add_argument(
        "--prompt",
        dest="chat_prompt",
        action="store_true",
    )

    # 対話モード
    parser.add_argument(
        "--chat",
        action="store_true",
        help="interactive chat mode",
    )

    parser.add_argument("--system", default=None)

    args = parser.parse_args()

    if not args.chat and args.prompt is None:
        parser.error("prompt is required unless --chat is specified")

    reader = GGUFReader(args.model)

    print(f"GGUF version: {reader.version}")
    print(f"tensors: {len(reader.tensors)}")

    tokenizer = Tokenizer(reader)
    model = LlamaModel(reader, load_tensor)

    try:
        if args.chat:
            chat_loop(
                model,
                tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                system=args.system,
            )
        else:
            text = generate(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                chat=args.chat_prompt,
                system=args.system,
            )

            print(text)

    finally:
        reader.close()


if __name__ == "__main__":
    main()
