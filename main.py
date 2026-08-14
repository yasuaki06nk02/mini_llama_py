import argparse

from mini_llama.gguf import GGUFReader
from mini_llama.tensor import load_tensor
from mini_llama.tokenizer import Tokenizer
from mini_llama.model import LlamaModel
from mini_llama.generate import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", dest="chat_prompt", action="store_true")
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    reader = GGUFReader(args.model)

    print(f"GGUF version: {reader.version}")
    print(f"tensors: {len(reader.tensors)}")

    tokenizer = Tokenizer(reader)
    model = LlamaModel(reader, load_tensor)

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

    #for k, v in reader.metadata.items():
    #    print(k, "=", v)

    print(text)

    reader.close()


if __name__ == "__main__":
    main()
