from .sampler import sample
from .clean_generated_text import clean_generated_text

def chat_loop(
    model,
    tokenizer,
    max_new_tokens=128,
    temperature=0.0,
    top_k=40,
    system=None,
):
    """
    Interactive chat mode.

    会話履歴をテキストとして保持し、
    各ターンでチャットテンプレート全体を
    Prefillし直す。
    """

    print()
    print("================================")
    print(" Mini Llama Chat")
    print("================================")
    print("Type 'exit' or 'quit' to exit.")
    print()

    history = []

    system_text = system or (
        "You are Qwen, created by Alibaba Cloud. "
        "You are a helpful assistant."
    )

    while True:

        try:
            user_text = input("User: ")

        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_text = user_text.strip()

        if not user_text:
            continue

        if user_text.lower() in (
            "exit",
            "quit",
        ):
            break

        history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        messages = [
            {
                "role": "system",
                "content": system_text,
            }
        ]

        messages.extend(history)

        # Qwen3 chat templateをレンダリング
        prompt = tokenizer.format_chat_messages(
            messages
        )

        # --------------------------------
        # 1ターン生成
        # --------------------------------
        tokens = tokenizer.encode(
            prompt,
            add_bos=True,
        )

        model.reset_cache()

        logits = None

        for position, token in enumerate(tokens):
            logits = model.forward(
                token,
                position,
            )

        generated = []

        for step in range(max_new_tokens):

            token = sample(
                logits,
                temperature=temperature,
                top_k=top_k,
            )

            if token in getattr(
                tokenizer,
                "stop_ids",
                set(),
            ):
                break

            generated.append(token)

            position = len(tokens) + step

            logits = model.forward(
                token,
                position,
            )

        raw_text = tokenizer.decode(
            generated
        )

        print()
        print("Assistant:", end=" ")

        answer = clean_generated_text(
            raw_text
        )

        print(answer)
        print()

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )