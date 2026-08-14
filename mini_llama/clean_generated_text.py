def clean_generated_text(text):
    """
    Qwen3の生成結果からthinking部分や
    chat用特殊トークンを除去する。
    """

    # thinking部分を除去
    if "<think>" in text:

        if "</think>" in text:
            text = text.split(
                "</think>",
                1,
            )[1]

        else:
            # thinking中にmax_new_tokensへ
            # 到達した場合
            text = ""

    # Chat特殊トークンを除去
    for stop in (
        "<|im_end|>",
        "<|endoftext|>",
    ):
        idx = text.find(stop)

        if idx != -1:
            text = text[:idx]

    return text.strip()