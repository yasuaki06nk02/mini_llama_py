import numpy as np
from .sampler import sample


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=32,
    temperature=0.0,
    top_k=40,
    chat=False,
    system=None,
):
    if chat:
        # Turn the plain prompt into the chat format expected by instruction-tuned models.
        prompt = tokenizer.format_chat(prompt, system=system)
        print("formatted prompt:")
        print(repr(prompt))

    # Convert the prompt into token IDs once, then reuse the KV cache during decoding.
    tokens = tokenizer.encode(prompt, add_bos=True)
    print("tokens:", tokens)

    # Start with a clean cache for each new request.
    model.reset_cache()

    # -------------------------
    # Prefill
    # -------------------------
    # Every prompt token is evaluated in sequence so that the KV cache
    # contains the complete prompt context.
    logits = None

    for position, token in enumerate(tokens):
        # Prefill: run the full prompt through the model to populate the cache.
        logits = model.forward(token, position)

        # debug: print the logits for the first token of the prompt.
        if position == len(tokens) - 1:
            print(
                "PREFILL FINAL:",
                "position=", position,
                "token=", token,
                "logits:",
                "min=", float(np.min(logits)),
                "max=", float(np.max(logits)),
                "mean=", float(np.mean(logits)),
                "std=", float(np.std(logits)),
         )

        top = np.argsort(logits)[-10:][::-1]
        print("PREFILL FINAL TOP:", top)

    generated = []

    # -------------------------
    # Decode
    # -------------------------
    # From here only the newly generated token is evaluated.
    for step in range(max_new_tokens):
        # Sample one next token from the current logits.
        token = sample(
            logits,
            temperature=temperature,
            top_k=top_k,
        )

        # Stop as soon as a special end token appears.
        if token in getattr(tokenizer, "stop_ids", set()):
            break

        generated.append(token)

        # Decode: feed the newly generated token back into the model.
        position = len(tokens) + step
        logits = model.forward(token, position)

    # Convert token IDs back to text and remove any trailing special markers.
    text = tokenizer.decode(generated)
    for stop in ("<|im_end|>", "<|endoftext|>"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()
