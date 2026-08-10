import numpy as np


def greedy(logits):
    return int(np.argmax(logits))


def sample(logits, temperature=0.8, top_k=40, rng=None):
    if temperature <= 0:
        return greedy(logits)

    rng = rng or np.random.default_rng()

    x = logits.astype(np.float64) / temperature

    if top_k > 0 and top_k < len(x):
        idx = np.argpartition(x, -top_k)[-top_k:]
        values = x[idx]
        values -= np.max(values)
        probs = np.exp(values)
        probs /= probs.sum()
        return int(rng.choice(idx, p=probs))

    x -= np.max(x)
    probs = np.exp(x)
    probs /= probs.sum()

    return int(rng.choice(len(x), p=probs))
