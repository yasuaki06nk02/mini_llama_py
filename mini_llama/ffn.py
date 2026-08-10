from .layers import linear, rms_norm, silu


def swiglu(x, w_gate, w_up, w_down):
    gate = linear(x, w_gate)
    up = linear(x, w_up)
    hidden = silu(gate) * up
    return linear(hidden, w_down)
