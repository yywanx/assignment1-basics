import math


def get_d_ff(d_model: int, base: int = 64) -> int:
    raw_d_ff = int(8 * d_model / 3)
    d_ff = base * math.ceil(raw_d_ff / base)
    return d_ff