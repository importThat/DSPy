import numpy as np


def create_message(n=1000, m=50):
    if n < m:
        n = m

    out = np.arange(0, m)
    pad = np.random.randint(0, m - 1, n - len(out))
    out = np.concatenate([out, pad])
    np.random.shuffle(out)

    return out


