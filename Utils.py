import numpy as np


def create_message(n=1000, m=50):
    if n < m:
        n = m

    out = np.arange(0, m)

    pad = np.random.randint(0, m, n - len(out))
    out = np.concatenate([out, pad])
    np.random.shuffle(out)

    return out


def AWGN(n, power=0.01):
    # Create the noise
    n = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)  # AWGN with unity power
    n = n.astype(np.complex64)
    # Scale it
    n = n * np.sqrt(power)

    return n




