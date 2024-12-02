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


def moving_average(x, n, weights=None):
    if weights is None:
        window = np.ones(n)
    else:
        window = np.array(weights)

    return np.convolve(x, window, 'valid') / n


def markify(symbols):
    """
    Given some symbols returns an array of the pattern of the symbol occurences
    """
    index = np.arange(len(symbols))
    output = None

    for i in range(len(symbols)):
        marker = index[symbols == symbols[i]]
        if output is None:
            output = marker
        else:
            output = np.concatenate([output, marker])

    return output


def create_wave(t, f, amp, phase):
    angle = 2 * np.pi * f * t + phase
    wave = amp * np.cos(angle) + 1j * amp * np.sin(angle)

    return wave.astype(np.complex64)


