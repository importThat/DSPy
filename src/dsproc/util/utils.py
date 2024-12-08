"""
Utility functions for other dsproc classes
"""
import numpy as np


def create_message(n: int = 1000, m: int = 50) -> np.ndarray:
    """
    Randomly generates a numpy array of integers of length n with m unique values.
    :param n: The length of the message
    :param m: How many unique values there are in the message
    :return: A np array of values
    """
    n = max(n, m)

    out = np.arange(0, m)

    pad = np.random.randint(0, m, n - len(out))
    out = np.concatenate([out, pad])
    np.random.shuffle(out)

    return out


def AWGN(n: int, power: float = 0.01) -> np.ndarray:
    """
    Returns a np array of complex additive white gaussian noise of length n. Use the power argument to scale the
    amplitude of the noise
    :param n: The length of the output noise
    :param power: Scaling factor of the power
    :return: A np array of complex 64 floats with length n
    """
    # Create the noise
    n = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)  # AWGN with unity power
    n = n.astype(np.complex64)
    # Scale it
    n = n * np.sqrt(power)
    # Make sure its 64 bit
    n = n.astype(np.complex64)

    return n


def moving_average(x: np.ndarray, n: int, weights: np.ndarray = None) -> np.ndarray:
    """
    Runs a moving average of length n over the data. Can use custom weights or otherwise the weights are set to one.
    Custom weights must be a list or np array of the same length as n
    :param x: np array of your input data (1d)
    :param n: the length of the averaging window
    :param weights: np array of floats, the moving average weights
    """
    if weights is None:
        window = np.ones(n)
    else:
        window = np.array(weights)

    return np.convolve(x, window, 'valid') / n


def markify(symbols):
    """
    Given some symbols returns an array of the pattern of the symbol occurrences

    :param symbols: An array of ints
    :return: Array of occurrences of those symbols
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


def create_wave(t, f, amp, phase) -> np.ndarray:
    """
    Creates a complex wave from the given paramaters.
    :param t: A np array of times
    :param f: frequency, int
    :param amp: amplitude, float
    :param phase: phase in radians
    :return: Array of complex 64 floats with length same as t
    """
    angle = 2 * np.pi * f * t + phase
    wave = amp * np.cos(angle) + 1j * amp * np.sin(angle)

    return wave.astype(np.complex64)


