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


def find_nonzero_row(matrix, start_row, col):
    """
    Finds the first non-zero entry in the given column, starting from the start_row and working down
    """
    for row in range(start_row, matrix.shape[0]):
        if matrix[row, col] != 0:
            return row

    return None


def swap_rows(matrix, row1, row2):
    """
    swaps the two rows in a matrix
    """
    matrix[[row1, row2]] = matrix[[row2, row1]]


def binary_elimination(matrix, pivot_row):
    """
    Adds the pivot row to the ones below it in the matrix to eliminate any non-zero elements (through modulo 2 addition)
    """
    for row in range(pivot_row+1, matrix.shape[0]):
        matrix[row] += matrix[pivot_row]
        matrix[row] = matrix[row] % 2


def rre(matrix):
    """
    Computes the binary reduced row echelon of the given binary matrix
    """
    out = matrix.copy()

    pivot_row = 0
    cols = out.shape[1]

    for col in range(cols):
        nonzero_row = find_nonzero_row(out, pivot_row, col)

        if nonzero_row:
            swap_rows(out, pivot_row, nonzero_row)
            binary_elimination(out, pivot_row)
            pivot_row += 1

    return out


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


