"""
Encode theory.

The encoder handles mapping a bit vector of length K, called s, to a codeword, called x, which will be of length N > K,
This transform is written s.G = x, where G is some generator matrix that creates the codeword.

For a code with parity check matrix H, whose codewords satisfy Hx = 0, the generator must satisfy H.G_transposed = 0

H is systematic when written in the form, H = [-Pt | I], where P is the parity matrix and I is the identity matrix
(and -Pt = Pt for binary). A systematic matrix is useful because its output will contain the original code

The systematic form of G is then given by G = [I | P]

Because we know I already, whenever we create the matrices we just need to find P

A further property of these matrices is that we want them to be linearly independent. This means that any two columns
added to each other cannot be equal to zero. We can use this property when generating matrices. Indeed the number of
columns that add to zero will be the overall hamming distance of the matrix!
"""

import numpy as np


def hamming(m, n):
    """
    Returns H and G matricies for a m, n hamming code. Takes in a message of n bits and puts out a message of m bits,
    adding m-n bits of error correction
    """
    # TODO
    #  currently only works for 7, 4... make it more general!
    if m <= n:
        raise ValueError("m must be greater than n")
    if n < 2:
        raise ValueError("n must be >= 2")

    # Create H, the decoder matrix. The columns of H will be formed by the non-zero binary tuples of 2**(m-n) - 1
    H = []
    for i in range(1, (2**(m-n))):
        out = bin(i)[2:].zfill((n-1)) # ensure it is n bits long by padding with 0s
        H.append([int(i) for i in out])
    H = np.array(H)
    H = H.T

    # That gives us a non-systematic H. We know that a systematic H = [-Pt | I]. I is made of all the entries with
    # only 1 number per column, so we can just get rid of those columns and we'll be left with -Pt!

    col_sum = np.sum(H, axis=0)
    # Find all the rows with with more than one 1 in them, this is everything but the identity matrix (H = [-Pt | I])
    col_mask = col_sum > 1
    # Remove I, H = [-Pt]
    Pt = H[:, col_mask]

    # We can now get P, which we will use to make G from G = [I | P]
    P = Pt.T

    # So now we have the parity check matrix. If the hamming code is of a form where k == (2**n)-1 (I.e the current size
    # of P), then we don't need to do anything else. However if k is bigger or smaller we need to change P
    change = (m - n) - P.shape[1]
    if change > 0:
        # Pick two columns at random and make a third that is the linear combination of them
        # Note that this means that it is likely that not all columns will be linearly independent (which might be bad?)
        for i in range(change):
            cols = np.random.choice(P.shape[1], 2, replace=False)
            new_col = P[:, cols].sum(axis=1)%2  # select the two columns and modulo two add them
            new_col = new_col.reshape([-1, 1])  # Reshape so we can concatenate
            P = np.concatenate([P, new_col], axis=1)    # add the new column in

    elif change < 0:
        # We need to prune P back
        P = P[:, 0:change]  # Change is negative here, so will take the 0 to change'th column

    # Now we just need to make a fitting I and attach it to the P
    I = np.identity(P.shape[0], dtype="int8")
    G = np.concatenate([I, P], axis=1)

    Ih = np.identity(P.T.shape[0], dtype='int8')
    H = np.concatenate([P.T, Ih], axis=1)

    assert (H.dot(G.T) % 2).sum() == 0, f"H.GT != 0 for H {H}, G {G}"

    return H.T, G


def ldpc_parity_matrix(n, j, k):
    """
    Creates the Parity portion of an LDPC decoding matrix using Gallager's method
    Gallager's paper - https://www.inference.org.uk/mackay/gallager/papers/ldpc.pdf
    The method is found in section 2.2

    The parity portion is the Pt from H = [I, Pt]

    Code adapted from https://github.com/hichamjanati/pyldpc/blob/master/pyldpc/code.py

    n: int, Number of columns in the generator matrix

    j: int, Number of ones in the columns of the matrix, must be greater than 2

    k: int, Number of ones in the rows of the matrix, must be greater than j and must divide n

    return H: Matrix, the LDPC decoding matrix. It will be of size (n*j//k, n)
    """
    if j <= 1:
        raise ValueError("j must be 2 or more")
    if k <= j:
        raise ValueError("k must be greater than j")
    if n % k :
        raise ValueError("n must be an integer multiple of k")

    # Number of rows in the matrix, this is also the block length that must be passed in
    rows = n * j // k

    starting_block = np.zeros((rows // j, n), dtype=int)
    Pt = starting_block

    # We need to turn k consecutive bits to 1 in each row
    for i in range(starting_block.shape[0]):
        starting_block[i, i*k:i*k+k] = 1

    # Now we construct H by taking random permutations of those columns
    for i in range(j-1):
        # take column wise permutations of the starting_block matrix (i.e create a new matrix by randomly selecting
        # columns)
        permutation = np.random.permutation(starting_block.T).T
        Pt = np.concatenate([Pt, permutation])

    # Check to make sure it's correct!
    assert np.all(np.sum(Pt, axis=1) == k), (f"H is not correct, k is {k} but column wise sum is"
                                            f" {np.sum(Pt, axis=1)}")
    assert np.all(np.sum(Pt, axis=0) == j), (f"H is not correct, j is {j} but row wise sum is"
                                            f" {np.sum(Pt, axis=0)}")
    return Pt


def ldpc(n, j, k):
    """
    Returns the decoder (H) and generator (G) matrices for an LDPC code. H is a matrix comprised of [I, Pt] (for the
    binary case) and G is [I, P]. See the ldpc_parity_matrix function for more information

    n: int, Number of columns in the generator matrix

    j: int, Number of ones in the columns of the matrix, must be greater than 2

    k: int, Number of ones in the rows of the matrix, must be greater than j and must divide n
    """
    Pt = ldpc_parity_matrix(n, j, k)
    P = Pt.T

    I = np.identity(P.shape[0], dtype="int8")
    G = np.concatenate([I, P], axis=1)

    Ih = np.identity(P.T.shape[0], dtype='int8')
    H = np.concatenate([P.T, Ih], axis=1)

    return H.T, G


def crc(data: np.ndarray, polynomial: np.ndarray | str = "32"):
    """
    cyclic redundancy check

    polynomial accepts either string arguments aligning with popular CRC algorithms, or an np array of bits that defines
    a polynomial, e.g np.array([1,1,0,1])

    The known CRCs are just pulled from the wiki -
    https://en.wikipedia.org/wiki/Cyclic_redundancy_check#Standards_and_common_use

    """

    # coded CRCs
    crcs = {"1": np.array([1, 1]), # Parity bit
            "3": np.array([1, 0, 1, 1]),   # GSM, mobile networks
            "4": np.array([1, 0, 0, 1, 1]),
            "5": np.array([1, 0, 1, 0, 0, 1]),  # GEN 2 RFID
            "6A": np.array([1, 1, 0, 0, 1, 1, 1]),  # CDMA2000-A, used in mobile networks
            "6GSM": np.array([1, 1, 0, 1, 1, 1, 1]),    # GSM, used in mobile networks
            "7": np.array([1, 0, 0, 0, 1, 0, 0, 1]),    # Used in telcoms apparently
            "8": np.array([1, 1, 1, 0, 1, 0, 1, 0, 1]), # DVB-S2, satellite tv
            "8DARC": np.array([1, 0, 0, 1, 1, 1, 0, 0, 1]), # Data radio channel apparently
            "10GSM": np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]), # Mobile networks
            "11": np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]),    # Flexray (used in automotive)
            "12": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]),    # T.v apparently
            "13": np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]), # Radio teleswitch, used to switch electricity
                                                                        # meter rates
            "14": np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]),  # Another data radio channel
            "15": np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),   # CAN, used to control ECUs
            "16": np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),

            "32": np.array([0, 0, 0, 0, 0, 1, 0, 0,
                            1, 1, 0, 0, 0, 0, 0, 1,
                            0, 0, 0, 1, 1, 1, 0, 1,
                            1, 0, 1, 1, 0, 1, 1, 1])
            }

    if isinstance(polynomial, str):
        try:
            poly = crcs[polynomial]
        except KeyError as e:
            print(f"{e}: Polynomial must be an array or a string index of the polynomial dictionary")

    elif isinstance(polynomial, np.ndarray):
        poly = polynomial

    else:
        raise ValueError("Polynomial must be an array or a string index of the polynomial dictionary")

    out = data.copy()

    # If the message  is just a 1d vector we can reshape it to a 1,n matrix so that it can take matrix operations
    if len(out.shape) == 1:
        out = out.reshape([1, -1])

    len_p = len(poly)
    len_block = out.shape[1]

    out = np.concatenate([out, np.zeros(shape=(out.shape[0], len_p-1))], axis=1)
    out = out.astype(np.uint8)

    for row in out:
        index= np.where(row==1)[0][0]

        while row[0:len_block].sum() > 0:
            row[index:index+len_p] = row[index:index+len_p] ^ poly

            if row.sum() > 0:
                index = np.where(row == 1)[0][0]

    return out[:, len_block:]


def BCH():
    """"""
    pass


def RS():
    """"""
    pass


def golay():
    """"""
    pass


