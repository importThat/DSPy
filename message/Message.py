import numpy as np

"""
A class for handling the input, compression, and encoding of message data. Can read in and encode any file (although the
file is stored in ram so the size may impact performance)
"""

# Tomorrow
# generate LDPC codes
# start working on the decoders


class Message:
    def __init__(self, fn=None):
        self.fn = fn
        self.data = None
        self.codewords = np.array([])

    def read(self):
        # Read in the data
        data = np.fromfile(self.fn, dtype="uint8")
        # Convert to a bit array
        self.data = np.unpackbits(data)

    def create_message(self, n=10):
        # Creates a random binary message
        self.data = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

    def encode(self, G):
        """
        Given a generator matrix, G, encodes the data into codewords
        """
        n = G.shape[0]  # The block size

        # If the data isn't neatly divisible into n chunks then we pad it out
        remainder = self.data.size % n
        if remainder:
            pad = np.zeros([abs((n - remainder)),])
            self.data = np.concatenate([self.data, pad])

        # split the data into N sized chunks
        self.data = self.data.reshape([-1, n])

        # Encode!
        self.codewords = self.data.dot(G) % 2

    def min_hamming(self):
        """
        Computes the minimum hamming distance of the message codewords.
        """
        min_ham = 9999999999999999  # just start with a big number
        unique_codes = np.unique(self.codewords, axis=0)
        cum_sum = []

        for i in range(unique_codes.shape[0]):
            for j in range(i+1, unique_codes.shape[0]):
                ham = np.count_nonzero(unique_codes[i] != unique_codes[j])
                cum_sum.append(ham)

                if ham < min_ham:
                    min_ham = ham

        avg_ham = np.array(cum_sum)
        avg_ham = np.mean(avg_ham)

        return min_ham, avg_ham

    def ldpc_beliefprop(self):
        # Belief propagation for LDPC
        # https://yair-mz.medium.com/decoding-ldpc-codes-with-belief-propagation-43c859f4276d
        pass

    def ldpc_hard(self):
        # Hard decision rule for LDPC
        # https://uweb.engr.arizona.edu/~ece506/readings/project-reading/5-ldpc/LDPC_Intro.pdf
        pass





