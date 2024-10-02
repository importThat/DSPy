import numpy as np
from collections import Counter
from collections import namedtuple
from heapq import heapify, heappop, heappush
from .encode import *

"""
A class for handling the input, compression, and encoding of message data. Can read in and encode any file (although the
file is stored in ram so the size may impact performance)
"""


class Message:
    def __init__(self, fn=None, data=None):
        self.pseudo_rand_sequence = None
        self.fn = fn
        self.data = data
        self.codewords = np.array([])

        # Compression stuff
        self.compression_codes = None
        self.decompression_codes = None
        self.preamble = None
        self.padding = 0 # Tracker of how many bits have been padded

        if self.fn:
            self.read()

        self.lfsr_lookup = {
            "2": (2, 1),
            "3": (3, 2),
            "4": (4, 3),
            "5": (5, 3),
            "6": (6, 5),
            "7": (7, 6),
            "8": (8, 6, 5, 4),
            "9": (9, 5),
            "10": (10, 7),
            "11": (11, 9),
            "12": (12, 11, 10, 4),
            "13": (13, 12, 11, 8),
            "14": (14, 13, 12, 2),
            "15": (15, 14),
            "16": (16, 15, 13, 4),
            "17": (17, 14),
            "18": (18, 11),
            "19": (19, 18, 17, 14),
            "20": (20, 17),
            "21": (21, 19),
            "22": (22, 21),
            "23": (23, 18),
            "24": (24, 23, 22, 17)
        }

    def read(self):
        # Read in the data
        data = np.fromfile(self.fn, dtype="uint8")
        # Convert to a bit array
        self.data = np.unpackbits(data)

    def create_message(self, n=10):
        # Creates a random binary message
        self.data = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

    def apply_encode(self, G):
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

    def symbolise(self, bits_per_symbol):
        """
        Converts a message of bits to integer symbols, pads with 0s if the data isn't an integer number of bits per
        symbol
        :return:
        """
        # If the data isn't an integer number of bits per symbol long
        overflow = len(self.data) % bits_per_symbol
        if overflow:
            # Pad the data with the remainder
            pad_amount = bits_per_symbol - overflow
            pad = np.zeros(pad_amount)
            self.data = np.concatenate([self.data, pad])
            self.padding += pad_amount

        self.data = self.data.reshape([-1, bits_per_symbol])

        symbol_matrix = np.arange(bits_per_symbol)
        symbol_matrix = 2**symbol_matrix
        symbol_matrix = symbol_matrix[::-1] # Reverse it

        symbol_matrix = symbol_matrix.reshape([bits_per_symbol, -1])

        self.data = self.data.dot(symbol_matrix)
        self.data = self.data.flatten()

    def desymbolise(self, bits_per_symbol):
        """
        Takes in symbols and turns them into bits
        :param bits_per_symbol:
        :return:
        """
        symbol_matrix = np.arange(2**bits_per_symbol)

        # A function in a function? I know, I'm an utter savage
        def binify(symbol, bits_per_symbol):
            x = bin(symbol)[2:].zfill(bits_per_symbol)
            return list(x)

        bins = np.array([binify(i, bits_per_symbol) for i in symbol_matrix])
        bins = bins.astype(np.uint8).flatten()
        bins = bins.reshape([2**bits_per_symbol, -1])

        self.data = bins[self.data]
        self.data = self.data.flatten()

    def encode(self, encoder="crc", blocksize=64, decode=False, **kwargs):
        """
        Handler for the encoder functions
        :param encoder:
        :param blocksize:
        :param kwargs:
        :return:
        """
        # Block the message up
        self.data = self.data.reshape([-1, blocksize])

        # apply encodings
        if encoder == "crc":
            crc_checks = crc(self.data, polynomial=kwargs['polynomial'])
            if decode:
                return crc_checks
            else:
                self.data = np.concatenate([self.data, crc_checks], axis=1)

        elif encoder == "ldpc":
            H, G = ldpc(kwargs['n'], kwargs['j'], kwargs['k'])
            if decode:
                return H
            else:
                self.apply_encode(G)

        elif encoder == "hamming":
            H, G = hamming(kwargs['m'], kwargs['n'])
            if decode:
                return H
            else:
                self.apply_encode(G)

        else:
            raise ValueError("Unknown encoder")

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

    def huffman_compress(self, n=8):
        """
        compress the data using the huffman coding algorithm. Compresses the data using bit strings of length n
        (computing complexity grows exponentially with n)

        n must be 128 or less, as there are 8 bits assigned to it in the communication of the compression dictionary
        """

        if n > 128:
            raise ValueError("N must be < 128")

        # Squish message to a string
        message = ''.join([str(i) for i in self.data])

        # Pad data to an integer length of n
        if len(message) % n:
            pad_amount = n - (len(message) % n)
            message += pad_amount * '0'

        # Count up the occurrences of the chunks in the message
        c = Counter([message[i:i+n] for i in range(0, len(message), n)])

        # Create the Node named tuple type which will be used to form the nodes of the tree
        Node = namedtuple('Node', ['count', 'letter', 'left', 'right'])

        # Create all the nodes
        nodes = [Node(count, letter, None, None) for (letter, count) in c.items()]

        # Turn the nodes into a queue
        heap = nodes.copy()
        heapify(heap)

        # Iterate through the queue, continually adding the two smallest elements of the queue to a new node
        while len(heap) > 1:
            left_child = heappop(heap)
            right_child = heappop(heap)
            new_count = left_child.count + right_child.count
            merged_node = Node(new_count, '\0', left_child, right_child)

            heappush(heap, merged_node)

        # Create the codes
        self.compression_codes = self.generate_huffman_codes(heap[0])

        # Apply the compression
        self.apply_compression(n=n)

    def apply_compression(self, n=8):
        """
        Applies the precomputed compression codes to the stored data
        """
        if n > 128:
            raise ValueError("N must be < 128")
        message = ''.join([str(i) for i in self.data])
        message_compress = ""
        codes = [message[i:i+n] for i in range(0, len(message), n)]

        for code in codes:
            message_compress += self.compression_codes[code]

        self.data = np.array([i for i in message_compress], dtype=np.uint8)

    def block_interleave(self, n=8, deinterleave=False):
        """
        Interleaves the data using a block system
        :param n: The number of columns in the block, the number of rows are inferred from the data
        :return:
        """
        # Firstly, flatten the message
        self.data = self.data.flatten()

        if (self.data.shape[0] % n) != 0:
            pad = n - (self.data.shape[0] % n)
            padding = np.ones(pad) * 9     # Make it a weird number
            self.data = np.concatenate([self.data, padding], dtype=np.int8)

        # reshape
        if deinterleave:
            self.data = self.data.reshape([n, -1])
        else:
            self.data = self.data.reshape([-1, n])
        self.data = self.data.T
        self.data = self.data.flatten()

        # remove any padding
        self.data = self.data[self.data != 9]

    def apply_decompression(self):
        """
        Decompresses the data using the known decompression codes
        """
        message = ''.join([str(i) for i in self.data])
        message_decompress = ""
        all_keys = self.decompression_codes.keys()
        max_len = map(len, all_keys)
        max_len = max(list(max_len))

        i=0

        while i < len(message):
            for j in range(max_len+1):
                test_code = message[i:i+j]

                if test_code in all_keys:
                    message_decompress += self.decompression_codes[test_code]
                    i += j
                    break

                if j == max_len:
                    raise RuntimeError(f"Failed to find decompression codeword in data i is {i} code is {test_code}")

        return message_decompress

    def generate_huffman_codes(self, node, code="", huffman_codes={}):
        """
        Recursively generate the huffman codes for any sub nodes of a parent node
        :param node:
        :param code:
        :param huffman_codes:
        :return:
        """
        # TODO
        #  split this compression step out into another module
        if node is not None:
            if node.letter != "\0":
                huffman_codes[node.letter] = code

            self.generate_huffman_codes(node.left, code + "0", huffman_codes)
            self.generate_huffman_codes(node.right, code + "1", huffman_codes)

        return huffman_codes

    def encode_compression_dict(self):
        """
        Encodes the compression dict and prepends it to the message. This is the information that the receiver will
        then use to de-compress the data on the other side

        Schema
        The codes are not fixed length, but the bits of data they compress are. So, we send the chunk size of the data
        that was compressed, as well as the number of each codes of each length (all in binary and padded to byte sized
        chunks). There also needs to be a tracker of how many bits we have padded the message by

        A binary example would be:
            00000000-11111110-00001000-00000001-00000001-00000010-00000010-00000011-00000100
            which means 256 bits of padding, 8 bit chunks, 1 1 bit code word, 2 2 bit code words, 4 3 bit code words.
            We then send the codes and codewords in order:
            0-01010111-01-11010101-11-01010101-111-10101111 etc.
            Where the 0 is the first code and that corresponds to 01010111 and so on.

        Currently the maximum chunk size is 254, the maximum code word size is 254 and the maximum number of code
        words of any size is 254, maximum padding is 512
        """
        n = len(list(self.compression_codes.keys())[0])

        if n > 254:
            raise ValueError("Compression blocks cannot be > 254")

        codes_chunks = zip(self.compression_codes.values(), self.compression_codes.keys())
        # Tuples of the codes and the chunks
        z = [i for i in codes_chunks]
        # Sort by length of code
        z.sort(key=lambda z: len(z[0]))
        # Make it an array for ease of use
        z = np.array(z)

        # For the header we need to figure out how many codewords there are of each length
        c = Counter([len(i) for i in z[:, 0]])

        # The padding, starts as all 0s and is overwritten at the final step
        header = bin(0)[2:].zfill(16)
        # the number of bits per chunk
        header += bin(n)[2:].zfill(8)
        # The number of codewords of each length
        for length, num in c.items():
            if length > 254 or num > 254:
                raise ValueError("the maximum code word size is 254 and the maximum number"
                                 " of codewords of any size is 254")

            header += bin(length)[2:].zfill(8)
            header += bin(num)[2:].zfill(8)

        # Add a byte of ones to separate the data types
        header += '11111111'

        # Now we need to encode the code dict
        header += ''.join([''.join(i) for i in z])    # The ol' double join!
        # Save the header
        self.preamble = np.array([i for i in header], dtype=np.uint8)

    def decode_preamble(self):
        """
        Removes and interprets the preamble from a received message
        """
        if self.data.ndim > 1:
            self.data = self.data.flatten()

        byte_chunks = []
        flag_index = 0
        for i in range(0, len(self.data), 8):
            chunk = self.data[i:i+8]
            if np.all(chunk == 1) and i > 16: # if it's the all 1s flag
                flag_index = i+8
                break
            else:
                byte_chunks.append(''.join([str(i) for i in chunk]))

        padding = ''.join(byte_chunks[0:2])
        byte_chunks = byte_chunks[2:]
        padding = int(padding, 2)

        n = int(byte_chunks.pop(0), 2)
        # Pull the code length and the number of codes out of the preamble
        codes = [(int(byte_chunks[i], 2), int(byte_chunks[i + 1], 2)) for i in range(0, len(byte_chunks), 2)]

        # now we know how many codes of each length there are
        code_dict = {}
        for length, num in codes:
            for i in range(num):
                code = self.data[flag_index:flag_index+length]
                value = self.data[flag_index+length:flag_index+length+n]

                code = ''.join([str(i) for i in code])
                value = ''.join([str(i) for i in value])

                code_dict[code] = value
                flag_index += length + n

        self.decompression_codes = code_dict
        self.data = self.data[flag_index:]  # cut the preamble off the start and the padding off the end
        if padding:
            self.data = self.data[:-padding]

    def pack_message(self, blocksize):
        """
        Packs up the message for the encoder. First pads the message and then creates the preamble and packs it with
        the header. Returns the message to be sent
        """
        # Flatten the data if it's currently a matrix
        if self.data.ndim > 1:
            self.data = self.data.flatten()

        if self.compression_codes:
            # Create the preamble protocol that communicates the compression dictionary
            self.encode_compression_dict()
        else:
            raise RuntimeError("Transmission of Uncompressed data is not supported")

        # Add the preamble in
        self.data = np.concatenate([self.preamble, self.data])

        # Find how much padding we need and add it in
        if len(self.data) % blocksize:
            pad = blocksize - (len(self.data) % blocksize)
            padding = np.ones(pad, dtype=np.uint8)
            self.data = np.concatenate([self.data, padding])

            # Overwrite the padding bits in the header
            pad_bits = bin(pad)[2:].zfill(16)
            pad_bits = [i for i in pad_bits]
            self.data[0:16] = pad_bits

    def LFSR(self, n, taps=None):
        """
        Generates a recursive sequence using a linear feedback shift register. Utilises a fast bit shifting algorithm
        as seen here - https://en.wikipedia.org/wiki/Linear-feedback_shift_register

        Initial fill is all ones

        n: the length of the register

        taps: an iterable containing the tap positions. Tap positions must be unique and less than or equal to n. Tap
        positions cannot be 0.
        """
        if not taps:
            taps = self.lfsr_lookup[str(n)]

        start_state = (2 ** n) - 1
        lfsr = start_state
        random_code = bin(start_state)[2:]

        # Go for the maximal iterations I guess
        for k in range((2 ** n) - 1):

            # Get the first state, this will then be iterated upon for every tap
            start = lfsr >> (n - taps[0])
            bit = lfsr

            for j in range(1, len(taps)):
                bit = (bit >> (n - taps[j])) ^ start

            bit = bit & 1
            lfsr = (lfsr >> 1) | (bit << (n - 1))

            if lfsr == start_state:
                if k < (2 ** n) - 2:
                    print("non-maximal LFSR detected")
                break
            else:

                random_code += f'{lfsr:b}'.zfill(n)

        self.pseudo_rand_sequence = np.array([i for i in random_code], dtype=np.uint8)

        return self.pseudo_rand_sequence

    def additive_scramble(self, n, taps):
        """
        XoRs the message data with the given linear recursive sequence. Initial fill is all ones.
        See https://en.wikipedia.org/wiki/Linear-feedback_shift_register for typical n and tap values.
        examples:
            x^4 + x^3 + 1:  n = 4, taps = (4, 3)
            x^10 + x^7 + 1: n = 10, taps = (10, 7)
            x^14 + x^13 + x^12 + x^2 + 1: n = 14, taps = (14, 13, 12, 2)

        n: the length of the register

        taps: an iterable containing the tap positions. Tap positions must be unique and less than or equal to n. Tap
        positions cannot be 0. For polynomial representations, enter the powers of x as a tuple e.g. x^4 + x^3 + 1 would
        be (4, 3)

        """
        self.LFSR(n, taps)

        # Pad if it isn't a good multiplier
        self.data = self.data.flatten()

        pad_amount = None
        if len(self.data) % n:
            pad_amount = n - (len(self.data) % n)
            pad = np.zeros(shape=pad_amount, dtype=np.uint8)
            self.data = np.concatenate((self.data, pad))

        # Tile the pseudo random sequence as many times as we have to so as to xor with the data
        tiles = np.ceil(self.data.size / self.pseudo_rand_sequence.size)
        tiles = int(tiles)
        self.data ^ np.tile(self.pseudo_rand_sequence, tiles)[0:self.data.size]

        # Cut off the padding
        if pad_amount:
            self.data = self.data[:-pad_amount]

    def ldpc_beliefprop(self):
        # Belief propagation for LDPC
        # https://yair-mz.medium.com/decoding-ldpc-codes-with-belief-propagation-43c859f4276d
        return None

    def ldpc_hard(self):
        # Hard decision rule for LDPC
        # https://uweb.engr.arizona.edu/~ece506/readings/project-reading/5-ldpc/LDPC_Intro.pdf
        return None










