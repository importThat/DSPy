from .message import Message
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from random import shuffle
from math import factorial
from ..util.utils import markify


class Symbol2bit:
    def __init__(self, pattern, bits_per_symbol):
        self.number_of_possible_maps = None     # a deduced count of all the possible maps
        self.s2bmaps = None     # The known maps
        self.prob_map = None    # symbol probability map created by the matches
        self.matches = None     # What values the symbols appear to match too
        self.cut_patterns = None      # the symbol patterns of the cuts
        self.cuts = None    # All the different cuts of the original pattern
        self.symbol_map = None

        self.message = None
        self.pattern = pattern
        self.pattern_len = len(self.pattern)

        self.bits_per_symbol = bits_per_symbol

    def load_message(self, data):
        self.message = Message(data=data)

    def pad_message(self):
        remainder = self.message.data.size % self.bits_per_symbol

        if remainder:
            pad = np.zeros([abs((self.bits_per_symbol - remainder)), ])
            self.message.data = np.concatenate([self.message.data, pad])
        self.message.data = self.message.data.astype(np.uint8)

    def create_symbols(self):
        self.pad_message()
        self.message.symbolise(bits_per_symbol=self.bits_per_symbol)

    def randomise_symbols(self):
        """
        randomise the symbols so that they are unknown (this function is for testing the mapper)
        """
        old = np.arange(2 ** self.bits_per_symbol, dtype=np.uint8)
        new = np.random.choice(old, 2 ** self.bits_per_symbol, replace=False)
        self.message.data = new[self.message.data]  # Randomly swaps the symbol values
        self.symbol_map = new

    def sync_cuts(self):
        """
        Creates each cut of the sync, i.e. the sync pattern in symbols starting from each bit position of the bits per
        symbol
        """
        patterns = []

        # create each 'cut' of the sync
        for j in range(self.bits_per_symbol):
            remainder = (self.pattern_len - j) % self.bits_per_symbol
            length = self.pattern_len - remainder
            pattern = self.pattern[j:length]
            patterns.append(pattern)

        cut_symbols = []
        # Turn each cut into symbols and store them
        for pattern in patterns:
            data = np.array([i for i in pattern], dtype=np.uint8)
            cut = self.symbolise_data(data)
            cut_symbols.append(cut)

        self.cuts = cut_symbols

    def symbolise_data(self, data):
        """
        Turns the data into symbols as per the stored symbol map
        """
        # Create the symbol matrix
        symbol_matrix = np.arange(self.bits_per_symbol)
        symbol_matrix = 2 ** symbol_matrix
        symbol_matrix = symbol_matrix[::-1]  # Reverse it
        symbol_matrix = symbol_matrix.reshape([self.bits_per_symbol, -1])

        # Change the bits to symbols
        bits = data.reshape([-1, self.bits_per_symbol])
        bits = bits.dot(symbol_matrix)
        bits = bits.flatten().astype(np.uint8)

        return bits

    def markify_cuts(self):
        """
        Changes the cut symbols into markers
        """
        marks = []

        for j in range(len(self.cuts)):
            symbols = self.cuts[j]
            marks.append(markify(symbols))

        self.cut_patterns = marks

    def pattern_search(self):
        """
        Searches through the message symbols for the pattern of symbols seen in the sync
        """

        # Look for each pattern at every position of the message
        matches = []

        # Now we test the patterns against chunks of the symbols until we find a match I guess!
        for i in range(len(self.cut_patterns)):
            # The length is going to be the length of the cut that the pattern came from
            symbols_length = len(self.cuts[i])

            marker = self.cut_patterns[i]
            marker_len = len(marker)

            # Go through the message and test each symbol chunk to see if it follows the same pattern as the marker
            for j in range(0, len(self.message.data)):
                test = markify(self.message.data[j:j + symbols_length])

                # If they're the same length then it's a possible match
                if len(test) == marker_len:

                    # If we have found the marker
                    if np.all(test == marker):
                        # print(f"Marker successfully found at position {j}")
                        # print(f"symbols are {self.message.data[j:j + symbols_length]}")
                        # print(f"marker is {marker}")
                        # print(f"pattern is {self.cut_patterns[i]}")
                        new_match = np.column_stack((self.message.data[j:j + symbols_length], self.cuts[i]))
                        matches.append(new_match)

        # Just get the unique ones
        # TODO
        #  Check this line, do we want to make it unique here??
        matches = [np.unique(i, axis=0) for i in matches]

        # store the matches
        self.matches = np.vstack(matches)

        # Calculate how many possible matches there are
        n_symbols = len(np.unique(self.matches[:, 0]))
        tests = 2 ** self.bits_per_symbol - n_symbols
        self.number_of_possible_maps = factorial(tests)
        if self.number_of_possible_maps == 0:
            self.number_of_possible_maps = 1

    def plot_matches(self):
        """
        Creates a scatter plot showing all the observed matches
        """
        plt.scatter(self.matches[:, 0], self.matches[:, 1], alpha=0.2, s=100)
        plt.xlabel("Known Symbol")
        plt.ylabel("Observed Symbol")
        plt.title("Observed Symbol Pattern Matches")

    def create_probability_map(self):
        """
        Turns the observed matches into a probability map
        """

        prob_map = []

        for i in range(2 ** self.bits_per_symbol):
            occurrences = self.matches[self.matches[:, 0] == i]  # Get the tuples for the ith symbol
            counts = Counter(occurrences[:, 1])  # The occurrences of the observed values

            probs = []
            for j in range(2 ** self.bits_per_symbol):
                if j in counts.keys():
                    probs.append(counts[j])
                else:
                    probs.append(0)

            probs = np.array(probs)
            if probs.sum() != 0:
                probs = probs / probs.sum()
            prob_map.append(probs)

        self.prob_map = np.vstack(prob_map)  # Store a map in memory

    def blur_prob_map(self, sd=0.4):
        """
        Applies a gaussian blur to the probability map with the given standard deviation
        """
        self.prob_map = gaussian_filter(self.prob_map, sigma=sd, axes=1)

    def plot_prob_map(self):
        """
        Creates a mesh plot of the probability map
        """
        plt.pcolormesh(self.prob_map)
        ax = plt.gca()
        # invert the axis
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.xaxis.tick_top()

        plt.xlabel("Observed Symbol")
        plt.ylabel("Known Symbol")
        plt.title("Observed Symbol Pattern Matches")

    def raster(self, blocksize=244):
        """
        creates a raster plot of the message data
        """
        self.message.data = self.message.data.reshape([-1, blocksize])
        plt.pcolormesh(self.message.data, cmap='binary')
        # ax = plt.gca()
        # # invert the axis
        # ax.set_ylim(ax.get_ylim()[::-1])
        # ax.xaxis.tick_top()
        plt.title("Raster Plot of Data")

        self.message.data = self.message.data.flatten()

    def symbols_to_binary(self, map):
        bins = np.array([self.binify(i) for i in map])
        bins = bins.astype(np.uint8).flatten()
        bins = bins.reshape([-1, self.bits_per_symbol])

        return bins

    def binify(self, symbol):
        x = bin(symbol)[2:].zfill(self.bits_per_symbol)
        return list(x)

    def test_probs(self, iters=1000):
        """
        creates successive symbol to bit maps from the probability map
        """
        possible_maps = []

        # Loop
        for j in range(iters):
            if len(possible_maps) == self.number_of_possible_maps:
                print("\nFound all maps!")
                break
            prob_map = self.prob_map.copy()

            # Now we draw probabilities from them
            draw = []
            possibilities = range(2 ** self.bits_per_symbol)
            for i in range(2 ** self.bits_per_symbol):

                if (prob_map[i].sum() - 0.1) <= 0:  # If the probabilities are all zero or nearly zero just move on
                    draw.append(99999)  # Big placeholder
                    continue

                # Pick a symbol to map to this symbol
                choice = np.random.choice(possibilities, size=1, p=prob_map[i])
                draw.append(choice[0])

                # Once we select something we need to zero out that column and add any prob in that to the other columns
                # so that the probs sum to 1 still
                divisor = prob_map.shape[1] - len(draw) + draw.count(99999)

                col = prob_map[:, choice]
                if divisor:
                    col = col / divisor

                change = np.ones(shape=prob_map.shape)
                change_map = col * change

                # Add the change map in to the prob_map
                prob_map = prob_map + change_map
                # Zero the chosen columns
                mask = [i for i in draw if i != 99999]
                prob_map[:, mask] = 0

            # Ugh... This is getting hairy
            # Figure out which symbols haven't been used yet and randomly add them in
            unused = [i for i in possibilities if i not in draw]
            shuffle(unused)

            draw = np.array(draw)
            draw[draw == 99999] = unused

            # Now we can try the symbol to bit map!
            test_bits = self.symbols_to_binary(draw)

            # Try the bit map
            demod = test_bits[self.message.data]
            demod = demod.flatten()

            if (j % 20) == 0:
                print(f"On loop {j} of {iters}")

            if np.all(demod[0:self.pattern_len] == self.pattern):
                if len(possible_maps) > 0 and np.any([np.all(test_bits==i) for i in possible_maps]):  # If the map is
                    # already stored and there is something in the possible maps
                    continue
                else:
                    print("")
                    print("********************************")
                    print("Successfully found a bit map!")
                    print(f"bit map is:")
                    print(test_bits)
                    possible_maps.append(test_bits)

        self.s2bmaps = possible_maps

    def bitstr_to_ascii(self, bits):
        out = ''
        end = len(bits) - (len(bits) % 8)

        for i in range(0, end, 8):
            character = chr(int(bits[i:i + 8], 2))
            out += character

        return out

    def test_s2bmaps(self, sync_len, data_len):
        """
        Attempts to print the ascii recovered from the symbols using the discovered maps. Mostly for testing because
        the data is unlikely to be plain ascii without a randomizer
        """
        j = 0
        for s2b in self.s2bmaps:
            text_bits = s2b[self.message.data]
            text_bits = text_bits.flatten()

            # Put it in block format
            blocksize = sync_len + data_len
            trim = int(len(text_bits) / blocksize)
            text_bits = text_bits[0:trim*blocksize]

            # Reshape it
            text_bits = text_bits.reshape([-1, blocksize])
            # Cut off the preamble
            text = text_bits[:, sync_len:]
            text = text.flatten()

            # Convert to ascii and print it out
            text_str = ''.join([str(i) for i in text])
            text_ascii = self.bitstr_to_ascii(bits=text_str)

            print(f"S2B map {j} gives text:\n{text_ascii}")

            j += 1

    def save(self, fn):
        """
        Saves out the possible bit files and a text file containing the possible bits
        """
        for i in range(len(self.s2bmaps)):
            file_name = fn + f"_bitmap={i}.bits"
            print(f"Saving bits from map {i} as {file_name}")

            s2b = self.s2bmaps[i]
            text_bits = s2b[self.message.data]
            text_bits = text_bits.flatten()
            text_bits = text_bits.astype(np.uint8)
            text_bits = np.packbits(text_bits, axis=-1)
            text_bits.tofile(file_name)

            if i >= 100:
                break

        s2b_text = ""

        for i in range(len(self.s2bmaps)):
            s2b_text += f"Symbol to bit map {i}:\n"
            s2b = self.s2bmaps[i]
            for i in range(len(s2b)):
                z = str(s2b[i])
                z = z.replace("[", "")
                z = z.replace("]", "")
                z = z.replace(" ", ",")
                z += "\n"

                if i == 0:
                    s2b_text += "{\n"

                s2b_text += f"\t{i} : "

                s2b_text += z
            s2b_text += "}\n"

        print(f"Saving symbol to bit maps as {fn}_s2b_maps.txt")
        with open(f"{fn}_s2b_maps.txt", "w") as f:
            f.writelines(s2b_text)
