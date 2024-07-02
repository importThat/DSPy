import sys
import dsproc
import numpy as np
from math import factorial


"""
Python symbol to bit mapper written by importThat

Takes in a bit file and outputs one or more bit files and a file containing the matching symbol to bit maps
Usage is s2bmap -file- -bits_per_symbol-
"""

variables = sys.argv

if len(variables) <= 3:
    print("\n")
    print("s2bmap file pattern bits_per_symbol\n")

    print("file -> The file to be symbol to bit mapped")

    print("pattern -> The known pattern which is used to construct the symbol to bit map. Can be a string of binary, a"
          "bit file of type uint8 ora pointer to a text file containing the pattern. Note that the text document must "
          "have the extension .txt")

    print("bits_per_symbol -> The number of bits per symbol."
          "A 4QAM has 2 bits per symbol, an 8QAM has 3 bits per, a 16QAM has 4 bits per symbol etc.")

    print("Output -> One or more files containing the mapped bits, and a text file containing the "
          "symbol to bit maps")
    print("\n")
    # Break
    sys.exit()

file = variables[1]
pattern = variables[2]
bits_per_symbol = int(variables[3])


print("\n")
# Tests
if type(file) != str:
    raise ValueError("File must be a string")

if type(bits_per_symbol) != int:
    raise ValueError("bits_per_symbol must be an integer")

if bits_per_symbol > 8:
    raise Warning("Large QAMS (8+ bits per symbol) may take a long time. Good luck!")

# Figure out the pattern
if set(pattern) == set("10"):   # If it's binary
    pattern = np.array([i for i in pattern], dtype=np.uint8)

elif ".txt" in pattern: # If it's a text document
    with open(pattern, "r") as f:
        data = f.readlines()
    data = "".join([i.strip() for i in data])
    data = np.array([i for i in data], dtype=np.uint8)
    pattern = data

else:
    data = np.fromfile(pattern, dtype=np.uint8)

if set(pattern) != set(np.array([1, 0])):
    raise ValueError("pattern must be a binary string, e.g. '10101010100' or a pointer to a text document containing"
                     "binary, e.g. pattern_doc.txt")

# Computation starts

print(f"Reading file {file}")
bits = np.fromfile(file, dtype=np.uint8)

# Create the mapper object
mapper = dsproc.Symbol2bit(pattern=pattern, bits_per_symbol=bits_per_symbol)
# Load the message
mapper.load_message(bits)
# Plot a raster to view the data
#mapper.raster()
# Change the message to symbols
mapper.create_symbols()

# This is where the real stuff begins
# Cut the sync up into the symbol representation of every bit position
print(f"Constructing symbol patterns")
mapper.sync_cuts()
# Change the cuts into 'markers' that show where every symbol occurs in the message
mapper.markify_cuts()
# Search through the symbols for the marker patterns
print(f"Searching for patterns")
mapper.pattern_search()

# Test to see if we have every symbol, if we don't have enough symbols we might have to break here
n_symbols = len(np.unique(mapper.matches[:, 0]))
tests = 2**bits_per_symbol - n_symbols
all_tests = factorial(tests)
if all_tests > 100:
    raise ValueError("There are more than 100 possible bit maps. Consider using a different pattern (W.S.N.H!).")

# Turn the matches into a probability map
mapper.create_probability_map()
# Show the probability map
#mapper.plot_prob_map()

# Test the probabilities and store compatible maps
print(f"Testing possible maps")
mapper.test_probs(iters=100)

# save the bits and the bit maps
mapper.save(fn=file)

print("\n")