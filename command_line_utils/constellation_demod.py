import sys
import dsproc
import numpy as np
from math import log


def binify(symbol, bits_per_symbol):
    x = bin(symbol)[2:].zfill(bits_per_symbol)
    return list(x)


def symbol_to_bit(symbols, bits_per_symbol):
    bins = np.array([binify(i, bits_per_symbol) for i in symbols])
    bins = bins.astype(np.uint8).flatten()

    return bins


print("\n")

variables = sys.argv

if len(variables) <= 2:
    print("constellation_demod file N samples=1000")
    print("file -> A file of 64 bit complex points")
    print("N -> Your guess at how many constellation points there are")
    print("samples -> how many samples to plot, default = 1000")
    print("Returns bits using a generic symbol to bit mapping")
    sys.exit()

file = variables[1]
N = int(variables[2])
if len(variables) > 3:
    samples = int(variables[3])
else:
    samples = 1000

s = dsproc.Demod(filename=file, fs=1)
c = s.detect_clusters(N)
s.view_constellation(c, samples=samples)

out = s.QAM(c)

n_points = max(out)
bps =int(np.ceil(log(n_points, 2)))

bits = symbol_to_bit(out, bits_per_symbol=bps)
bits = bits.astype(np.uint8)

fn = file + ".bits"

print(f"saving demod as {fn}")
bits.tofile(fn)


