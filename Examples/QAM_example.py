from sig.Mod import Mod
from util import Utils

"""
Generate QAMs
"""

# Intermediate Hz, This will be the highest frequency that will occur in the modulated wave
F = 1000

# Create a random message of n symbols with m levels
MESSAGE = Utils.create_message(n=10000, m=5028)

SYMBOL_RATE = 250      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration (in seconds)

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate (for ease of use reasons)
Fs = SYMBOL_RATE
while Fs <= F * 2:
    Fs += SYMBOL_RATE

# Create the sig object with the given params
s = Mod(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

# There are a few different QAM types that are currently supported. Use the one that appeals to you.
# the .QAM method generates a constellation map for the number of symbols provided and then applys that map
# to the sig

# The QAM method works with arbitrary numbers of unique symbols and will trim the constellation down to the correct
# size. Try changing the m in the create_messages functions to any integer

#s.QAM(type="square")
s.QAM(type="sunflower")
# s.QAM(type="star")
# s.QAM(type="square_offset")     # AKA regular hexagon

# Baseband the sig
s.baseband()

# Look at the IQ plot of the sig
s.iq()

s.save("QAM_Test")
