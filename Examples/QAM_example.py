import dsproc

"""
Generate QAMs
"""
# Create a random message of n symbols with m levels
MESSAGE = dsproc.utils.create_message(n=10000, m=32)

Fs = 2000
sps = 2

# Create the sig object with the given params
s = dsproc.Mod(Fs, MESSAGE, sps)

# There are a few different QAM types that are currently supported. Use the one that appeals to you.
# the .QAM method generates a constellation map for the number of symbols provided and then applys that map
# to the sig

# The QAM method works with arbitrary numbers of unique symbols and will trim the constellation down to the correct
# size. Try changing the m in the create_messages functions to any integer

#s.QAM(type="square")
#s.QAM(type="sunflower")
# s.QAM(type="star")
s.QAM(type="square_offset")     # AKA regular hexagon

# Baseband the sig
s.baseband()

# Look at the IQ plot of the sig
s.iq()

s.save("QAM_Test")
