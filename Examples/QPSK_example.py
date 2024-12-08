import dsproc

"""
Generate a QPSK sig
"""

# Intermediate Hz, This will be the highest frequency that will occur in the modulated wave
f = 1000

# Create a random message of n symbols with m levels
MESSAGE = dsproc.create_message(n=2000, m=16)

sps = 8      # Symbols per sample
fs = 4000

# Create the sig object with the given params
s = dsproc.Mod(message=MESSAGE, fs=fs, sps=sps, f=f, amplitude=1)

s.QPSK()

# Baseband the sig
s.baseband()

# Look at the IQ plot of the sig
# s.iq()

# You can see that it is on the unit circle and has placed each symbol into one of M different phases

# Save
#s.save("QPSK_Test")
