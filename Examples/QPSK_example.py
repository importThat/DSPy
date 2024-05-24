from Mod import Mod
import Utils

"""
Generate a QPSK signal
"""

# Intermediate Hz, This will be the highest frequency that will occur in the modulated wave
F = 1000

# Create a random message of 1000 symbols with 256 levels
MESSAGE = Utils.create_message(n=2000, m=16)

SYMBOL_RATE = 250      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration (in seconds)

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate (for ease of use reasons)
Fs = SYMBOL_RATE
while Fs <= F * 2:
    Fs += SYMBOL_RATE

# Create the signal object with the given params
s = Mod(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

s.QPSK()

# Baseband the signal
s.baseband()

# Look at the IQ plot of the signal
s.iq()

# You can see that it is on the unit circle and has placed each symbol into one of M different phases

# Save
s.save("QPSK_Test")
