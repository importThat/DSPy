import dsproc
import numpy as np
from matplotlib import pyplot as plt

"""
Frequency shift keying example
"""

np.random.seed(50)

# Intermediate Hz, This will be the highest frequency that will occur in the modulated wave
F = 2000

# 1000 symbols with 2 levels (so 1000 bits)
MESSAGE = dsproc.utils.create_message(1000, 2)
print(f"Message bits {MESSAGE[0:20]}")

SYMBOL_RATE = 250      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration (in seconds)

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate (for ease of use reasons)
Fs = SYMBOL_RATE
while Fs <= F * 2:
    Fs += SYMBOL_RATE

# Create the sig object with the given params
s = dsproc.Mod(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

# Apply the frequency shift keying
s.FSK()
# baseband
s.baseband()

# Ok we have our wave and we need to demod it, ref:
# https://www.allaboutcircuits.com/technical-articles/digital-signal-processing-in-scilab-how-to-decode-an-fsk-signal
# We can demod FSk by multiplying it with a wave from the expected 0 frequency and the expected 1 frequency,
# average them, and that will produce a nice square wave of our symbols
# This uses the following identites:
#   sin(ω1t)∗sin(ω2t) = 1/2(cos((ω1−ω2)t) − cos((ω1+ω2)t))
#   cos(ω1t)∗cos(ω2t) = 1/2(cos((ω1−ω2)t) + cos((ω1+ω2)t))
# Which means that if our expected frequencies are correct we should get a spike at 1/2(cos(0)) - cos(0), which is 0.5
# or 1/2(cos(0) + cos(0) which is 1.5

freq_0 = -500
freq_1 = 500

# Create the two waves
t = 1/Fs * np.arange(DUR * Fs)

angle_0 = 2 * np.pi * freq_0 * t
angle_1 = 2 * np.pi * freq_1 * t
wave_0 = np.cos(angle_0) + 1j * np.sin(angle_0)
wave_1 = np.cos(angle_1) + 1j * np.sin(angle_1)

# Now we multiply them by the wave
decode_0 = wave_0 * s.samples
decode_1 = wave_1 * s.samples

# It's working! We can also clearly see the samples per symbol which is really cool.
plt.plot(decode_0.real[0:200])
plt.plot(decode_1.real[0:200])

# Now we just average them over the sample
sps = 17 # from the graph
av_0 = np.array([np.mean(decode_0.real[i*sps:i*sps+sps]) for i in range(int(len(decode_0)/sps))])
av_1 = np.array([np.mean(decode_1.real[i*sps:i*sps+sps]) for i in range(int(len(decode_1)/sps))])

# We can cleary see the symbols!
plt.plot(av_0[0:200])
plt.plot(av_1[0:200])

# Get the symbols out
av_0 = np.round(av_0, 0).astype(np.int32)

# Check the difference between the original message
if np.all(MESSAGE==av_0):
    print("Message succesfully decoded")



