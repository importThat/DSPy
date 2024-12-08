import numpy as np
import dsproc
from matplotlib import pyplot as plt
import os
np.random.seed(42)  # Set the seed for reproducibility

"""
This file steps through demodulating a noisy sig. Run the "generate_noisy_4QAM.py" program first to create the sig

I suggest stepping through this line by line in a console
"""
# Filename
fn = "QAM_generated_m=4_fs=2000_sps=8"
# These are the symbols that are stored in the QAM file. We're generating them here so we can test to see if our
# output is correct
known_message = dsproc.create_message(10000, 4)

# You will need to change this to point at the directory which contains the noisy QAM sig
path = f"{os.getcwd()}\\Examples\\{fn}"

# Read in the file
# (Change fs if you changed the sampling rate)
s = dsproc.Demod(fn=path, fs=2000)

# # Look at the sig
# s.fft()
# s.iq()
# s.time()

# You can see in the time view that the sig doesn't start at 0
# We need to trim the excess from the sig
s.trim_by_power()

# ********** Frequency and Phase offsets ************
# Correct the frequency offsets

# By raising the sig to the power of 4 we can see the frequency offset and the samples per symbol
# freq_offset is just the biggest peak in the graph
freq_offset = s.exponentiate(order=4)

# Baseband the signal using the frequency offset
s.freq_offset(freq_offset)

# from the exponentiate graph we can also see that our symbol rate is 250 symbols per second. This is just the
# difference between the spikes in the plot.
# This means our samples per symbol is:
sps = int(s.fs / 250)

# Looks a bit better!
# s.iq()

# Next we want to resample so that we only have 1 sample per symbol.
# By upsampling we also seek to correct any sub-sample phase offsets

# First upsample (aka interpolate, fit extra samples in between the currently known ones)
up=10
s.resample(up=up, down=1)
up_sps = int(sps * up)

# We want to sample at the peak of the wave. This will correct small phase offsets and also reduce our sig down
# to one sample per symbol
# The magnitudes
mags = []

# For every possible phase offset, resample the data and then calculate it's average magnitude
for i in range(int(up_sps)):
    test_division = s.samples[i::int(up_sps)]
    mag = np.mean(np.abs(test_division))
    mags.append(mag)

# You can see that the magnitude oscillates. We want to sample at one of the peaks. One will be the highest positive
# magnitude, the other will be the highest negative magnitude
# plt.plot(mags)

# Get the index of the biggest magnitude
n = mags.index(max(mags))

# re-sample
# Starting at a peak, get each symbol!
s.samples = s.samples[n::int(up_sps)]
# s.iq()

# Nice! It looks like a noisy sig
# Let's bring in a constellation map and get the message back
c = dsproc.Constellation(M=4)
# Make the map
c.square()
# normalise it
c.normalise()

# We need to normalise our sig so it's on the same scale as the constellation
s.normalise_amplitude()

# Show the points next to the qam
# plt.scatter(s.samples.real[0:1000], s.samples.imag[0:1000])
# plt.scatter(c.map.real, c.map.imag)

# There's clearly a phase offset, lets try to line the dots up
s.phase_offset(angle=-55)
s.normalise_amplitude()

# Looks close enough. The clusters just have to be closest to a single dot.
# plt.scatter(s.samples.real[0:1000], s.samples.imag[0:1000])
# plt.scatter(c.map.real, c.map.imag)

# We have lost a sample somewhere

# There's a problem though. The last symbol is sitting in the middle of the plot. This is an artifact from our trimming
# function. That's ok, we can just drop it.
# Decode the message, trying all different phase rotations
for i in range(0, 360, 90):
    s.phase_offset(i)
    message = s.QAM(c=c)
    if np.all(message[1:] == known_message[:len(message)-1]):
        print("message successfully recovered!")
        break

# print(message[1:100])