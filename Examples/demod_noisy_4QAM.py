from sig import Demod
import numpy as np
from matplotlib import pyplot as plt
from constellation.Constellation import Constellation

"""
This file steps through demodulating a noisy sig. Run the "generate_noisy_4QAM.py" program first to create the sig

I suggest stepping through this line by line in a console
"""
# Filename
fn = "QAM_generated_m=4_fs=155000_sr=5000"

# You will need to change this to point at the directory which contains the noisy QAM sig
path = f"C:\\Users\\Justi\\Documents\\PythonProjects\\DSPy\\modulations\\{fn}"

# Read in the file
# (Change fs if you changed the sampling rate)
s = Demod.Demod(filename=path, fs=155000)

# # Look at the sig
s.fft()
s.iq()
s.time()

# You can see in the time view that the sig doesn't start at 0
# We need to trim the excess from the sig
s.trim_by_power()
# s.samples = s.samples[62000:-62000]     # Trim by hand

# ********** Frequency and Phase offsets ************
# Correct the frequency offsets

# By raising the sig to the power of 4 we can see the frequency offset and the samples per symbol
freq_offset = s.exponentiate(order=4)

# Move down by freq_offset
s.freq_offset(-1*freq_offset)

# from the exponentiate graph we can also see that our symbol rate is 5000 (6600 - 1600)
# This means our samples per symbol is:
sps = s.fs / 5000

# Looks a bit better!
s.iq()

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
plt.plot(mags)

# Get the index of the biggest magnitude
n = mags.index(max(mags))

# re-sample
# Starting at a peak, get each symbol!
s.samples = s.samples[n::int(up_sps)]
s.retime()  # Re-time because we have dropped a bunch of samples
s.iq()

# Nice! It looks like a noisy sig
# Let's bring in a constellation map and get the message back
c = Constellation(M=4)
# Make the map
c.square()
# normalise it
c.normalise()

# We need to normalise our sig so it's on the same scale as the constellation
s.normalise_pwr()

# Show the points next to the qam
plt.scatter(s.samples.real[0:1000], s.samples.imag[0:1000])
plt.scatter(c.map.real, c.map.imag)
# close enough for government work (as they say)

# Decode the message
message = s.QAM(c=c)

print(message[0:30])

# Nearly there, but there's still a 90 degree phase offset, so if we rotate we should get the exact symbols
s.phase_offset(angle=-90)

# Decode again
message = s.QAM(c=c)

print(message[0:30])