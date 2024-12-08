import dsproc
import numpy as np

"""
This program generates a 4-QAM (4-QPSK) and adds noise, a frequency offset, and a phase offset to the sig to
simulate travelling through a channel and is then saved. The program "demod_noisy_4QAM.py" steps through demodulating
the sig

I suggest stepping through this line by line in a console
"""

#                                   PART ONE - CREATE THE SIGNAL

np.random.seed(42)  # Set the seed for reproducibility

M = 4   # The number of unique symbols
MESSAGE = dsproc.create_message(n=10000, m=M)  # Message symbols
# Print a sample of the message
print(MESSAGE[0:50])

# Symbol rate
fs = 2000
f = 500

# Samples per symbol
sps=8

# Create the sig
s = dsproc.Mod(fs=fs, message=MESSAGE, sps=sps, f=f)

# Apply QAM modulation
s.QAM(constellation="square")

# Look at the spectrum
# s.fft()

# we can see that the sig is entirely unfiltered and spreads throughout the spectrum. We should filter it down
# to be a bit more polite!
s.baseband()
_ = s.butterworth_filter(frequencies=250, filter_type="lowpass", order=5)

# The spectrum looks much cleaner now
# s.fft()

# Shift it down to baseband


# Here is the IQ plot, it looks pretty funky!
# s.iq()

#                              PART TWO - SIMULATE CHANNEL INTERFERENCE

# Add white gaussian noise. This is the "background noise of the universe" or some such. Just random noise that
# effects the sig while it is propagating
# Create the noise
noise = dsproc.AWGN(n=len(s.samples), power=0.02)
# Add it in
s.samples = s.samples + noise

# Note how the spots spread out a bit
# s.iq()

# Add some noise at the start and the end of the sig to simulate a real capture
noise_amount = int(0.2*len(noise))
s.samples = np.concatenate([noise[0:noise_amount], s.samples, noise[0:noise_amount]])

# Note how we now have a bunch of dots around 0,0
# s.iq()

# You can see the preceeding and following noise in the time domain
# s.time()

# Add a phase offset

# First create the time vector
t = 1 / s.fs * np.arange(s.dur * s.fs)
t = t[0:len(s.samples)]

# Next compute the angle. This comes from the equation:
#   Signal = A * np.cos(2 * np.pi * f * t + theta) + A * 1j * np.sin(2 * np.pi * f * t + theta)
# Here we're just interested in changing the theta bit by pi/4 (45 degrees)
angle = 2 * np.pi * 0 * t + np.pi/4    # Add a 45 degree phase offset
phase_offset = np.cos(angle) + 1j * np.sin(angle)
phase_offset = phase_offset.astype(np.complex64)

# rotate the sig by the phase offset
s.samples = s.samples * phase_offset

# (Alternatively just use the native phase_offset function - s.phase_offset(angle))

# Note how the symbols have all rotated by 45 degrees (pi/4)
# s.iq()


# Add a frequency offset
freq = 2 * np.pi * 400 * t + 0    # Add a 400 hz offset
# Create the wave
offset = np.cos(freq) + 1j * np.sin(freq)
offset = offset.astype(np.complex64)
# Apply the frequency offset to the samples
s.samples = s.samples * offset

# (Alternatively just use the native freq_offset function - s.freq_offset(freq))

# Note how we now have a circle, because the frequency offset is causing the IQ points to spin
# s.iq()
# slightly up from 0
# s.fft()

# Save the wave
# s.save_wave(fn=f"QAM_generated_m={M}_fs={fs}_sps={sps}")

