import Filter
import Utils
from Mod import Mod
import numpy as np

"""
This program generates a 4-QAM (4-QPSK) and adds noise, a frequency offset, and a phase offset to the signal to
simulate travelling through a channel and is then saved. The program "demod_noisy_4QAM.py" steps through demodulating
the signal

I suggest stepping through this line by line in a console
"""

#                                   PART ONE - CREATE THE SIGNAL

np.random.seed(42)  # Set the seed for reproducibility

F = 50000   # Intermediate
M = 4   # The number of unique symbols
MESSAGE = Utils.create_message(n=10000, m=M)  # Message symbols
# Print a sample of the message
print(MESSAGE[0:50])

SYMBOL_RATE = 5000    # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate
Fs = SYMBOL_RATE
while Fs <= F * 3:
    Fs += SYMBOL_RATE

# Create the signal
s = Mod(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

# Apply QAM modulation
s.QAM(type="square")

# Look at the spectrum
s.fft()

# we can see that the signal is entirely unfiltered and spreads throughout the spectrum. We should filter it down
# to be a bit more polite!

# Create a filter object with sps*10+1 taps (the magic number!)
my_filter = Filter.Filter(num_taps=s.sps*10 + 1, fs=Fs)
# Create a finite infinite response filter
my_filter.FIR(width=SYMBOL_RATE*2)
# Apply the filter to the signal at the correct frequency
s.samples = my_filter.apply(s.samples, f_shift=F)

# The spectrum looks much cleaner now
s.fft()

# Shift it down to baseband
s.baseband()

# Here is the IQ plot, it looks pretty funky!
s.iq()

#                              PART TWO - SIMULATE CHANNEL INTERFERENCE

# Add white gaussian noise. This is the "background noise of the universe" or some such. Just random noise that
# effects the signal while it is propagating
# Create the noise
noise = Utils.AWGN(n=len(s.samples), power=0.02)
# Add it in
s.samples = s.samples + noise

# Note how the spots spread out a bit
s.iq()

# Add some noise at the start and the end of the signal to simulate a real capture
noise_amount = int(0.2*len(noise))
s.samples = np.concatenate([noise[0:noise_amount], s.samples, noise[0:noise_amount]])
s.dur += 2*noise_amount/s.fs

# Note how we now have a bunch of dots around 0,0
s.iq()

# You can see the preceeding and following noise in the time domain
s.time()

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

# rotate the signal by the phase offset
s.samples = s.samples * phase_offset

# Note how the symbols have all rotated by 45 degrees (pi/4)
s.iq()


# Add a frequency offset
angle = 2 * np.pi * 400 * t + 0    # Add a 400 hz offset
# Create the wave
offset = np.cos(angle) + 1j * np.sin(angle)
offset = offset.astype(np.complex64)
# Apply the frequency offset to the samples
s.samples = s.samples * offset

# Note how we now have a circle, because the frequency offset is causing the IQ points to spin
s.iq()
# slightly up from 0
s.fft()

# Save the plot
s.save(fn=f"QAM_generated_m={M}_fs={Fs}_sr={SYMBOL_RATE}")

