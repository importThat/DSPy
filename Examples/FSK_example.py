import dsproc

"""
Frequency shift keying example
"""

# Intermediate Hz, This will be the highest frequency that will occur in the modulated wave
# I use a higher frequency here so we can see the fsk more easily
F = 2000

# Our message in symbol form. In this example we have 4 symbols, so each symbol would typically represent 2 bits
# 0 = "00"
# 1 = "01"
# 2 = "10"
# 3 = "11" etc.
# Create a random message of 100 symbols with 4 levels
MESSAGE = dsproc.utils.create_message(100, 4)

SYMBOL_RATE = 250      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration (in seconds)

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate (for ease of use reasons)
Fs = SYMBOL_RATE
while Fs <= F * 2:
    Fs += SYMBOL_RATE

# Increase the symbol rate a bit so the FSK is easier to see
Fs += SYMBOL_RATE * 2

# Create the sig object with the given params
s = dsproc.Mod(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

# Apply the frequency shift keying
s.FSK()

# We can then explore the wave using the plot functions
# plot the first 400 samples in the time domain
s.time(n=400)
# You can see from the fft that the wave hasn't been filtered at all
s.fft()

# The wave is currently at 2000hz, we can baseband it with the baseband function
s.baseband()

# Note that the graphs have now changed.
s.time()
s.fft()
s.psd()
s.specgram(nfft=1024)

# the actual samples that we have generated can be explored
# You can see that they are complex 64 type, which can be thought of as x, y coordinates (kind of)
s.samples[0:20]

# This sample can then be saved and transmitted with GNU-radio or the USRP python interface. Doing it either way
# is very straightforward

# I believe gnuradio and the usrp python api will mix your sig back up to the transmit frequency, but if not you can
# apply a frequency shift with the freq_offset method
s.freq_offset(freq=500)
s.fft()
# baseband it
s.baseband()

# Saves the samples as complex64 (compatible with gnuradio/usrp)
s.save("FSK_test")

