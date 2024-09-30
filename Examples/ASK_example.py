import dsproc

"""
Amplitude shift keying example
"""

# Our message in symbol form. In this example we have 4 symbols, so each symbol would typically represent 2 bits
# 0 = "00"
# 1 = "01"
# 2 = "10"
# 3 = "11" etc.
MESSAGE = [0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2, 1, 3]
# For automated message generation use the create_message function in the Utils file

# We need to specify the sampling frequency and the samples per symbol
Fs = 2000
sps = 2
# Create the sig object with the given params
s = dsproc.Mod(Fs, MESSAGE, sps)

# Apply the amplitude shift keying
s.ASK()

# We can then explore the wave using the plot functions
s.time()
s.fft()

# The wave is currently at 50Hz, we can baseband it with the baseband function
s.baseband()

# Note that the graphs have now changed. The waves are square (because there is no frequency), the imaginary part
# has been removed and the frequency in the fft has moved to zero
s.time()
s.fft()
s.psd()

# the actual samples that we have generated can be explored
# You can see that they are complex 64 type, which can be thought of as x, y coordinates
s.samples[0:20]

# This sample can then be saved and transmitted with GNU-radio or the USRP python interface. Doing it either way
# is very straightforward

# Saves the samples as complex64 (compatible with gnuradio/usrp)
#s.save("ASK_test")

