import dsproc

"""
Continuous Phase Frequency shift keying! This is FSK but with a phase offset applied to every sample to reduce
instantaneous phase transitions
"""

# Intermediate Hz, This will be the highest frequency that will occur in the modulated wave
f = 2000
sps = 32    # samples per symbol
fs = 8000   # sampling rate. I'm going to use a high fs here so the CPFSK is easier to see

# Our message in symbol form. In this example we have 4 symbols, so each symbol would typically represent 2 bits
MESSAGE = dsproc.create_message(5000, 8)

# Create the sig object with the given params
s = dsproc.Mod(message=MESSAGE, fs=fs, sps=16, f=f, amplitude=1)

# Apply the frequency shift keying
s.CPFSK()

# CPFSK uses a phase offset to smooth the transitions between frequency shifts. You can see this in the time
# graph
s.time(n=s.sps*10)

# The fft is also a bit cleaner than the fsk we looked at earlier
s.fft()

# baseband it
s.baseband()

# The specgram also looks quite well formed
s.specgram(nfft=1024)

# Saves the samples as complex64 (compatible with gnuradio/usrp)
#s.save("CPFSK_test")
