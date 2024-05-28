from Mod import Mod
import Utils

"""
Continuous Phase Frequency shift keying! This is FSK but with a phase offset applied to every sample to reduce
instantaneous phase transitions
"""

# Intermediate Hz, This will be the highest frequency that will occur in the modulated wave
F = 2000

# Our message in symbol form. In this example we have 4 symbols, so each symbol would typically represent 2 bits
MESSAGE = Utils.create_message(5000, 8)

SYMBOL_RATE = 500      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration (in seconds)

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate (for ease of use reasons)
Fs = SYMBOL_RATE
while Fs <= F * 2:
    Fs += SYMBOL_RATE

# Increase the sample rate a bit so the CPFSK is easier to see
Fs += SYMBOL_RATE * 100

# Create the signal object with the given params
s = Mod(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

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
s.save("CPFSK_test")
