import Utils
from Signal import Signal
import numpy as np


F = 100     # Hz, This will be the highest frequency that will occur in the modulated wave

MESSAGE = Utils.create_message(n=32, m=64) # Message symbols

SYMBOL_RATE = 50      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate
Fs = SYMBOL_RATE
while Fs < F * 2:
    Fs += SYMBOL_RATE

s = Signal(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

s.QAM(type="square")
#s.plot("scatter")
s.plot("constellation")
s.save(fn="QAM_Test")
