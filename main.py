import Utils
from Signal import Signal
import numpy as np


F = 100     # Hz, This will be the highest frequency that will occur in the modulated wave
DUR = 100     # Seconds

MESSAGE = Utils.create_message(n=2000, m=1200) # Message symbols

SYMBOL_RATE = 1 / (DUR / len(MESSAGE))      # Symbols per second

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
