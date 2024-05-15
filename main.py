import Utils
from Signal import Signal
import numpy as np


F = 50     # Intermediate Hz, This will be the highest frequency that will occur in the modulated wave

MESSAGE = Utils.create_message(n=100, m=6) # Message symbols

SYMBOL_RATE = 50      # Symbols per second
DUR = len(MESSAGE) / SYMBOL_RATE    # Message duration

# We want the sampling rate to satisfy Nyquist's level (2* highest frequency) and to also be an integer
# multiple of the symbol rate
Fs = SYMBOL_RATE
while Fs < F * 2:
    Fs += SYMBOL_RATE

s = Signal(message=MESSAGE, f=F, fs=Fs, duration=DUR, amplitude=1)

#s.QAM(type="square")
s.ASK()
#s.plot("scatter")
#s.plot("fft")
s.save(fn="ASK_Test")


# import wave
#
# with wave.open("sound.wav", "wb") as f:
#     f.setnchannels(1)
#     f.setsampwidth(2)
#     f.setframerate(0)
#     f.writeframesraw(s.samples)