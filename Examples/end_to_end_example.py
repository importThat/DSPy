from message.Message import Message
import numpy as np
from sig.Mod import Mod
from util.Utils import AWGN
from sig.Demod import Demod
from constellation.Constellation import Constellation


# *********************************** Encoding ***********************************************

# Create the message and read in the data from the file
message = Message(fn='test_file')

# compress the message, n is the block length
prior_len = len(message.data)
message.huffman_compress(n=8)

print(f"Compressed data from {prior_len} to {len(message.data)} bits."
      f" {100 - (len(message.data)/prior_len*100):.2f}% reduction")

# Pack the preamble and the compressed data
BLOCKSIZE = 64
message.pack_message(blocksize=BLOCKSIZE)

# Apply CRC-16 encoding
prior_len = len(message.data)
crc_args = {"polynomial": "16"}
message.encode(encoder="crc", blocksize=BLOCKSIZE, **crc_args)

print(f"Added error correction (out, in) = {message.data.size, prior_len}")

# interleave
message.block_interleave(n=8)

# Now we need to convert it to symbols
message.symbolise(bits_per_symbol=2)


# *********************************** Modulation ***********************************************

# Now we can modulate, lets use QPSK
F = 100e3
SYMBOL_RATE = 20e3      # Symbols per second
DUR = len(message.data) / SYMBOL_RATE    # Message duration (in seconds)
Fs = SYMBOL_RATE
while Fs <= F * 2:
    Fs += SYMBOL_RATE

s = Mod(message=message.data, f=F, fs=Fs, duration=DUR, amplitude=1)

s.QPSK()

# Baseband the sig
s.baseband()


# ****************************** Noise Sim ****************************************
# Create some noise and add it into the signal
noise = AWGN(n=len(s.samples), power=0.02)
# Add it in
s.samples = s.samples + noise


# ***************************** Demod ********************************************
d = Demod(fs=F*2, filename=None)
d.samples = s.samples.copy()
sent_message = message.data.copy()
del s, message  # Cleanup
d.dur = len(d.samples) / d.fs   # Usually this would be set if you read the file in
d.normalise_pwr()   # This too

# We need to get it to one sample per symbol
# just from looking at the samples I can see that it is 11 samples per symbol
# TODO:
#  UPSAMPLE FIRST
sps = 11
mags = []

# For every possible phase offset, resample the data and then calculate it's average magnitude
for i in range(int(sps)):
    test_division = d.samples[i::int(sps)]
    mag = np.mean(np.abs(test_division))
    mags.append(mag)

# Get the index of the biggest magnitude
n = mags.index(max(mags))
d.samples = d.samples[n::int(sps)]
# retime because we have dropped a bunch of samples
d.retime()

# Do the demod
c = Constellation(M=4)
c.square()
symbols = d.QAM(c)

if np.all(sent_message == symbols):
    print("Symbols successfully demodulated")
del d

# ******************************** Decode ************************************
# So now we have to do the reverse process to decode.
#   1 - map from symbols back to bits
#   2 - De-interleave the bits
#   3 - Apply the FEC to the data
#   4 - Interpret the header to get the compression dictionary out
#   5 - De-compress the data using that info
#   6 - Print the message!

# 1 - Map from symbols to bits
# Because we sent the message we know the bit map
message = Message(fn=None, data=symbols)
message.desymbolise(bits_per_symbol=2)

# 2 - De-interleave the bits
message.block_interleave(n=8, deinterleave=True)    # we know it's 8

# 3 - Apply the Fec to the data, we know that we're using crc 16 and the blocks were size 80 (including the fec)
syndromes = message.encode(encoder='crc', blocksize=80, decode=True, **crc_args)

# Check the fec!
if np.all(syndromes == 0):
    print("No errors detected!")

# Seems all good, we can drop the fec (which we know is the last 16 bits of each block)
message.data = message.data[:, 0:-16]

# 4 - Interpret the header to get the decompression data
message.decode_preamble()

# 5 - De-compress the data
# Added 62 bits of padding to the data. We know this already but it in the future it will be put into the header
original_message = message.apply_decompression()

# 6 - print the message
# Turn it back into ascii
text = ""
for i in range(0, len(original_message), 8):
    text += chr(int(original_message[i:i+8], 2))

print(text)
