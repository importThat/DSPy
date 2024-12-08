<picture align="center">
  <img alt="dsproc logo" src="dsproc_logo.png">
</picture>

------------------

# dsproc: a powerful digital signals processing toolkit

## What is it?

**dsproc** is a Python package that enables the analysis and processing of digital radio
signals using an intuitive and approachable framework. It supports end to end digital communcations,
 which gives users the ability to fully encode and modulate data into radio waves of all types.

<picture align="center">
  <img alt="dsproc logo" src="e2e_digital_comms.png">
</picture>

------------------

## Main Features
Here are some of the things you can do with dsproc:
    
- Perform end to end digital signal processing! Compress, randomise, error correct, 
  interleave and then modulate data into a complex wave ready for transmitting through
  your software defined radio (SDR) of choice.
- Supports a variety of modulation and demodulation types such as ASK, FSK, QAM, MFSK,
  and PSK using symbol sets of arbitrary size.
- Create custom QAM constellations using a simple click gui.
- Use clustering to aid in automatic demodulation of FSK, ASK and QAM signals.
- Create spectrum art by converting images to waves and transmit them via SDR!

## Minimal example

```python
# Import the library
import dsproc

# Read in a file
message = dsproc.Message(fn="my_picture.png")
# The file is current stored as bits. We could leave it that way and transmit one bit per symbol,
# but instead lets convert the message to symbols. Here I will convert to symbols using 2 bits per symbol,
# this gives us four possible symbols, e.g.

#     Symbol  |  Bits    
#   ---------------------
#       0     |    00
#       1     |    01
#       2     |    10
#       3     |    11

# Convert to symbols
message.symbolise(bits_per_symbol=2)

# Now we want to create a modulation object which we will use to write our bits to a signal
# fs = the sampling frequency, which is how many times per second we are creating samples for our radio wave
# sps = samples per symbol, how many samples we will allow per symbol transmitted. minimum is 1 but it's best to do
# at least 4.

radio_wave = dsproc.Mod(fs=10000, message=message.data, sps=8)
# Apply Amplitude Shift Keying (aka amplitude modulation) to our radio wave. This encodes the radio wave so each 
# symbol is represented by a unique amplitude
radio_wave.ASK()

# The wave can be viewed with various plotting methods
radio_wave.amp_view()
radio_wave.time()

# Save the wave as 64 bit complex numbers. This file can then be used directly with GNU radio or Software defined radio
# software. It's best to save the file with the sample rate (fs) in the name because you will need this number for 
# transmitting this wave.
radio_wave.save_wave("my_picture_wave_fs=10000_sps=8")
```

## Installation
To install dsproc and it's dependencies I recommend using the pip installer:
```commandline
pip install dsproc
```


## Dependencies
- [NumPy - Adds support for multi-dimensional arrays](https://www.numpy.org)
- [SciPy - Filter and clustering functions](https://scipy.org/)
- [matplotlib - Plotting](https://matplotlib.org/)

## Testing
I use hatch, 'hatch test --doctest-modules --ignore="Examples/*', to run all the tests and the doctests while ignoring
the Examples folder.
Otherwise, tests are in the tests folder, and many functions have doctests.





