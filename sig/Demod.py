import numpy as np
from plot import Plot
from matplotlib import pyplot as plt
from scipy import signal
from sig.Signal import Signal

"""
Class with demod functions, ideally this is automated but very much still a work in progress

"""

# TODO
#  Update Readme
#  Add license
#  Add pyproject.toml
#  Add Multiple access functionality


class Demod(Signal):
    def __init__(self, fs, filename=None):
        self.fn = filename

        if filename:
            samples = self.read_file()
        else:
            samples = np.array([])

        if fs and filename:
            if (len(samples) > 0):
                dur = len(samples) / fs
        else:
            dur = None

        super().__init__(f=None, fs=fs, message=[], duration=dur, amplitude=1)
        self.samples = samples
        self.samplerate = None

    def read_file(self, folder=""):
        file = folder + self.fn
        samples = np.fromfile(file, np.complex64)
        return samples

    def normalise_pwr(self):
        """
        normalises samples to be between 1 and 0
        """
        max_real = max(abs(self.samples.real))
        max_imag = max(abs(self.samples.imag))

        max_val = max(max_imag, max_real)
        self.samples = (self.samples / max_val)

    def trim_by_power(self, padding=0, std_cut=1.5, n=10):
        """
        Trims the sig by looking at the power envelope. Adds a slight padding to each end
        :param padding: N sample padding either side of the cut
        :param std_cut: Decide that the sig begins this many stds from the mean
        :param n: The number for the moving average
        """
        # If we do a moving average over the abs value of the samples (the abs value being the power!) we get a suuuper
        # clear spike where the sig begins
        av = np.convolve(np.abs(self.samples), np.ones(n), 'valid') / n
        sdev = np.std(av)

        index = np.arange(len(av))[abs(av) > std_cut*sdev]

        # first is the turn on (hopefully) last is turn off (hopefully)
        first_ind = index[0] - int(padding)
        if first_ind < 0:
            first_ind = 0

        last_ind = index[-1] + int(padding)

        self.samples = self.samples[first_ind:last_ind]
        self.dur = len(self.samples)/self.fs

    def detect_params(self):
        """
        detects the parameters of the sample if it follows the GQRX naming convention
        """
        if "_" in self.fn:
            params = self.fn.split("_")
        else:
            raise ValueError("Capture does not appear to be in gqrx format")

        if params[0] != "gqrx":
            raise ValueError("Capture does not appear to be in gqrx format")

        else:
            try:
                self.fs = int(params[3])
                self.f = int(params[4])
            except:
                raise ValueError("Capture does not appear to be in gqrx format")

    def quadrature_demod(self):

        delayed = np.conj(self.samples[1:])
        self.samples = delayed * self.samples[:-1]  # Drops the last sample, this may be bad
        self.samples = np.angle(self.samples)

    def exponentiate(self, order=4):
        """
        Raises a sig to the nth power to find the frequency offset and the likely samples per symbol
        """
        # copy the samples and raise to the order
        samps = self.samples.copy()
        samps = samps**order

        # Take the fft to find the freq and sps spikes
        ffts = np.fft.fftshift(np.abs(np.fft.fft(samps)))
        axis = np.arange(self.fs / -2, self.fs / 2, self.fs / len(ffts))

        # Get indices of the 1 largest element, which will be the freq spike
        largest_inds = np.argpartition(ffts, -1)[-1:]
        largest_val = axis[largest_inds]

        # The frequency offset
        freq = largest_val / order
        freq = round(freq[0]) # Make an int

        if len(axis) > len(ffts):
            axis = axis[0:len(ffts)]

        plt.plot(axis, ffts)

        return freq

    def QAM(self, c):
        """
        Converts the samples in memory to the closest symbols found in a given constellation plot and returns the
        output
        """
        symbols = np.arange(len(c.map))
        out = []
        for sample in self.samples:
            index = (np.abs(c.map - sample)).argmin()
            out.append(symbols[index])

        return out



