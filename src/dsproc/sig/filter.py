from scipy import signal
from matplotlib import pyplot as plt
import numpy as np


class Filter:
    def __init__(self, fs, num_taps=101):
        self.n_taps = num_taps
        self.fs = fs
        self.taps = []

    def FIR(self, width):
        """
        Finite infinite response filter
        :return:
        """
        self.taps = signal.firwin(self.n_taps, width, nyq=self.fs / 2)
        self.taps = self.taps + 1j * self.taps
        self.taps = self.taps.astype(np.complex64)

    def ir(self):
        """
        Plots the impulse response of the filter
        """
        if len(self.taps) <= 0:
            raise Warning("No taps detected")

        plt.plot(self.taps, '.-')
        plt.show()

    def apply(self, signal: np.ndarray, f_shift: int=0):
        """
        Applies the filter to a sig

        :param signal: a complex array
        :param freq_shift: The amount which the center-point of the filter should be shifted
        :return: filtered sig
        """

        if f_shift != 0:
            Ts = 1.0 / self.fs  # sample period
            t = np.arange(0.0, Ts * len(self.taps), Ts)  # time vector. args are (start, stop, step)
            t = t[0:len(self.taps)]  # Ensure it's not longer than the taps
            exponential = np.exp(2j * np.pi * f_shift * t)  # this is essentially a complex sine wave
            exponential = exponential.astype(np.complex64)

            taps = self.taps * exponential  # do the shift

        else:
            taps = self.taps

        return np.convolve(signal, taps, mode="same")









