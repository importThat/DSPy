import numpy as np
from dsproc.sig import plot
from time import time
import os
from scipy.io.wavfile import write
from scipy import signal


class Signal:
    def __init__(self, f, fs, message, duration, amplitude):
        self.f = f
        self.fs = fs
        self.message = message
        self.dur = duration
        self.amp = amplitude
        self.M = len(np.unique(self.message))  # The number of symbols

        if len(message) > 0:
            self.sps = int(self.dur * self.fs / len(self.message))  # Samples per symbol
        else:
            self.sps = None

        self.samples = None
        self.fsk = False

    def create_samples(self, freq, theta=0, amp=1):
        """
        Signal = A * np.cos(2 * np.pi * f * t + theta) + A * 1j * np.sin(2 * np.pi * f * t + theta)
        where:
            A = amplitude
            f = frequency
            t = a time vector of the times samples are taken
            theta = a phase offset

            (j is the programming term for i, the complex number)
        """
        t = 1 / self.fs * np.arange(self.dur * self.fs)
        z = np.ndarray([])

        # If we're supplying a frequency vector (for FSK) then the length might not be compatible with t
        if type(freq) == np.ndarray:
            t = t[0:len(freq)]

        # Same for phase
        if type(theta) == np.ndarray:
            t = t[0:len(theta)]

        # same for amplitude
        if type(amp) == np.ndarray:
            t = t[0:len(amp)]

        # If there's no frequency (for example we just want to do a phase offset)
        if freq is int:
            if freq == 0:
                z = amp * np.cos(theta) + 1j * amp * np.sin(theta)

        else:
            angle = 2 * np.pi * freq * t + theta

            # equivalent to z = amp * np.exp(1j * (2 * np.pi * freq * t + theta))
            z = amp * np.cos(angle) + 1j * amp * np.sin(angle)

        z = z.astype(np.complex64)

        return z

    def baseband(self):
        """
        Move the sig to baseband (0 frequency)
        """
        if not self.f:
            raise ValueError("Cannot baseband signal because the center frequency is unknown. Set the attribute 'f' to "
                             "some integer value")
        if self.fsk:
            freq = (np.arange(self.M) + 1) / self.M
            freq = np.mean(freq * self.f)

            f = -1 * self.f + freq
            offset = self.create_samples(freq=-1*freq)
        else:
            offset = self.create_samples(freq=-1*self.f)
        self.samples = self.samples * offset
        self.f = 0

    def retime(self):
        """
        Recalculates the duration of the message
        """
        self.dur = len(self.samples)/self.fs

    def phase_offset(self, angle=40):
        """
        Adds a phase offset of x degrees to the sig
        """
        # degrees to radians
        phase_offset = angle*np.pi / 180
        z = 1 * np.cos(phase_offset) + 1j * np.sin(phase_offset)

        self.samples = self.samples * z

    def freq_offset(self, freq=1000):
        """
        Moves the sig up by some amount of Hz
        """
        freq_offset = self.create_samples(freq=freq, theta=0, amp=1)

        self.samples = self.samples * freq_offset
        if self.f:
            self.f += freq
        else:
            self.f = freq

        self.fsk = False    # stupid fsk...

    def resample(self, up=16, down=1):
        """
        A simple wrapper for scipy's resample
        """
        self.samples = signal.resample_poly(self.samples, up, down)

    # ***********************************                    ************************************
    # ************************************ Plotting Functions ************************************
    # *************************************                    ************************************

    def specgram(self, nfft=1024):
        # Nfft shouldn't be bigger than the samples
        if nfft >= len(self.samples):
            nfft = int(len(self.samples)/4)

        kwargs = {"type": "specgram",
                "nfft": nfft,
                "fs": self.fs,
                "title": f"Specgram at Baseband (NFFT={nfft})"}

        plot.plot(self.samples, **kwargs)

    def psd(self, nfft=1024):
        kwargs = {"type": "psd",
                  "nfft": nfft,
                  "fs": self.fs,
                  "title": f"PSD at Baseband (NFFT={nfft})"}
        plot.plot(self.samples, **kwargs)

    def iq(self):
        kwargs = {"type": "iq",
                  "title": "IQ Scatter"}

        plot.plot(self.samples, **kwargs)

    def fft(self, nfft=1024):
        kwargs = {"type": "fft",
                  "title": "FFT of Signal",
                  "fs": self.fs,
                  "nfft":nfft}
        plot.plot(self.samples, **kwargs)

    def time(self, n=0):
        t = 1 / self.fs * np.arange(self.dur * self.fs)
        t = t[0:len(self.samples)]

        kwargs = {"type": "time",
                  "t": t,
                  "title": "Time Domain",
                  "n": n}

        plot.plot(self.samples, **kwargs)

    def save(self, fn=None, path=None, wav=False):
        # TODO
        #  Make this path better
        # If there is no path provided then provide one
        if not path:
            path = os.getcwd()
            path = path.split("\\")     # I don't think this will work on non-windows?
            dspy_index = path.index("dsproc")
            path = '\\'.join(path[0:dspy_index+1]) + "\\" + "modulations" + "\\"

        # If no file name make one
        if not fn:
            fn = f"Sig_f={self.f}_fs={self.fs}_dur={self.dur}_{int(time())}"

        save_string = path + fn

        # If we're saving it as a wav
        if wav:
            if self.f == 0:
                self.freq_offset(800)

            audio = self.samples.real
            # Target sample rate
            sample_rate = 44100
            audio = signal.resample_poly(audio, up=sample_rate, down=self.fs)

            write(fn+".wave", sample_rate, audio.astype(np.float32))

            self.baseband()

        else:
            self.samples.tofile(save_string)



