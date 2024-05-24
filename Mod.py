import os
import numpy as np
from time import time
from Constellation import Constellation
import Plot


class Mod:
    def __init__(self, f, fs, message, duration=1, amplitude=1):
        self.f = f              # Frequency of the signal
        self.fs = fs            # Sampling frequency/rate
        self.dur = duration     # In seconds
        self.amp = amplitude    # Amplitude
        self.message = message      # Message as an array of symbols, np.array([0, 1, 2, 3, 0, 1, 2, 2]) etc
        self.M = len(np.unique(self.message))   # The number of symbols
        self.sps = int(self.dur * self.fs / len(self.message))  # Samples per symbol

        self.samples = None
        self.fsk = False        # Flag for FSK because it behaves a bit differently

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
        t = 1/self.fs * np.arange(self.dur * self.fs)

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

    def ASK(self):
        """
        samples = A * e^i(*2pi*f*t + theta)
        With ASK we are simply applying the A in the above equation
        """
        amp_mod_z = np.repeat(self.message, self.sps)       # repeat each of the element of the message, sps times
        amp_mod_z += 1  # Add 1 so amplitude is never 0 (I think this is necessary but it might not be)
        amp_mod_z = amp_mod_z / max(amp_mod_z)      # Scale it

        self.samples = self.create_samples(freq=self.f, amp=amp_mod_z)

    def FSK(self):
        """
        samples = A * e^i(2pi*f*t + theta)

        FSK creates new samples where the f in the equation above is modulated by the symbols. Makes no attempt
        to fix phase shifts
        """
        freqs = self.message + 1      # Add one to avoid zero frequency
        freqs = freqs / max(freqs)   # Normalize
        freqs = freqs * self.f
        f_mod_z = np.repeat(freqs, self.sps)

        z = self.create_samples(freq=f_mod_z, theta=0, amp=1)

        self.samples = z.astype(np.complex64)
        self.fsk = True

    def PSK(self):
        """
        samples = A * e^i(2pi*f*t + theta)

        PSK creates new samples where the theta in the equation above is modulated by the symbols.
        """
        phases = np.pi + np.pi * (self.message / max(self.message))   # Create a different phase shift for each symbol
        p_mod_z = np.repeat(phases, self.sps)

        z = self.create_samples(freq=self.f, theta=p_mod_z)
        self.samples = z.astype(np.complex64)

    def PSM(self, symbol_gaps, xmit_dur):
        """
        Pulse-spacing modulation, also pulse position modulation. Modulates signals by changing the time difference
        between pulses.
        :param: symbol_gaps: The number of samples to be left between pulses for each symbol
        """
        # This creates more samples than we need
        xmit_samps = self.create_samples(freq=self.f)
        samps_needed = xmit_dur * self.fs
        xmit_samps = xmit_samps[0:samps_needed]

        output = xmit_samps.copy()

        for symbol in self.message:
            phrase = np.repeat(symbol, symbol_gaps[symbol]) # repeat for the appropriate length
            phrase = phrase * 0+0j  # Make it complex and zero it
            np.concatenate([output, phrase, xmit_samps])    # attach the pulse

        self.samples = output

    def QPSK(self):
        """
         samples = A * e^i(2pi*f*t + theta)
         from euler, e^i(x) = cos(x) + isin(x)

         so:
          samples = A * cos(2 * pi * f * t + theta) + A * i * sin(2 * pi * f * t + theta)

          if QPSK we manipulate phase, i.e we encode our data into the theta term of the above equation
        """
        M = len(np.unique(self.message))    # The number of symbols

        # Convert the message symbols to M radian phase offsets with a pi/M bias from zero
        # i.e. if we had 4 symbols make them 45, 135, 225, 315 degree phase offsets (1/4pi, 3/4pi, 5/4pi, 7/4pi)
        symbols = self.message * 2 * np.pi / M + np.pi/ M
        message = np.repeat(symbols, self.sps)

        z = self.create_samples(freq=self.f, theta=message)

        self.samples = z.astype(np.complex64)

    def QAM(self, type="square"):
        """
        It's QAM! Creates the most ideal square QAM possible for the number of symbols supplied and the type
        """
        # Create the constellation map - a lookup table of values that will be indexed by the message values
        c = Constellation(M=self.M)

        if type == "square":
            c.square()
        elif type == "sunflower":
            c.sunflower()
        elif type == "star":
            c.star()
        elif type == "square_offset":
            c.square_offset()
        else:
            raise ValueError("Incorrect Constellation type")

        c.prune()
        c.normalise()

        message = np.repeat(self.message, self.sps)

        offsets = c.map[message]      # Index the map by the symbols

        z = self.create_samples(freq=self.f, theta=np.angle(offsets), amp=np.abs(offsets))

        self.samples = z

    def CPFSK(self):
        """
        samples = A * e^i(2pi*f*t + theta)

        Continuous phase frequency shift keying. Uses a phase offset vector to minimise phase jumps arising
        from frequency shift keying, which makes it more spectrally efficient.

        resource:
        https://dsp.stackexchange.com/questions/80768/fsk-modulation-with-python

        """

        # TODO
        #   Change phase vector so it is always < 2pi
        #   To speed up computation, can precompute the phase offset per symbol
        #   Make FM so it is around the carrier signal
        #
        freqs = self.message + 1      # Add one to avoid zero frequency
        freqs = freqs / max(freqs)   # Normalize
        freqs = freqs * self.f
        f_mod_z = np.repeat(freqs, self.sps)     # FSK message

        # Cumulative phase offset
        delta_phi = 2.0 * f_mod_z * np.pi / self.fs    # Change in phase at every timestep (in radians per timestep)
        phi = np.cumsum(delta_phi)              # Add up the changes in phase

        z = self.amp * np.exp(1j * phi)  # creates sinusoid theta phase shift
        z = np.array(z)
        z = z.astype(np.complex64)

        self.samples = z
        self.fsk = True

    def baseband(self):
        """
        Move the signal to baseband (0 frequency)
        """
        if self.fsk:
            freq = (np.arange(self.M) + 1) / self.M
            freq = np.mean(freq * self.f)

            f = -1 * self.f + freq
            offset = self.create_samples(freq=-1*freq)
        else:
            offset = self.create_samples(freq=-1*self.f)
        self.samples = self.samples * offset
        self.f = 0

    def phase_offset(self, angle=40):
        """
        Adds a phase offset of x degrees to the signal
        """
        # degrees to radians
        phase_offset = angle*np.pi / 180
        phase_offset = self.create_samples(freq=0, theta=int(phase_offset))

        self.samples = self.samples * phase_offset

    def freq_offset(self, freq=1000):
        """
        Moves the signal up by some amount of Hz
        """
        freq_offset = self.create_samples(freq=freq, theta=0, amp=1)

        self.samples = self.samples * freq_offset
        self.f += freq
        self.fsk = False    # stupid fsk...



    def specgram(self, nfft=1024):
        # Nfft shouldn't be bigger than the samples
        if nfft >= len(self.samples):
            nfft = int(len(self.samples)/4)

        kwargs = {"type": "specgram",
                "nfft": nfft,
                "fs": self.fs,
                "title": f"Specgram at Baseband (NFFT={nfft})"}

        Plot.plot(self.samples, **kwargs)

    def psd(self, nfft=1024):
        kwargs = {"type": "psd",
                  "nfft": nfft,
                  "fs": self.fs,
                  "title": f"PSD at Baseband (NFFT={nfft})"}
        Plot.plot(self.samples, **kwargs)

    def iq(self):
        kwargs = {"type": "iq",
                  "title": "IQ Scatter"}

        Plot.plot(self.samples, **kwargs)

    def fft(self, nfft=1024):
        kwargs = {"type": "fft",
                  "title": "FFT of Signal",
                  "fs": self.fs,
                  "nfft":nfft}
        Plot.plot(self.samples, **kwargs)

    def time(self, n=0):
        t = 1 / self.fs * np.arange(self.dur * self.fs)
        t = t[0:len(self.samples)]

        kwargs = {"type": "time",
                  "t": t,
                  "title": "Time Domain",
                  "n": n}

        Plot.plot(self.samples, **kwargs)

    def save(self, fn=None, path=None):
        # If there is no path provided then provide one
        if not path:
            path = os.getcwd()
            path = path.split("\\")     # I don't think this will work on non-windows?
            dspy_index = path.index("DSPy")
            path = '\\'.join(path[0:dspy_index+1]) + "\\" + "signals" + "\\"

        # If no file name make one
        if not fn:
            fn = f"Sig_f={self.f}_fs={self.fs}_dur={self.dur}_{int(time())}"

        save_string = path + fn
        self.samples.tofile(save_string)





