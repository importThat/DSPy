import numpy as np
from matplotlib import pyplot as plt
from time import time
from Constellation import Constellation


class Signal:
    def __init__(self, f, fs, message, duration=1, amplitude=1):
        self.f = f              # Frequency of the signal
        self.fs = fs            # Sampling frequency/rate
        self.dur = duration     # In seconds
        self.amp = amplitude    # Amplitude
        self.message = message      # Message as an array of symbols, np.array([0, 1, 2, 3, 0, 1, 2, 2]) etc
        self.M = len(np.unique(self.message))   # The number of symbols
        self.sps = int(self.dur * self.fs / len(self.message))  # Samples per symbol

        self.samples = None

    def create_samples(self, freq, theta=0, amp=1):
        """
        Signal = A * np.cos(2 * np.pi * f * t + theta) + A * 1j * np.sin(2 * np.pi * f * t + theta)
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

        z = amp * np.cos(2 * np.pi * freq * t + theta) + amp * 1j * np.sin(2 * np.pi * freq * t + theta)
        z = np.array(z)
        z = z.astype(np.complex64)

        return z

    def ASK(self):
        """
        samples = A * e^i(*2pi*f*t + theta)
        With ASK we are simply applying the A in the above equation
        """
        amp_mod_z = np.repeat(self.message, self.sps)       # repeat each of the element of the message, sps times
        amp_mod_z += 1  # Add 1 so amplitude is never 0 (I think this is necessary but it might not be)

        self.samples = self.samples[0:len(amp_mod_z)]   # Trim so they are the same length
        self.samples = self.samples * amp_mod_z     # Apply the modulation
        self.samples = self.samples.astype(np.complex64)

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

        z = self.create_samples(freq=f_mod_z, theta=0)

        self.samples = z
        self.samples = self.samples.astype(np.complex64)

    def PSK(self):
        """
        samples = A * e^i(2pi*f*t + theta)

        PSK creates new samples where the theta in the equation above is modulated by the symbols.
        """
        phases = np.pi + np.pi * (self.message / max(self.message))   # Create a different phase shift for each symbol
        p_mod_z = np.repeat(phases, self.sps)

        z = self.create_samples(freq=self.f, theta=p_mod_z)
        self.samples = z.astype(np.complex64)

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
        It's QAM! Creates the most ideal square QAM possible for the number of symbols supplied
        """
        # Create the constellation map - a lookup table of values that will be indexed by the message values
        c = Constellation(M=self.M)

        if type == "square":
            c.square()
        elif type == "sunflower":
            c.sunflower()
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

        # ** TO DO **
        #   Change phase vector so it is always < 2pi
        #   To speed up computation, can precompute the phase offset per symbol
        # ***
        freqs = self.message + 1      # Add one to avoid zero frequency
        freqs = freqs / max(freqs)   # Normalize
        freqs = freqs * self.f
        f_mod_z = np.repeat(freqs, self.sps)     # FSK message

        # Cumulative phase offset
        delta_phi = 2.0 * f_mod_z * np.pi / self.fs    # Change in phase at every timestep (in radians per timestep)
        phi = np.cumsum(delta_phi)              # Add up the changes in phase

        z = self.amp * np.exp(1j * phi)  # creates sinusoid at f Hz with theta phase shift
        z = np.array(z)
        z = z.astype(np.complex64)

        self.samples = z

    def plot(self, type, nfft=1024):

        if type == "specgram":
            plt.specgram(self.samples, NFFT=nfft, Fs=self.fs)

        elif type == 'psd':
            plt.psd(self.samples, NFFT=nfft, Fs=self.fs)

        elif type == "scatter":
            plt.scatter(self.samples.real[0:1000000], self.samples.imag[0:1000000])

        elif type == 'constellation':
            #sps = int(self.dur * self.fs / len(self.message))

            intermediate = self.create_samples(freq=-100)
            base_band = self.samples * intermediate
            plt.scatter(base_band.real, base_band.imag)
            plt.grid(True)
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
            plt.show()

        elif type == "fft":
            S = np.fft.fftshift(np.fft.fft(self.samples))
            S_mag = np.abs(S)
            f_axis = np.arange(self.fs/-2, self.fs/2, self.fs/len(self.samples))
            if len(f_axis) > len(S_mag):
                f_axis = f_axis[0:len(S_mag)]

            plt.plot(f_axis, S_mag)
            plt.show()

        else:
            raise ValueError("type must be one of 'specgram', 'psd', 'scatter', 'fft', 'constellation'")

    def save(self, fn=None):
        if not fn:
            fn = f"Sig_f={self.f}_fs={self.fs}_dur={self.dur}_{int(time())}"
        self.samples.tofile(f"signals\\{fn}")





