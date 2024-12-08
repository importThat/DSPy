from time import time
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from .plot import plot


class Signal:
    """
    Main class for the modulation and demodulation of data into radio waves. Contains functionality for writing
    waves and performing operations on the waveform such as frequency shifting, phase shifting, and resampling.
    """
    def __init__(self, fs: int, message: np.ndarray | list,
                 sps: int = 2,
                 amplitude: float = 1.0,
                 f: int = 100):
        """
        Initialise the signal object.

        :param fs: Sampling frequency. How often samples will be created for the wave. A wave with a sampling rate of
            100Hz would have 100 samples per second.
        :param message: A numpy array of ints containing the message symbols which will be written into a wave.
        :param sps: How many samples to generate per symbol. Typical values are between 8 and 20. Lowering the samples
            per symbol will increase the data rate at the expense of making it more susceptible to errors.
        :param amplitude: The (approximate) max amplitude of the wave, typically 1.
        :param f: The centre frequency of the signal. Should be somewhere between -fs/2 and fs/2.

        >>> s = Signal(10000, message=np.array([1, 0, 1, 1, 1]), sps=32, f=3000)    # instance a Signal object
        >>> (s.fs, s.message, s.sps, s.f)   # Display the parameters
        (10000, array([1, 0, 1, 1, 1]), 32, 3000)

        """
        # Sampling frequency
        self.fs = fs
        # Message as an array of symbols
        self.message = np.array(message)
        # Samples per symbol
        self.sps = sps
        # Intermediate frequency
        self.f = f
        # Mas amplitude of signal
        self.amp = amplitude

        # Container for the samples which hold the radio wave
        self.samples = np.array([], dtype=np.complex64)


    @property
    def n_samples(self) -> int:
        """
        Length of the transmit wave in samples. Updates when the wave is modulated

        >>> s = Signal(10000, message=np.array([1, 0, 1, 1, 1]), sps=32, f=3000)
        >>> s.n_samples
        160

        """
        # This first clause occurs if we are coming from the demodulation side, because we will have loaded in
        # samples but not a message
        # TODO: Consider overwriting this instead in the child classes
        if len(self.message) == 0:
            if len(self.samples) == 0:  # It's nice to be able to init the demod class without any data sometimes
                return 0

        if len(self.samples) == 0:
            return self.sps * len(self.message)

        return len(self.samples)

    @property
    def dur(self) -> float:
        """
        The duration of the signal in seconds.

        >>> s = Signal(10000, message=np.array([1, 0, 1, 1, 1]), sps=32, f=3000)
        >>> s.dur
        0.016
        """
        return self.n_samples / self.fs

    @property
    def M(self) -> int:
        """
        The number of unique symbols in the message, which is how many different levels the modulation scheme
        should have.

        >>> s = Signal(10000, message=np.array([1, 0, 1, 1, 1]), sps=32, f=3000)
        >>> s.M # Two symbols, 0 and 1
        2
        """
        return len(np.unique(self.message))  # The number of symbols

    @property
    def t(self) -> np.ndarray:
        """
        A 1d array which contains when the samples occur. This is used to construct the wave.

        >>> s = Signal(10000, message=np.array([1, 0, 1, 1, 1]), sps=32, f=3000)
        >>> s.t[0:5]
        array([0.    , 0.0001, 0.0002, 0.0003, 0.0004])

        """
        return 1 / self.fs * np.arange(self.n_samples)

    def create_samples(self, freq: int | np.ndarray,
                       theta: int | np.ndarray = 0,
                       amp: float | np.ndarray = 1) -> np.ndarray:
        """
        Creates the samples of a complex wave using the following formula:

        Signal = A * np.cos(2 * PI * f * t + theta) + A * 1j * np.sin(2 * PI * f * t + theta)

        where:
            A = amplitude
            f = frequency
            t = a time vector of the times samples are taken (self.t)
            theta = a phase offset
            and j is the programming term for i, the complex number

        :param freq: The frequency value/s. Int or np array
        :param theta: The phase value/s in radians. Int or np array
        :param amp: The amplitude values. Float or np array

        # TODO: Might be best if this lived in the Mod class
        >>> s = Signal(10000, message=np.array([1, 0, 2]), sps=3, f=3000)
        >>> # Make an array of amplitudes to do amplitude shift keying, Add one to the message to avoid a zero amplitude
        >>> # signal, then scale to between 1 and 0
        >>> amplitudes  = s.message + 1
        >>> amplitudes = amplitudes / max(amplitudes)
        >>> # Repeat for every sample
        >>> amplitudes = np.repeat(amplitudes, s.sps)
        >>> samples = s.create_samples(s.f, 0, amplitudes)  # Create the samples
        >>> print(np.round(samples, 5)) # Print to avoid doctest formatting
        [ 0.66667+0.j      -0.20601+0.63404j -0.53934-0.39186j  0.26967-0.19593j
          0.10301+0.31702j -0.33333+0.j       0.30902-0.95106j  0.80902+0.58779j
         -0.80902+0.58779j]

        """
        z = np.ndarray([])
        # Create a copy of t, because we may need to alter it
        t = self.t.copy()

        # If we're supplying a frequency vector (for FSK) then the length might not be compatible with t
        if isinstance(freq, np.ndarray):
            t = t[0:len(freq)]

        # Same for phase
        if isinstance(theta, np.ndarray):
            t = t[0:len(theta)]

        # same for amplitude
        if isinstance(amp, np.ndarray):
            t = t[0:len(amp)]

        # Frequency has to be a non-zero value
        if isinstance(freq, int):
            if freq == 0:
                raise ValueError("Cannot make a zero frequency wave, set frequency to some non zero value")
        elif isinstance(freq, np.ndarray):
            if np.all(freq == 0):
                raise ValueError("Cannot make a zero frequency wave, set frequency to some non zero value")

        angle = 2 * np.pi * freq * t + theta

        # equivalent to z = amp * np.exp(1j * (2 * np.pi * freq * t + theta))
        # but this way is faster
        z = amp * np.cos(angle) + 1j * amp * np.sin(angle)

        # If the dtype has changed due to different input dtypes
        if z.dtype != np.complex64:
            z = z.astype(np.complex64)

        return z

    def baseband(self) -> None:
        """
        Basebands the signal by shifting the centre frequency to zero. This function works by creating a wave of
        appropriate length with the centre frequency of -1*self.f, and multiplying the signal by this wave.

        >>> s = Signal(10000, message=np.array([1, 0, 1, 0, 1, 1]), sps=16, f=3000)
        >>> # Make an array of amplitudes to do amplitude shift keying, Add one to the message to avoid a zero amplitude
        >>> # signal, then scale to between 1 and 0
        >>> amplitudes  = s.message + 1
        >>> amplitudes = amplitudes / max(amplitudes)
        >>> # Repeat for every sample
        >>> amplitudes = np.repeat(amplitudes, s.sps)
        >>> s.samples = s.create_samples(s.f, 0, amplitudes)  # Create the samples
        >>> s.baseband()    # Baseband it
        >>> s.f
        0
        """
        if not self.f:
            raise ValueError("Cannot baseband signal because the center frequency is unknown. Set the attribute 'f' to "
                             "some integer value")

        offset = self.create_samples(freq=-1*self.f)
        self.samples = self.samples * offset
        self.f = 0

    def normalise_amplitude(self) -> None:
        """
        normalises the amplitude of the signal to be between 0 and 1. This means that the real and imaginary parts
        will be between -1.0 and 1.0

        Example
        >>> s = Signal(10000, message=np.array([1, 0, 1, 0, 1, 1]), sps=16, f=3000)
        >>> # Put some samples in for demo purposes
        >>> s.samples = np.array([6.6+0.j, 5.3+3.9j, 2.0+6.3j, -2.0+6.3j, -5.39+3.9j])
        >>> s.normalise_amplitude()
        >>> np.round((max(s.samples.real), min(s.samples.real), max(s.samples.imag), max(s.samples.imag)), 2)
        array([ 1.  , -0.82,  0.95,  0.95])
        """
        max_real = max(abs(self.samples.real))
        max_imag = max(abs(self.samples.imag))

        max_val = max(max_imag, max_real)
        self.samples = (self.samples / max_val)

    def phase_offset(self, angle: int = 40) -> None:
        """
        Adds a phase offset of x degrees to the signal

        :param angle: Phase offset in degrees
        """
        # degrees to radians
        phase_offset = angle*np.pi / 180
        z = 1 * np.cos(phase_offset) + 1j * np.sin(phase_offset)

        self.samples = self.samples * z
        # The type gets coerced to complex 128, so lets fix that
        self.samples = self.samples.astype(np.complex64)

    def freq_offset(self, freq: int = 1000) -> None:
        """
        Moves the signal up by the given frequency. Adds the frequency offset to the 'f' attribute. Note that the
        frequency components of a signal are (kind of) bounded to between -fs/2 and fs/2, so shifting a signal up in
        frequency may cause it to wrap around to a negative frequency.

        :param freq: Int, the frequency to shift the signal by. Can be negative.
        """
        freq_offset = self.create_samples(freq=freq, theta=0, amp=1)
        freq_offset = freq_offset[0:len(self.samples)]  # In case it's a bit longer

        self.samples = self.samples * freq_offset
        if self.f:
            self.f += freq
        else:
            self.f = freq

    def resample(self, up: int = 16, down: int = 1) -> None:
        """
        A simple wrapper for scipy's resample. Resamples the signal a factor equal to up/down.

        :param up: Factor to upsample by
        :param down: Factor to downsample by

        Examples
        # Create a signal and upsample it
        >>> s = Signal(10000, message=np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1]), sps=10, f=3000)
        >>> s.samples = s.create_samples(freq=s.f, theta=0, amp=1)
        >>> # Upsample 10x, also changes the sample rate (fs) and the samples per symbol
        >>> s.resample(up=10, down=1)
        >>> s.fs, s.sps, len(s.samples)
        (100000, 100, 1000)

        # Create a signal and downsample it, can also use the decimate function
        >>> s = Signal(10000, message=np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1]), sps=70, f=3000)
        >>> s.samples = s.create_samples(freq=s.f, theta=0, amp=1)
        >>> # Upsample 10x, also changes the sample rate (fs) and the samples per symbol
        >>> s.resample(up=1, down=10)
        >>> s.fs, s.sps, len(s.samples)
        (1000, 7, 70)
        """
        self.samples = signal.resample_poly(self.samples, up, down)
        self.fs = int(self.fs * up/down)
        self.sps = int(self.sps * (up/down))

    def decimate(self, n: int, filter_order: int = 8, ftype: str = 'iir') -> None:
        """
        wrapper for scipy's decimate. First filters out high frequency components and then takes every nth sample

        :param n: The down sampling factor. If greater than 13 it is reccommended to decimate in stages
        :param filter_order: The order of the filter, defaults to 8 for iir
        :param ftype: The filter type, 'iir' (infinite impulse response) or 'fir' (finite impulse response)

        # Create a signal and decimate it 10x
        >>> s = Signal(10000, message=np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1]), sps=70, f=3000)
        >>> s.samples = s.create_samples(freq=s.f, theta=0, amp=1)
        >>> # Upsample 10x, also changes the sample rate (fs) and the samples per symbol
        >>> s.resample(up=1, down=10)
        >>> s.fs, s.sps, len(s.samples)
        (1000, 7, 70)
        """

        if n > 13:
            raise Warning("it is recommended to call decimate multiple times for downsampling factors greater than 13")

        self.samples = signal.decimate(self.samples, n, n=filter_order, ftype=ftype)
        self.fs = int(self.fs / n)
        self.sps = int(self.sps / n)

    def efficiency(self):
        """
        Calculates bandwidth efficiency of the signal. This is the total area under the curve of the signals fft plot
        """
        ft = np.abs(np.fft.fft(self.samples))
        integral = np.sum(ft)
        return integral

    def power_spill(self, band_low, band_high):
        """
        Returns the power of the signal that lies outside the given bands. Use this when looking at the harmonics being
        generated by a signal
        """
        sos = signal.butter(5, (band_low, band_high), "bandstop", fs=self.fs, output="sos")
        filtered = signal.sosfilt(sos, self.samples)
        filtered = filtered.astype(np.complex64)
        power = np.sum(np.abs(filtered))
        return power

    def butterworth_filter(self, frequencies: int | list | tuple,
                           filter_type: str,
                           order: int = 5) \
            -> tuple[np.ndarray, np.ndarray | float, np.ndarray]:
        """
        Generic wrapper for scipy's butterworth filter. Creates and applies a digital butterworth filter to the signal,
        Returns the filter taps for investigation of the filter parameters.

        :param frequencies: int, or tuple/list of length 2. The frequencies to apply the filter at
        :param filter_type: String, the direction the filter works in. One of
            ['lowpass', 'highpass', 'bandpass', 'bandstop']
        :param order: the order of the filter (how many taps it has)

        :return: the filter taps as an np.ndarray
        """
        filt = signal.butter(N=order, Wn=frequencies, btype=filter_type, analog=False, output='sos', fs=self.fs)
        self.samples = signal.sosfilt(filt, self.samples)
        self.samples = self.samples.astype(np.complex64)

        return filt

    def _gen_rrc(self, alpha: float, N: int):
        """
        TODO: Get this working correctly
        **** UNTESTED ****

        Code adapted from: https://github.com/veeresht/CommPy/blob/master/commpy/filters.py

        Generates a root raised cosine (RRC) filter (FIR) impulse response.

        Parameters
        ----------
        N : int
            Length of the filter in samples.

        alpha : float
            Roll off factor (Valid values are [0, 1]).

        Returns
        ---------

        rrc : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.
        """
        Ts = self.sps/self.fs
        T_delta = 1 / float(self.fs)
        sample_num = np.arange(N)

        rrc = np.zeros(N, dtype=float)

        for x in sample_num:
            t = (x - N / 2) * T_delta
            if t == 0.0:
                rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)

            elif alpha != 0 and t == Ts / (4 * alpha):
                rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                                                    (np.sin(np.pi / (4 * alpha)))) + (
                                                               (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))

            elif alpha != 0 and t == -Ts / (4 * alpha):
                rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                                                    (np.sin(np.pi / (4 * alpha)))) + (
                                                               (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))

            else:
                rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) + \
                            4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / \
                           (np.pi * t * (1 - (4 * alpha * t / Ts) * (4 * alpha * t / Ts)) / Ts)

        return rrc

    def rrc(self, alpha: float = 0.4, N: int = 0):
        """
        TODO: Get this working correctly
        **** UNTESTED ****

        Applies a root raised cosine filter to the signal

        Parameters
        ----------
        N : int
            Length of the filter in samples. If not specified uses the samples per symbol

        alpha : float
            Roll off factor (Valid values are [0, 1]).

        Returns
        ----------
        rcc_vals : 1d array of floats
            The impulse response of the rrc filter
        """
        if N == 0:
            N = 10*self.sps + 1

        rrc_vals = self._gen_rrc(alpha, N)
        self.samples = np.convolve(self.samples, rrc_vals, mode='valid')
        # Rescale the samples
        self.normalise_amplitude()

        # Return the filter values
        return rrc_vals

    def trim_by_power(self, padding: int = 0, std_cut: float = 1.5,
                      n: int = 10, drop: bool = True):
        """
        Trims low power noise from the beginning and end of a signal. Runs a moving average over the samples first.
        When recording a signal manually there will always be unwanted noise before the signal and after the signal.
        Use this function to cut it out.

        :param padding: n sample padding either side of the cut
        :param std_cut: The threshold that is used to decide if a sample is part of the signal or not, measured in
                        standard deviations from the mean
        :param n: The length of the moving average that is applied to the samples before cutting
        :param drop: If drop is True then the samples are cut out from the signal, otherwise they are set to 0+0j
        """
        # If we do a moving average over the abs value of the samples (the abs value being the power) we get a
        # clear spike where the sig begins
        av = np.convolve(np.abs(self.samples), np.ones(n), 'valid') / n
        sdev = np.std(av)

        index = np.arange(len(av))[abs(av) > std_cut * sdev]

        # first is the turn on, last is turn off
        first_ind = index[0] - int(padding)
        first_ind = max(first_ind, 0)

        last_ind = index[-1] + int(padding)

        if drop:
            self.samples = self.samples[first_ind:last_ind]
        else:
            self.samples[:first_ind] = 0 + 0j
            self.samples[last_ind:] = 0 + 0j

    # ***********************************                    ************************************
    # ************************************ Plotting Functions ************************************
    # *************************************                    ************************************

    def phase_view(self, n: int = 4000000, start_sample: int = 0):
        """
        Generates and displays a phase trace plot. This is the phase of each samples.

        :param n: How many samples to plot
        :param start_sample: The sample to start plotting from
        """
        kwargs = {
            "type": "view",
            "subtype": "phase",
            "start": start_sample
            }
        plot(self.samples[start_sample:start_sample + n], **kwargs)

    def freq_view(self, n: int = 4000000, start_sample: int = 0):
        """
        TODO: Fix the big freq spikes somehow
        Generates and displays a frequency trace plot. This is the instantaneous frequencies of the samples.

        :param n: How many samples to plot
        :param start_sample: The sample to start plotting from
        """
        kwargs = {
            "type": "view",
            "subtype": "freq",
            "fs": self.fs,
            "start": start_sample
            }
        plot(self.samples[start_sample:start_sample + n], **kwargs)

    def amp_view(self, n: int = 4000000, start_sample: int = 0):
        """
        Generates and displays a amplitude trace plot. This is the absolute value of the samples

        :param n: How many samples to plot
        :param start_sample: The sample to start plotting from
        """
        kwargs = {
            "type": "view",
            "subtype": "amp",
            "start": start_sample
            }
        plot(self.samples[start_sample:start_sample + n], **kwargs)

    def specgram(self, nfft: int = 1024):
        """
        Creates a spectrogram plot

        :param nfft: The size of the fft filter. Make this larger for more detail in the plot
        """
        # Nfft shouldn't be bigger than the samples
        if nfft >= len(self.samples):
            nfft = int(len(self.samples)/4)

        kwargs = {"type": "specgram",
                "nfft": nfft,
                "fs": self.fs,
                "title": f"Specgram at Baseband (NFFT={nfft})"}

        plot(self.samples, **kwargs)

    def psd(self, nfft: int = 1024):
        """
        Generates and displays a power spectrum density plot

        :param nfft: How many bins the fft uses. A bigger nfft results in more detail
        """
        kwargs = {"type": "psd",
                  "nfft": nfft,
                  "fs": self.fs,
                  "title": f"PSD at Baseband (NFFT={nfft})"}
        plot(self.samples, **kwargs)

    def iq(self, n: int = 500000, start_sample: int = 0):
        """
        Generates and displays a IQ plot

        :param n: How many samples to plot
        :param start_sample: The sample to start plotting from
        """
        kwargs = {"type": "iq",
                  "title": "IQ Scatter"}

        plot(self.samples[start_sample:start_sample+n], **kwargs)

    def fft(self, nfft: int = 1024) -> None:
        """
        Generates and displays a frequency domain plot

        :param nfft: How many bins the fft uses. A bigger nfft results in more detail
        """
        kwargs = {"type": "fft",
                  "title": "FFT of Signal",
                  "fs": self.fs,
                  "nfft": nfft}
        plot(self.samples, **kwargs)

    def time(self, n: int = 400000, start_sample: int = 0) -> None:
        """
        Generates and displays a time series plot

        :param n: How many samples to plot
        :param start_sample: The sample to start plotting from
        """
        t = self.t
        t = t[start_sample:start_sample+n]

        kwargs = {"type": "time",
                  "t": t,
                  "title": "Time View",
                  "n": n}

        plot(self.samples[start_sample:start_sample+n], **kwargs)

    def save_wave(self, fn: str = None, path: Path = None, wav: bool = False) -> None:
        """
        Saves the samples to a file. Either saves as complex64 or as a wav file if wav=True
        :param fn: str, filename
        :param path: path object of the directory to save to. defaults to the current working directory
        :param wav: bool, if true outputs the samples as a .wav file which can be listened to.
        """
        # If there is no path provided then save it in the directory the function is called from
        path_object = None
        if not path:
            path_object = Path().absolute()
        else:
            path_object = Path(path)

        # Check to make sure that worked
        if not path_object:
            raise ValueError("Enter a valid absolute path into the path argument or leave it blank")

        # If no file name make one
        if not fn:
            fn = f"Sig_f={self.f}_fs={self.fs}_sps={self.sps}_{int(time())}"

        save_path = path_object.joinpath(fn)

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
            if self.samples.dtype != np.complex64:
                raise Warning(f"Data type is {self.samples.dtype} instead of complex64")
            self.samples.tofile(save_path)

    def save_message(self, fn: str) -> None:
        """
        Saves the stored message to a raw bytes file. This is useful for demodulating.
        :param fn: Filename
        """
        message_bytes = np.packbits(self.message)
        message_bytes.tofile(fn)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

