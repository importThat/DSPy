import numpy as np
from dsproc.sig.constellation import Constellation
from dsproc.sig._sig import Signal


class Mod(Signal):
    def __init__(self, f, fs, message, duration=1, amplitude=1):
        super().__init__(f=f, fs=fs, message=message, duration=duration, amplitude=amplitude)

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
        Pulse-spacing modulation, also pulse position modulation. Modulates modulations by changing the time difference
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

    def QAM(self, type="square", custom_map=None):
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
        elif type == "custom":
            if custom_map is None:
                raise ValueError("Provide a custom constellation map in the custom_map argument")
            else:
                c.map = custom_map
        else:
            raise ValueError("Incorrect Constellation type")

        c.prune()
        c.normalise()

        message = np.repeat(self.message, self.sps)

        offsets = c.map[message]      # Index the map by the symbols

        z = self.create_samples(freq=self.f, theta=np.angle(offsets), amp=np.abs(offsets))

        self.samples = z

    def CPFSK(self, squish_factor=20):
        """
        samples = A * e^i(2pi*f*t + theta)

        Continuous phase frequency shift keying. Uses a phase offset vector to minimise phase jumps arising
        from frequency shift keying, which makes it more spectrally efficient.

        The squish factor squishes the frequencies together. The higher the squish the closer together they are.

        resource:
        https://dsp.stackexchange.com/questions/80768/fsk-modulation-with-python


        """

        # TODO
        #   Change phase vector so it is always < 2pi
        #   To speed up computation, can precompute the phase offset per symbol
        #
        freqs = self.message + squish_factor      # The larger the number added here to more the frequencies are pushed together
        freqs = freqs / max(freqs)   # Normalize
        freqs = freqs * self.f
        f_mod_z = np.repeat(freqs, self.sps)     # FSK message

        # Cumulative phase offset
        delta_phi = 2.0 * f_mod_z * np.pi / self.fs    # Change in phase at every timestep (in radians per timestep)
        phi = np.cumsum(delta_phi)              # Add up the changes in phase

        z = self.amp * np.exp(1j * phi)  # creates sinusoid theta phase shift
        z = np.array(z)
        self.samples = z.astype(np.complex64)
        self.fsk = True

    def CPFSK_smoother(self, squish_factor=20, smooth_factor=1):
        """
        samples = A * e^i(2pi*f*t + theta)

        Continuous phase frequency shift keying. Uses a phase offset vector to minimise phase jumps arising
        from frequency shift keying, which makes it more spectrally efficient.

        The squish factor squishes the frequencies together. The higher the squish the closer together they are.

        Smooth factor determines over how many samples the frequencies will be smoothed. 1 is 100% of the samples
        and 0 is 0% of the samples. There is some rounding that occurs, so for smaller samples-per-second symbols
        the effect will be noticeably stepped.

        resource:
        https://dsp.stackexchange.com/questions/80768/fsk-modulation-with-python
        """

        #
        freqs = self.message + squish_factor  # The larger the number added here to more the frequencies are pushed together
        freqs = freqs / max(freqs)  # Normalize
        freqs = freqs * self.f
        f_mod_z = np.repeat(freqs, self.sps)  # FSK message

        # Now we pass an averaging window over the frequencies. This will ensure we slowly transition from one
        # frequency to the next.
        # I guess we just do it linearly
        f_smooth = np.cumsum(f_mod_z)
        if smooth_factor == 0:
            sn = 1
        else:
            sn = max(1, int((self.sps / (1/smooth_factor))))
        f_smooth[sn:] = f_smooth[sn:] - f_smooth[:-sn]
        f_smooth = f_smooth[sn - 1:] / sn

        # Cumulative phase offset
        delta_phi = 2.0 * f_smooth * np.pi / self.fs  # Change in phase at every timestep (in radians per timestep)
        phi = np.cumsum(delta_phi)  # Add up the changes in phase

        z = self.amp * np.exp(1j * phi)  # creates sinusoid theta phase shift
        z = np.array(z)
        self.samples = z.astype(np.complex64)
        self.fsk = True




