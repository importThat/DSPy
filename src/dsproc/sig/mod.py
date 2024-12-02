import numpy as np
from .constellation import Constellation
from ._sig import Signal
from ..util.utils import moving_average


class Mod(Signal):
    def __init__(self, fs: int, message: np.ndarray | list, sps: int = 16, amplitude: float = 1, f: int = 100):
        """
        Class used for modulating data into a wave. Extends functionality from the Signal class by adding functions
        for doing various modulation operations.


        fs: Sampling frequency. How often samples will be created for the wave. A wave with a sampling rate of 100Hz
            would have 100 samples per second.
        message: A numpy array of ints containing the message symbols which will be written into a wave.
        sps: How many samples to generate per symbol. Typical values are between 8 and 20. Lowering the samples
            per symbol will increase the data rate at the expense of making it more susceptible to errors.
        amplitude: The (approximate) max amplitude of the wave, typically 1.
        f: The centre frequency of the signal. Should be somewhere between -fs/2 and fs/2.

        # Example
        >>> s = Mod(fs = 20000, message=np.array([1, 0, 1, 1, 1, 2, 3]), sps=20, f=5000)    # instance a Signal object
        >>> (s.fs, s.message, s.sps, s.f)   # Display the parameters
        (20000, array([1, 0, 1, 1, 1, 2, 3]), 20, 5000)
        """
        super().__init__(fs=fs, message=message, sps=sps, amplitude=amplitude, f=f)

    def ASK(self) -> None:
        """
        Amplitude shift keying. Writes the message symbols into the amplitude of the wave by changing the A value in
        the equation:

            samples = A * e^i(*2pi*f*t + theta)

        Avoids zero amplitude symbols.

        # Example
        >>> s = Mod(10000, message=np.array([1, 0, 1, 0, 1, 1, 1, 0]), sps=2, f=3000)
        >>> s.ASK()
        >>> # The absolute value is the amplitude of the wave, so this shows that the input data has been written to
        >>> # different amplitude levels with 2 samples per symbol
        >>> print(np.abs(s.samples))
        [1.  1.  0.5 0.5 1.  1.  0.5 0.5 1.  1.  1.  1.  1.  1.  0.5 0.5]
        """
        amp_mod_z = np.repeat(self.message, self.sps)       # repeat each of the elements of the message, sps times
        amp_mod_z += 1  # Add 1 so amplitude is never 0
        amp_mod_z = amp_mod_z / max(amp_mod_z)      # Scale it so its <= 1

        self.samples = self.create_samples(freq=self.f, amp=amp_mod_z)

    def create_FSK_vector(self, spacing: int) -> np.ndarray:
        """
        Creates the frequency shift keying vector which will be used by the FSK function to write data into a wave

        spacing: The Hz spacing of the FSK peaks. A bigger spacing makes it easier to seperate the symbols but spreads
        the signal across a bigger bandwidth.

        >>> s = Mod(10000, message=np.array([1, 0, 1, 1, 1]), sps=4, f=3000)    # Low FSK just for testing
        >>> FSK_vector = s.create_FSK_vector(spacing=200)
        >>> print(FSK_vector[0:18])
        [500 500 500 500 300 300 300 300 500 500 500 500 500 500 500 500 500 500]

        """
        freqs = self.message + 1      # Add one to avoid zero frequency
        freqs = freqs.astype(np.int64)  # self.message is np.uint8 so we have to change here to 64bit
        freqs = freqs * spacing

        # This centers it back on self.f, so that the centre frequency of the signal is maintained
        max_diff = abs((self.M)*spacing - self.f)
        min_diff = abs(spacing - self.f)
        change = int(abs(max_diff - min_diff)/2)

        if max_diff > min_diff:
            # We shift down
           freqs -= change
        elif min_diff > max_diff:
            # we shift up
            freqs += change

        # Stretch the vector so it lines up with the symbol transitions
        f_mod_z = np.repeat(freqs, self.sps)

        return f_mod_z

    def FSK(self, spacing: int) -> None:
        """
        Frequency shift keying. Writes the message symbols into the frequency of the wave by changing the f value in
        the equation:

            samples = A * e^i(*2pi*f*t + theta)

        frequency changes are additive with the centre frequency of the wave. Adds one to the centre frequency of the
        wave to avoid a zero frequency signal.

        spacing: The Hz spacing of the FSK peaks. A bigger spacing makes it easier to seperate the symbols but spreads
        the signal across a bigger bandwidth.

        >>> s = Mod(10000, message=np.array([1, 0, 1, 1, 1]), sps=32, f=3000)
        >>> s.FSK(spacing=300)  # 300hz between each peak
        >>> np.round(s.samples, 5).sum()    # Some test for doctest reproducibility
        np.complex64(-0.7637102-3.7421002j)

        """
        f_mod_z = self.create_FSK_vector(spacing)

        z = self.create_samples(freq=f_mod_z, theta=0, amp=1)
        self.samples = z.astype(np.complex64)

    def QPSK(self) -> None:
        """
        Quadrature phase shift keying. This function writes data into the phase of the a wave by changing the theta
        parameter in the equation:

            samples = A * e^i(2pi*f*t + theta)

        Using QPSK will result in the symbols being mapped to the unit circle on the IQ graph.

        >>> m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> s = Mod(10000, message=m, sps=16, f=3000)
        >>> s.QPSK()
        >>> np.round(s.samples, 5).sum()  # Some test for reproducibility
        np.complex64(0.20153952+1.9175801j)

        """
        M = len(np.unique(self.message))    # The number of symbols

        # Convert the message symbols to M radian phase offsets with a pi/M bias from zero
        # i.e. if we had 4 symbols make them 45, 135, 225, 315 degree phase offsets (1/4pi, 3/4pi, 5/4pi, 7/4pi)
        symbols = self.message * 2 * np.pi / M + np.pi/ M
        message = np.repeat(symbols, self.sps)

        z = self.create_samples(freq=self.f, theta=message)

        self.samples = z.astype(np.complex64)

    def QAM(self, constellation: str | np.ndarray = "square") -> None:
        """
        Quadrature amplitude modulation. Writes symbols to two-dimensional points on the real/imaginary (IQ) plane.
        Uses a constellation object to determine where to place the points. If the number of unique symbols is less
        than the number constellation points then the most power efficient points are selected.

        constellation: A string that specifies one of the generic constellation types, or a map of points
        that will be used to do the modulation. Generic options are square, sunflower, star, or square_offset.
        If providing a map, the map must be a numpy array of complex values with length >= the number of unique
        symbols

        Examples
        # Using a generic constellation map
        >>> m = [0, 1, 2, 3, 0, 1, 2, 3]
        >>> s = Mod(fs=1000, message=m, sps=4)   # Low sps for printing
        >>> s.QAM(constellation='square')   # Maps points to a square constellation
        >>> # Use s.iq() to view the points that have been mapped to the constellation. If the points are spinning,
        >>> # then use the baseband() method to have them display correctly
        >>> # If you print the samples you can see how they move from the points -1+1j to 1+-1j, which are two of the
        >>> # points in the constellation map
        >>> print(s.samples[0:6])
        [ 1.        +1.j          0.22123174+1.3968023j  -0.64203954+1.2600735j
         -1.2600735 +0.64203954j  0.22123174-1.3968023j   1.        -1.j        ]

        # Using a user-defined constellation map
        >>> m = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        >>> s = Mod(fs=1000, message=m, sps=3)
        >>> # Use a triangle constellation map.
        >>> constellation_map = np.array([-0.01100416+1.41675662j, -0.99324728-0.99439021j, 1.00232262-0.98607591j])
        >>> s.QAM(constellation=constellation_map)
        >>> print(s.samples[0:10])
        [-0.00776715+1.j         -0.594069  +0.8044516j  -0.9534567 +0.30163j
          0.8841685 -0.44986615j  0.97973204+0.15575185j  0.7010712 +0.70187795j
         -0.9814649 +0.14723893j -0.88056666-0.4577718j  -0.44332188-0.88792926j
          0.5815015 +0.8135824j ]
        """
        # Create the constellation map - a lookup table of values that will be indexed by the message values
        c = Constellation(M=self.M)

        if type(constellation) is str:
            if constellation == "square":
                c.square()
            elif constellation == "sunflower":
                c.sunflower()
            elif constellation == "star":
                c.star()
            elif constellation == "square_offset":
                c.square_offset()
            else:
                raise ValueError(f"{constellation} is not a recognised Constellation generic. "
                                 f"See docstring for options")

        elif type(constellation) is np.ndarray:
            # Test to see if constellation is a complex ndarray
            if not hasattr(constellation, "imag"):
                raise TypeError("Provided constellation map must be a numpy array with complex type")

            # Test to see if the constellation map has enough points
            elif self.M > len(constellation):
                raise ValueError("Length of the provided constellation map must be greater than or equal to the number"
                                 "of unique symbols")
            c.map = constellation
        else:
            raise ValueError("Incorrect Constellation type")

        # Remove unused points
        c.prune()
        # Scale to between 1 and -1
        c.normalise()

        message = np.repeat(self.message, self.sps)

        offsets = c.map[message]      # Index the map by the symbols

        z = self.create_samples(freq=self.f, theta=np.angle(offsets), amp=np.abs(offsets))
        z = z.astype(np.complex64)  # Ensure type

        self.samples = z

    def CPFSK(self, spacing: int) -> None:
        """
        TODO: Currently has instability for long messages due to floating point error in the np.cumsum operation

        Continuous phase frequency shift keying. Uses a phase offset vector to minimise phase jumps arising
        from frequency shift keying, which makes it more spectrally efficient.

        spacing: The Hz spacing of the FSK peaks. A bigger spacing makes it easier to seperate the symbols but spreads
        the signal across a bigger bandwidth.

        resource:
        https://dsp.stackexchange.com/questions/80768/fsk-modulation-with-python


        """
        # Create the frequency modulating vector
        f_mod_z = self.create_FSK_vector(spacing)

        # Cumulative phase offset
        delta_phi = 2.0 * f_mod_z * np.pi / self.fs    # Change in phase at every timestep (in radians per timestep)
        phi = np.cumsum(delta_phi)              # Add up the changes in phase

        z = self.amp * np.exp(1j * phi)  # creates sinusoid theta phase shift
        z = np.array(z)
        self.samples = z.astype(np.complex64)


    def CPFSK_smoother(self, spacing: int, smooth_n: int = 10, weights: np.ndarray | list | tuple = None):
        """
        TODO: Currently has instability for long messages due to floating point error in the np.cumsum operation

        Continuous phase frequency shift keying. Uses a phase offset vector to minimise phase jumps arising
        from frequency shift keying, which makes it more spectrally efficient. Additionally this function applies
        a smoothing window over the symbol boundaries, which reduces noise.

        spacing: The Hz spacing of the FSK peaks. A bigger spacing makes it easier to seperate the symbols but spreads
        the signal across a bigger bandwidth.

        smooth_n: The number of samples to smooth over. Typically this will be around the samples per symbol

        weights: Custom weights can be used for the smoothing window. Iterable of floats.

        resource:
        https://dsp.stackexchange.com/questions/80768/fsk-modulation-with-python
        """
        # Create the frequency vector
        f_mod_z = self.create_FSK_vector(spacing)

        # Now we pass an averaging window over the frequencies. This will ensure we slowly transition from one
        # frequency to the next.

        # Test smooth_n argument
        if smooth_n <= 0:
            smooth_n = 1
        if smooth_n > self.sps:
            raise ValueError("smooth_n should not be greater than the samples per symbol")

        # Creating the smoothing window
        if weights is None:
            window = np.ones(smooth_n)
        else:
            window = np.array(weights)

        if smooth_n != len(window):
            raise ValueError("weights must have the same length as smooth_n")

        ma = moving_average(f_mod_z, smooth_n, weights=window)

        # Cumulative phase offset
        delta_phi = 2.0 * ma * np.pi / self.fs  # Change in phase at every timestep (in radians per timestep)
        phi = np.cumsum(delta_phi)  # Add up the changes in phase

        z = self.amp * np.exp(1j * phi)  # creates sinusoid theta phase shift
        z = np.array(z)
        self.samples = z.astype(np.complex64)

    def FHSS(self, hop_f: int, freqs: np.ndarray, pattern=np.array([])):
        """
        Frequency hopping spread spectrum, causes the signal to hop from frequency to frequency at a pre-define hop
        rate. Frequency hopping is made by multiplying the signal by a carrier wave that jumps to each different
        frequency. Note that FHSS doesn't encode any sort of pause, the wave simply hops directly from one frequency
        to another. This would make analog demodulation impossible, so if that is your aim you will have to
        input pauses.

        hop_f: How many hops per second there are, i.e. the hopping frequency. Higher hopping frequencies will require
            the receiver to retune at a faster rate to capture the transmit.

        freqs: A numpy array of ints of the frequencies that are to be hopped too. Note that the frequencies are additive
            with the center frequency of the signal. i.e. if the center frequency is 100Hz and the hop frequencies
            are [-50, 150], then the resultant hopped signal will be at -50Hz and then 250Hz.

        pattern: optional, np.array of int indices that align with some frequencies in the freqs parameter.
            The pattern is an optional parameter than can be defined to say what order the freqs list should be jumped
            through. e.g. [0, 2, 1] would start with the first frequency, then go to the 3rd, then the second.

        # Example
        >>> m = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        >>> s = Mod(fs=1000, message=m, sps=64)
        >>> s.ASK()  # Amplitude shift keying
        >>> # Filter the signal, otherwise it's hard to see the hops
        >>> s.baseband()
        >>> _ = s.butterworth_filter(25, 'lowpass') # Apply a butterworth lowpass filter
        >>> # Hop the signal
        >>> hop_frequencies = np.array([-200, -100, 0, 100, 200])
        >>> hop_pattern = np.array([0, 2, 1, 3, 4])
        >>> s.FHSS(hop_f=10, freqs=hop_frequencies, pattern=hop_pattern)
        >>> # Use the specgram plot and a large nfft to actually see the hops like so:
        >>> # s.specgram(nfft=8192)
        >>> s.n_samples == len(s.message) * s.sps    # Some test for doctest
        True

        """
        # If no pattern is given
        if len(pattern) == 0:
            pattern = np.arange(len(freqs))

        # The number of samples we transmit before hopping
        hop_samps = 1 / hop_f * self.fs

        # Make the FHSS vector
        f_mod_z = freqs[np.array(pattern)]
        f_mod_z = f_mod_z.repeat(hop_samps)
        n_tiles = int(np.ceil(len(self.samples) / len(f_mod_z)))
        # Repeat the pattern
        f_mod_z = np.tile(f_mod_z, n_tiles)
        # Trim it to fit
        f_mod_z = f_mod_z[0:len(self.samples)]

        # Now mod the wave with it
        angle = 2 * np.pi * f_mod_z * self.t
        z = np.cos(angle) + 1j * np.sin(angle)
        self.samples *= z[0:len(self.samples)]





