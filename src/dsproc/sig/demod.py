import numpy as np
from matplotlib import pyplot as plt
from ._sig import Signal
from scipy.cluster.vq import kmeans
from .constellation import Constellation

"""
Class with demod functions, ideally this is automated but very much still a work in progress

"""


class Demod(Signal):
    def __init__(self, fs, filename=None):
        self.fn = filename
        super().__init__(f=0, fs=fs, message=[], amplitude=1)

        if filename:
            self.samples = self.read_file()
        else:
            self.samples = np.array([])

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

    def detect_clusters(self, M, iters=3):
        """
        Detects M clusters of points in the demod samples. Returns a constellation object with the guessed cluster data
        which can then be used to map to symbols
        :param M: A guess at the number of clusters
        :param iters: The number of times to run the kmeans algorithm
        :return: A constellation object with the cluster data
        """
        if M < 0 or type(M) != int or M > len(self.samples):
            raise ValueError("M must be an integer > 0 and less than the number of samples available")

        # The points to cluster
        points = np.array([self.samples.real, self.samples.imag])
        points = points.T

        # create the clusters
        clusters = kmeans(points, M, iter=iters)
        # Put the cluster points into the shape that constellation objects expect array([1+1j, ...]
        cluster_points = np.array(clusters[0])
        cluster_points = np.array([i[0]+1j*i[1] for i in cluster_points])

        # Create a constellation object with the clusters
        c = Constellation(M=M)
        c.map = cluster_points

        return c

    def view_constellation(self, c, samples=2000):
        """
        Plots the map from the given constellation against the demod samples and allows you to click and change the
        constellation points
        :param c: a constellation object
        :param samples: the number of samples to view from the demod data. Randomly selected
        """
        fig, ax = plt.subplots()
        background_data = np.random.choice(self.samples, size=samples, replace=False)
        background = ax.scatter(background_data.real, background_data.imag, color="blue")
        art = ax.scatter(c.map.real, c.map.imag, picker=True, pickradius=6, color="orange")

        # A FUNCTION IN A FUNCTION!??? Utter savage!
        # (It makes the scoping easier)
        def onclick(event):
            # global c
            if event.button == 1:
                if event.xdata and event.ydata:
                    new_point = np.array([event.xdata + 1j*event.ydata])
                    c.map = np.concatenate([c.map, new_point])

                    # Add the new point in
                    arr = np.array([c.map.real, c.map.imag]).T
                    art.set_offsets(arr)

                    plt.draw()

        def onpick(event):
            if event.mouseevent.button == 3:  # If it's a right mouse click
                ind = event.ind
                # Only get the closest point
                if len(ind) > 1:
                    del_point = np.array([event.mouseevent.xdata + 1j*event.mouseevent.ydata])

                    # Find the index of the nearest point
                    test_points = c.map[ind]
                    best_ind = (np.abs(test_points - del_point)).argmin()
                    ind = ind[best_ind]

                c.map = np.delete(c.map, ind, axis=0)

                # add the point in
                arr = np.array([c.map.real, c.map.imag]).T
                art.set_offsets(arr)
                plt.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        cid = fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()

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
        largest_inds = np.argpartition(ffts, -1)[-1]
        largest_val = axis[largest_inds]

        # The frequency offset
        freq = int(round(largest_val, 0)) # Make an int

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

        return np.array(out)

    def demod_ASK(self, m, iterations=1000):
        """
        Attempts to demodulate an amplitude shift keyed signal. Looks for m levels in the signal and assigns symbol
        values to those levels. Assumes that the signal is currently at one sample per symbol.
        """
        # Convert to amplitude values
        amps = np.abs(self.samples)
        # No need to whiten if we only have 1 feature
        # Perform kmeans clustering
        clusters = kmeans(amps, m, iter=iterations, check_finite=False)
        # Get the actual levels
        levels = clusters[0]

        # Now we map the levels to symbols

        symbols = np.arange(len(levels))
        out = []

        for sample in amps:
            index = (np.abs(levels - sample)).argmin()
            out.append(symbols[index])

        return np.array(out)

    def demod_FSK(self, m, sps, iterations=1000):
        """
        Attempts to demodulate a frequency shift keyed signal. First it attempts to smooth any high frequency peaks
        that might be caused by phase changes. Then it averages the frequency across the sample and maps the averages
        to a symbol.

        This function assumes that the samples per symbol is an integer number and that each symbol is well formed. If
        any symbols are not fully represented (so there are dropped samples for that symbol) then this function will
        probably not work too well. If you are missing the start of the first sample, simply pad that symbol out.

        Note the frequency modulated data is very dependant upon the samples per symbol. More samples per symbol will
        make it easier to identify the actual frequency being transmitted.
        """
        phase = np.unwrap(np.angle(self.samples))
        freq = np.diff(phase) / (2 * np.pi) * self.fs

        # Replace the first sample of each symbol with the second sample of each symbol, this will eliminate most
        # peaks caused by instant phase shifts
        for i in range(len(freq)):
            # if it's the first symbol of the sample
            if (i + 1) % sps == 0:
                freq[i] = freq[i + 1]

        # Changes peaks into the average
        sd = np.std(freq)
        av = np.mean(freq)

        peak_mask = abs(freq) > 2 * sd + abs(av)

        for i in range(len(peak_mask)):
            # If it is a peak, we want to look ahead to the next non-peak and use that value
            # (because peaks will probably occur at symbol boundaries)
            if peak_mask[i]:
                non_peak_index = np.where(peak_mask[i:i + sps] == False)[0]
                if non_peak_index.size > 0:
                    # Get the next non-peak value and overwrite the current peak with it
                    freq[i] = freq[i + non_peak_index[0]]
                else:
                    # If there isn't an appropriate non_peak to use, just overwrite with the average
                    freq[i] = av

        # now we can average over the sample and be somewhat confident that the peaks aren't effecting our output value
        averaged = [np.mean(freq[i:i + sps]) for i in range(0, len(freq), sps)]
        averaged = np.array(averaged)

        # Now we just cluster and then categorize

        clusters = kmeans(averaged, m, iter=iterations, check_finite=False)
        levels = clusters[0]

        symbols = np.arange(len(levels))
        out = []

        for sample in averaged:
            index = (np.abs(levels - sample)).argmin()
            out.append(symbols[index])

        return np.array(out)








