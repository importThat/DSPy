import numpy as np
from matplotlib import pyplot as plt
from dsproc.sig._sig import Signal
from scipy.cluster.vq import kmeans
from dsproc.sig.constellation import Constellation

"""
Class with demod functions, ideally this is automated but very much still a work in progress

"""


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

    def detect_clusters(self, M, iter=3):
        """
        Detects M clusters of points in the demod samples. Returns a constellation object with the guessed cluster data
        which can then be used to map to symbols
        :param M: A guess at the number of clusters
        :param iter: The number of times to run the kmeans algorithm
        :return: A constellation object with the cluster data
        """
        if M < 0 or type(M) != int or M > len(self.samples):
            raise ValueError("M must be an integer > 0 and less than the number of samples available")

        # The points to cluster
        points = np.array([self.samples.real, self.samples.imag])
        points = points.T

        # create the clusters
        clusters = kmeans(points, M, iter=iter)
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
            #global c
            if event.button == 1:
                if event.xdata and event.ydata:
                    new_point = np.array([event.xdata + 1j*event.ydata])
                    c.map = np.concatenate([c.map, new_point])

                    # Add the new point in
                    arr = np.array([c.map.real, c.map.imag]).T
                    art.set_offsets(arr)

                    plt.draw()


        def onpick(event):
            #global c

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



