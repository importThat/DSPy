import numpy as np
from matplotlib import pyplot as plt
from ._sig import Signal
from scipy.cluster.vq import kmeans
from .constellation import Constellation
from scipy.signal import correlate, savgol_filter

"""
Class with demod functions, ideally this is automated but very much still a work in progress

"""


class Demod(Signal):
    def __init__(self, fs, fn=None, f=0):
        self.fn = fn
        super().__init__(f=f, fs=fs, message=[], amplitude=1)

        if fn:
            self.samples = self.read_file()
        else:
            self.samples = np.array([])

    def read_file(self, folder=""):
        file = folder + self.fn
        samples = np.fromfile(file, np.complex64)
        return samples

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
                # If it's the last sample
                if i == len(freq)-1:
                    continue
                else:
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
        levels = np.sort(clusters[0])

        symbols = np.arange(len(levels))
        out = []

        for sample in averaged:
            index = (np.abs(levels - sample)).argmin()
            out.append(symbols[index])

        return np.array(out)

    def transmit_window(self, min_amp: float, min_dur: int):
        """
        For use on recordings with signal pulses. Returns tuples which show the start and stop of each pulse.

        min_amp: The minimum amplitude that consitutes a pulse. Masks out parts of the signal that are less than this.
        min_dur: The minimum duration in samples for a pulse. Anything shorter than this will be dropped.

        return a two-dimensional array where the first index is the start of a pulse (in samples) and the second is
        the end (not-inclusive).
        """
        # Create an amplitude mask
        amp_mask = np.abs(self.samples)

        # If it's below it's noise, if it's above it's signal
        amp_mask[amp_mask < min_amp] *= 0
        amp_mask[amp_mask >= min_amp] = 1

        # Figure out xmit durations
        # Now we need to figure out the transmit durations, this is how long (in samples) each xmit lasts for
        change_indices = np.where(amp_mask[:-1] != amp_mask[1:])[0] + 1

        # So if the final index is on we need to add an off, but if it is off we don't need to add anything
        # TODO
        #  I feel like there is a bug here. My spidey senses are a-tinglin'
        if amp_mask[-1] == 1:
            change_indices = np.concatenate([change_indices, np.array([len(self.samples)])])

        # If the first index is on, we need to add in a change at zero
        if amp_mask[0] == 1:
            change_indices = np.concatenate([np.array([0]), change_indices])

        change_indices = change_indices.reshape((-1, 2))
        durations = change_indices[:, 1] - change_indices[:, 0]
        starts = change_indices[:, 0]

        # Find the valid xmits
        valid_xmits = np.where(durations >= min_dur)
        xmit_tups = np.vstack([starts[valid_xmits], starts[valid_xmits] + durations[valid_xmits]]).T

        return xmit_tups

    def find_header(self, header: np.ndarray, signal: np.ndarray):
        """
        Uses correlation to find where the header occurs in the given signal. The header and the signal are normalised,
        however large value differences can still skew results. Note that if no header is present, this function will
        still return a result.

        header: the header as a 1 dimensional array in the given domain
        signal: A 1 dimensional array containing the header, needs to be in the appropriate domain.

        return: Int, the sample number at which the header begins in the signal (the max value of the correlation)
        """

        norm_header = (header - np.mean(header)) / (np.std(header) * len(header))
        norm_signal = (signal - np.mean(signal)) / (np.std(signal))

        c = correlate(norm_signal, norm_header, mode='full')
        xmit_start = np.argmax(c) - len(header) + 1

        return xmit_start

    def freq_search(self, start: int, end: int, bandwidth: int, tuning_steps: int = 30, fft_smoothness: int = 3):
        """
        Searches a sample range (start:end) for the dominant frequency that fits within the given bandwidth. First
        computes an fft, then smooths that fft, then slides a window of size bandwidth over the fft to identify the
        strongest frequency band. After it identifies this area, the function performs some tuning to narrow in on
        the best spot.

        Returns the centre-point of the window.

        start: The index of the start of the sample range
        end: The index of the end of that sample range
        bandwidth: The expected bandwidth of the peak that you are looking for. Can be approximate
        tuning_steps: How many iterations the tuning algorithm should be run for. More steps have exponentially less
        effect
        fft_smoothness: How much to smooth the fft by. 1 is max smoothness, and the length of the sample range is min
        smoothness.
        """
        # Create the fft
        cut = self.samples[start:end]
        cut_fft = np.fft.fftshift(np.abs(np.fft.fft(cut)))
        smooth_fft = savgol_filter(cut_fft, window_length=int(len(cut_fft) / fft_smoothness), polyorder=3)
        cut_f_axis = np.arange(self.fs / -2, self.fs / 2, self.fs / len(cut))

        # Convert bandwidth to fft indices
        bw_per_bin = self.fs / len(smooth_fft)
        bw = int(bandwidth / bw_per_bin)

        # Figure out search parameters
        step_size = int(len(smooth_fft)/10)
        if not step_size:
            step_size = 1
            raise Warning("fft length is very small, consider using a larger sample window")

        # Search over the freq axis and store the results
        measurements = []

        for j in range(0, len(smooth_fft), step_size):
            fft_sum = sum(smooth_fft[j:j + bw])
            amp_normed = fft_sum / bw
            measurements.append(amp_normed)

        # find the best measurements
        measurements = np.array(measurements)
        best_indices = np.where(measurements == max(measurements))[0]

        # There might be multiple indices
        # We always want to pick an index from the middle of the list
        if len(best_indices) == 1:
            best_index = best_indices

        # If it's an even number of values long
        elif len(best_indices) % 2 == 0:
            # Randomly get one of the middle indices
            index = len(best_indices) / 2 + np.random.randint(0, 2)
            best_index = best_indices[index]

        # If it's an odd number
        elif len(best_indices) % 2 == 1:
            # Get the middle index
            index = np.ceil(len(best_indices) / 2)
            best_index = best_indices[index]

        else:
            raise RuntimeError("Cannot find best index")

        midpoint = best_index * step_size + int(bw / 2)
        midpoint = midpoint[0]
        # Now we have an approximate midpoint we can test it by testing the upper and lower sums
        for i in range(tuning_steps):
            tuning_step_size = int(bw / (4 + i))

            upper_sum = sum(smooth_fft[midpoint:midpoint + int(bw / 2)])
            lower_sum = sum(smooth_fft[midpoint:midpoint + int(bw / 2)])

            # Step up
            if upper_sum > lower_sum:
                midpoint += tuning_step_size
            # Step down
            elif upper_sum < lower_sum:
                midpoint -= tuning_step_size
            # Perfection...?
            else:
                break

        return cut_f_axis[midpoint]















