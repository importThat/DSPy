import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist


class Constellation:
    """
    Constellation objects are used for modulating and demodulating points into complex 2d space (IQ plane). This
    is used in quadrature amplitude signals.
    """
    def __init__(self, M: int):
        self.M = M  # The number of symbols
        self.order = int(np.ceil(np.sqrt(M)))    # defines how big the constellation must be to contain the symbols
        self.map = None

    def square(self):
        """
        Creates a square constellation that can be trimmed down to the correct number of symbols
        """
        spacing = np.array([2 + 2j, -2 + 2j, -2 - 2j, 2 - 2j])      # How far the squares are away from each other
        c1 = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]     # The starting points
        const = c1.copy()

        # Everytime the square adds another ring (or steps out) the size of the ring increases by 8
        # eg, ring size = 4, 12, 20, 28, 36...
        # The sum of the total squares increases with a perfect square pattern
        # 4, 16, 36, 64, 100, 144...

        n_rings = int(np.ceil(np.sqrt(self.M)/2))

        for i in range(n_rings - 1):    # -1 because we already did a ring with c1
            step_out = []

            # The middle bits
            for j in const:
                if np.real(j) > 0 and np.imag(j) > 0:
                    step_out += [j + 0 + 2j, j + 2 + 0j]

                elif np.real(j) < 0 < np.imag(j):
                    step_out += [j + 0 + 2j, j + -2 + 0j]

                elif np.real(j) < 0 and np.imag(j) < 0:
                    step_out += [j + 0 - 2j, j + -2 + 0j]

                elif np.real(j) > 0 > np.imag(j):
                    step_out += [j + 0 - 2j, j + 2 + 0j]

            # The corners
            corners = c1 + spacing + i * spacing
            corners = corners.tolist()

            step_out += corners
            const += step_out
            const = list(set(const))    # Removes the duplicates!

        self.map = np.array(const)

    def square_offset(self):
        """
        Square QAM but the rows are offset. Sometimes also called regular HQAM
        """
        self.square()

        # 2.
        # Order the points by ascending imaginary
        self.map = np.array(sorted(self.map, key=lambda x:x.imag))
        # The points are sequentially ordered so we can just subtract 0 from 1
        offset = abs(self.map[0].real - self.map[1].real) / 4
        # The number of points in a row is the sqrt of the total size
        row_len = int(np.sqrt(len(self.map)))
        # Move every first row right (positive) and every second row left (negative)
        # There will be row_len rows (because it's a square)
        for i in range(row_len):
            start_ind = i * row_len
            end_ind = row_len + i * row_len

            if i % 2 == 0:
                sign = 1
            else:
                sign = -1

            # Apply the offset to each row
            self.map[start_ind:end_ind].real = self.map[start_ind:end_ind].real + offset * sign

    def sunflower(self):
        """
        Creates a constellation type inspired by a sunflower (Credit - someone else)
        """
        # So at every timestep increase the angle by 137.5 degrees and increase the amplitude by 1/(n*pi*amplitude)
        # of the previous step
        perfect_angle = 137.5 / 180 * np.pi
        imag_angle = np.cos(perfect_angle) + 1j * np.sin(perfect_angle)     # Counter clockwise rotation

        coords = [0.03+0j]

        for i in range(self.M):
            amp = np.abs(coords[-1])
            amp_increase = 1/(100*np.pi * amp)      # I just changed the increase factor until it started looking ok
            new_coord = amp_increase + coords[i] * imag_angle

            coords.append(new_coord)

        self.map = np.array(coords)

    def hexagon(self):
        """
        https://www.sciencedirect.com/science/article/abs/pii/S1874490721001166
        """
        pass

    def star(self, n=8):
        """
        Creates a star constellation which comprises multiple concentric rings at different amplitudes with the
        same number of points at the same phase (https://ieeexplore.ieee.org/document/9382012)

        n: the number of points in each ring
        """
        # Find out how many rings are needed to capture all the symbols for a given n
        n_rings = int(np.ceil(self.M/n))

        # Create a different phase shift for each n
        phases = np.arange(0, 2*np.pi, 2*np.pi/n)
        # Make complex
        phases = np.cos(phases) + 1j * np.sin(phases)

        # Create the constellation
        self.map = np.concatenate([phases + i*phases for i in range(n_rings)])

    def rectangular(self):
        """
        Creates a rectangular constellation map. This form may be better than square if you're transmitting and odd
        number of bits per symbol, and depending upon the channel conditions
        (https://ieeexplore.ieee.org/document/9382012)
        """
        pass

    def prune(self):
        """
        prunes the constellation down to self.M number of points. removes the furthest away first
        """
        if self.map.shape[0] == self.M:
            # print("Map already pruned")
            return None

        amps = np.abs(self.map)
        n_drop = self.map.shape[0] - self.M      # The number of points to drop

        indexes = amps.argsort()[-n_drop::]     # Gets the indexes of the largest N items. Is indexes a word?

        self.map = np.delete(self.map, indexes)     # Remove the largest N values from the map

        return None

    def normalise(self):
        """
        Normalises the map to be between -1 and 1 in both dimensions
        """
        # We find the maximum value and then divide all the values by that
        all_vals = np.concatenate([self.map.imag, self.map.real])
        max_val = max(all_vals)

        self.map = self.map / max_val

    def iq(self):
        """
        Plots the constellation
        """
        plt.scatter(np.real(self.map), np.imag(self.map))
        plt.grid(True)
        plt.show()

    def average_distance(self):
        """
        Calculates the average Euclidean distance between points on the constellation (higher means that the
        constellation can withstand more noise before the points start overlapping)
        """
        # The pdist functions requires a two dimensional array
        two_dim = np.array([self.map.real, self.map.imag]).reshape((-1, 2))

        return np.mean(pdist(two_dim, metric="euclidean"))

    def average_power(self):
        """
        Calculates the average power per point of the constellation
        """
        return np.mean(np.abs(self.map))

    def error_floor(self):
        """
        To measure the effectiveness of a constellation we need to know how much noise if can withstand before
        detection errors start creeping in. This function calculates the amount of noise than can be added before
        symbols begin to overlap
        """
        pass




