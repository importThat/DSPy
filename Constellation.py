import numpy as np
from matplotlib import pyplot as plt


class Constellation:
    def __init__(self, M):
        self.M = M  # The number of symbols
        self.order = int(np.ceil(np.sqrt(M)))    # defines how big the constellation must be to contain the symbols
        self.map = None

    def square(self):
        """
        Creates a square constellation that is trimmed down to the correct number of symbols
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

                elif np.real(j) < 0 and np.imag(j) > 0:
                    step_out += [j + 0 + 2j, j + -2 + 0j]

                elif np.real(j) < 0 and np.imag(j) < 0:
                    step_out += [j + 0 - 2j, j + -2 + 0j]

                elif np.real(j) > 0 and np.imag(j) < 0:
                    step_out += [j + 0 - 2j, j + 2 + 0j]

            # The corners
            corners = c1 + spacing + i * spacing
            corners = corners.tolist()

            step_out += corners
            const += step_out
            const = list(set(const))    # Removes the duplicates!

        self.map = np.array(const)

    def sunflower(self):
        """
        Creates a constellation type inspired by a sunflower (Credit - someone else)
        """
        # So at every timestep increase the angle by 137.5 degrees and increase the amplitude by 1/(2pi*amplitude) of the
        # previous step
        perfect_angle = 137.5 / 180 * np.pi
        imag_angle = np.cos(perfect_angle) + 1j * np.sin(perfect_angle)     # Counter clockwise rotation

        coords = [0.2+0j]

        for i in range(self.M):
            amp = np.abs(coords[-1])
            amp_increase = 1/(2*np.pi * amp)
            new_coord = (amp_increase + coords[i]) * imag_angle
            coords.append(new_coord)

        self.map = np.array(coords)

    def hexagon(self):
        """
        https://www.sciencedirect.com/science/article/abs/pii/S1874490721001166
        """
        pass

    def uniform(self):
        """
        Creates a constellation where the symbols are spread such that the distance between them is maximised
        """
        pass

    def star(self):
        """
        Creates a star constellation which comprises of multiple concentric rings at different amplitudes with the
        same number of points at the same phase (https://ieeexplore.ieee.org/document/9382012)
        """
        pass

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
            print("Map already pruned")
            return None

        amps = np.abs(self.map)
        n_drop = self.map.shape[0] - self.M      # The number of points to drop

        indexes = amps.argsort()[-n_drop::]     # Gets the indexes of the largest N items. Is indexes a word?

        self.map = np.delete(self.map, indexes)     # Remove the largest N values from the map

    def normalise(self):
        """
        Normalises the map to be between -1 and 1 in both dimensions
        """
        # We find the maximum value and then divide all the values by that
        all_vals = np.concatenate([self.map.imag, self.map.real])
        max_val = max(all_vals)

        self.map = self.map / max_val


    def plot(self):
        """
        Plots the constellation
        """
        plt.scatter(np.real(self.map), np.imag(self.map))
        plt.grid(True)
        plt.show()





