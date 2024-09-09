import numpy as np
import dsproc
import unittest


class TestUtils(unittest.TestCase):
    def test_create_message(self):
        # Test the create symbols
        for i in range(2, 10000, 500):
            message = dsproc.utils.create_message(i, 2)
            self.assertTrue(len(message) == i)
            self.assertTrue(len(np.unique(message)) == 2)

        for j in range(2, 100, 10):
            message = dsproc.utils.create_message(100, j)
            self.assertTrue(len(message) == 100)
            self.assertTrue(len(np.unique(message)) == j)

    def test_create_AWGN(self):
        noise = dsproc.utils.AWGN(1000, power=0.01)
        self.assertTrue(len(noise) == 1000)

    def test_find_nonzero_row(self):
        matrix = np.zeros(shape=(10, 10))

        for i in range(matrix.shape[1]):
            new_matrix = matrix.copy()
            zero_index = np.random.randint(0, 9)
            new_matrix[zero_index, i] = 1

            zero = dsproc.utils.find_nonzero_row(new_matrix, 0, i)
            self.assertTrue(zero == zero_index)


if __name__ == "__main__":
    unittest.main(verbosity=1)