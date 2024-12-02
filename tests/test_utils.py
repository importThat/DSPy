import numpy as np
import unittest
from dsproc import create_message, AWGN


class TestUtils(unittest.TestCase):
    def test_create_message(self):
        # Test the create symbols
        for i in range(2, 10000, 500):
            message = create_message(i, 2)
            self.assertTrue(len(message) == i)
            self.assertTrue(len(np.unique(message)) == 2)

        for j in range(2, 100, 10):
            message = create_message(100, j)
            self.assertTrue(len(message) == 100)
            self.assertTrue(len(np.unique(message)) == j)

    def test_create_AWGN(self):
        noise = AWGN(1000, power=0.01)
        self.assertTrue(len(noise) == 1000)


if __name__ == "__main__":
    unittest.main(verbosity=1)
