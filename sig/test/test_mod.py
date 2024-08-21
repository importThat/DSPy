import unittest
import numpy as np
import dsproc
from dsproc import utils
from test_sig import create_wave

class TestMod(unittest.TestCase):
    def test_ASK(self):
        for m in range(2, 16):
            MESSAGE = utils.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.ASK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_FSK(self):
        for m in range(2, 16):
            MESSAGE = utils.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.FSK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_PSK(self):
        for m in range(2, 16):
            MESSAGE = utils.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.PSK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_QPSK(self):
        for m in range(2, 16):
            MESSAGE = utils.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.QPSK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_QAM(self):
        constellations = ["square", "sunflower", "star", "square_offset"]

        for c in constellations:
            for m in range(2, 16):
                MESSAGE = utils.create_message(1000, m)
                s = dsproc.Mod(200, MESSAGE, 2)
                s.QAM(type=c)
                self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
                self.assertIsNotNone(s.samples)

    def test_CPFSK(self):
        for m in range(2, 16):
            MESSAGE = utils.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.CPFSK(squish_factor=20)
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_CPFSK_smoother(self):
        n = 10
        for m in range(2, 16):
            MESSAGE = utils.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 20)
            s.CPFSK_smoother(smooth_n=n)
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE) - (n-1))    # -(n-1) due to the moving average
            self.assertIsNotNone(s.samples)

if __name__ == "__main__":
    unittest.main(verbosity=1)