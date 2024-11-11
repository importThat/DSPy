import unittest
from dsproc.src import dsproc


class TestMod(unittest.TestCase):
    def test_ASK(self):
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.ASK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_FSK(self):
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.FSK(spacing=50)
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_PSK(self):
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.PSK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_QPSK(self):
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.QPSK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_QAM(self):
        constellations = ["square", "sunflower", "star", "square_offset"]

        for c in constellations:
            for m in range(2, 16):
                MESSAGE = dsproc.create_message(1000, m)
                s = dsproc.Mod(200, MESSAGE, 2)
                s.QAM(type=c)
                self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
                self.assertIsNotNone(s.samples)

    def test_CPFSK(self):
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 2)
            s.CPFSK(spacing=50)
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_CPFSK_smoother(self):
        n = 10
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 20)
            s.CPFSK_smoother(spacing=50)
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE) - (n-1))    # -(n-1) due to the moving average
            self.assertIsNotNone(s.samples)

if __name__ == "__main__":
    unittest.main(verbosity=1)