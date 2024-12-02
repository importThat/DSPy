import unittest
import dsproc
from random import randint


class TestMod(unittest.TestCase):

    def test_init(self):
        """
        Tests the initialisation of the class
        """
        for i in range(10):
            # Create the message
            message_n = randint(1, 1000)
            message_m = randint(2, 12)
            message = dsproc.create_message(message_n, message_m)

            # Create the signal
            fs = randint(10000, 1000000)
            sps = randint(8, 24)
            amp = randint(1, 10)
            f = randint(int(-0.5*fs), int(0.5*fs))

            s = dsproc.Mod(fs, message, sps, amp, f)

            # Assert that all the params wrote correctly
            self.assertTrue((message == s.message).all())
            self.assertEqual(fs, s.fs)
            self.assertEqual(sps, s.sps)
            self.assertEqual(amp, s.amp)
            self.assertEqual(f, s.f)

    def test_ASK(self):
        """
        Tests to ensure that the ASK modulation produces the right number of samples
        """
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 8)
            s.ASK()
            self.assertEqual(len(s.samples), s.sps*len(MESSAGE))
            self.assertIsNotNone(s.samples)

    def test_FSK(self):
        for m in range(2, 16):
            MESSAGE = dsproc.create_message(1000, m)
            s = dsproc.Mod(200, MESSAGE, 8)
            s.FSK(spacing=50)
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
                s.QAM(constellation=c)
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