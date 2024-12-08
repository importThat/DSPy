import unittest
import numpy as np
import dsproc
from dsproc.sig._sig import Signal


MESSAGE = dsproc.create_message(1000, 2)


def create_wave(t, f, amp, phase):
    angle = 2 * np.pi * f * t + phase
    wave = amp * np.cos(angle) + 1j * amp * np.sin(angle)

    return wave.astype(np.complex64)


class TestSig(unittest.TestCase):
    def test_dur(self):
        s = dsproc.Signal(fs=100, message=MESSAGE, sps=2)
        self.assertEqual(s.dur, len(s.message) * s.sps / s.fs)
        s.fs = 200
        self.assertEqual(s.dur, len(s.message) * s.sps / s.fs)
        s.message = MESSAGE[0:400]
        self.assertEqual(s.dur, len(s.message) * s.sps / s.fs)
        s.sps = 10
        self.assertEqual(s.dur, len(s.message) * s.sps / s.fs)

    def test_M(self):
        s = dsproc.Signal(fs=100, message=MESSAGE, sps=2)
        self.assertEqual(s.M, len(np.unique(MESSAGE)))
        new_message = dsproc.create_message(100, 6)
        s.message = new_message
        self.assertEqual(s.M, len(np.unique(new_message)))

    def test_t(self):
        s = Signal(fs=100, message=MESSAGE, sps=2)
        self.assertEqual(len(s.t), len(1 / s.fs * np.arange(s.dur * s.fs)))
        s.fs = 1000
        self.assertEqual(len(s.t), len(1 / s.fs * np.arange(s.dur * s.fs)))
        s.sps = 100
        self.assertEqual(len(s.t), len(1 / s.fs * np.arange(s.dur * s.fs)))
        s.message = MESSAGE[0:400]
        self.assertEqual(len(s.t), len(1 / s.fs * np.arange(s.dur * s.fs)))

    def test_create_samples(self):
        s = Signal(fs=100, message=MESSAGE, sps=2)

        wave = create_wave(s.t, f=1, amp=1, phase=0)
        s_wave = s.create_samples(1, 0, 1)

        self.assertTrue(np.all(s_wave == wave))

        vector = np.random.randint(1, 3, len(MESSAGE)*s.sps)

        # Test freq
        wave = create_wave(s.t, f=vector, amp=1, phase=0)
        s_wave = s.create_samples(vector, 0, 1)

        self.assertTrue(np.all(s_wave == wave))

        # Test Amplitude
        wave = create_wave(s.t, f=1, amp=vector, phase=0)
        s_wave = s.create_samples(1, 0, vector)

        self.assertTrue(np.all(s_wave == wave))

        # Test phase
        wave = create_wave(s.t, f=1, amp=1, phase=vector)

        s_wave = s.create_samples(1, vector, 1)

        self.assertTrue(np.all(s_wave == wave))

    def test_baseband(self):
        # Shift down
        s = Signal(fs=100, message=MESSAGE, sps=2, f=100)
        s.samples = s.create_samples(freq=s.f)
        s.baseband()

        wave = create_wave(s.t, f=100, amp=1, phase=0)
        offset_wave = create_wave(s.t, f=-100, amp=1, phase=0)
        wave = wave * offset_wave

        self.assertTrue(np.all(s.samples == wave))

        # Shift up
        s = Signal(fs=100, message=MESSAGE, sps=2, f=-100)
        s.samples = s.create_samples(freq=s.f)
        s.baseband()

        wave = create_wave(s.t, f=-100, amp=1, phase=0)
        offset_wave = create_wave(s.t, f=100, amp=1, phase=0)
        wave = wave * offset_wave

        self.assertTrue(np.all(s.samples == wave))

    def test_phase_offsets(self):

        for angle in range(-360, 420, 60):
            s = Signal(fs=100, message=MESSAGE, sps=2, f=100)
            s.samples = s.create_samples(freq=s.f)

            # create wave to test against
            wave = s.samples.copy()
            phase_offset = angle * np.pi / 180
            z = 1 * np.cos(phase_offset) + 1j * np.sin(phase_offset)
            z = z.astype(np.complex64)
            wave *= z

            # Test
            s.phase_offset(angle=angle)
            difference = np.abs(np.sum((s.samples - wave)))
            self.assertAlmostEqual(difference, 0)   # See if the phase difference is almost zero

    def test_freq_offsets(self):
        for freq in range(-200, 250, 50):
            if freq == 0:
                continue
            s = Signal(fs=200, message=MESSAGE, sps=2, f=1)
            s.samples = s.create_samples(freq=s.f)

            # create wave to test against
            wave = create_wave(s.t, 1, amp=1, phase=0)
            freq_offset = create_wave(s.t, freq, amp=1, phase=0)
            wave *= freq_offset

            # Test
            s.freq_offset(freq=freq)
            self.assertTrue(np.all(s.samples == wave))


if __name__ == "__main__":
    unittest.main(verbosity=1)