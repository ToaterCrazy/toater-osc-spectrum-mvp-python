import unittest

import numpy as np

from tuner import TunerTracker, pitch_to_note
from utils import trigger_waveform


class TunerTests(unittest.TestCase):
    def test_notes_and_octaves(self):
        for frequency, note in ((27.5, "A0"), (110, "A2"), (440, "A4"),
                                (880, "A5"), (261.6255653, "C4"),
                                (493.8833013, "B4"), (523.2511306, "C5")):
            with self.subTest(note=note):
                reading = pitch_to_note(frequency)
                self.assertEqual(reading.note, note)
                self.assertAlmostEqual(reading.cents, 0, delta=0.0001)
                self.assertAlmostEqual(reading.frequency, frequency, places=8)

    def test_sharp_and_flat_cents(self):
        for cents in (-29.4, 29.4):
            reading = pitch_to_note(440 * 2 ** (cents / 1200))
            self.assertEqual(reading.note, "A4")
            self.assertAlmostEqual(reading.cents, cents, places=8)
            self.assertAlmostEqual(reading.target_frequency, 440)

    def test_chromatic_notes_and_octave_boundary(self):
        for midi, note in ((61, "C♯4"), (70, "A♯4"), (71, "B4"), (72, "C5")):
            frequency = 440 * 2 ** ((midi - 69) / 12)
            self.assertEqual(pitch_to_note(frequency).note, note)

    def test_custom_reference(self):
        reading = pitch_to_note(432, reference_hz=432)
        self.assertEqual(reading.note, "A4")
        self.assertAlmostEqual(reading.cents, 0)
        reading = pitch_to_note(440, reference_hz=432)
        self.assertAlmostEqual(reading.cents, 31.76665, places=4)

    def test_invalid_pitch_and_reference(self):
        for frequency in (None, 0, -440, np.nan, np.inf):
            self.assertIsNone(pitch_to_note(frequency))
        for reference in (0, -440, np.nan, np.inf):
            with self.assertRaises(ValueError):
                TunerTracker(reference_hz=reference)

    def test_silence_clears_old_note_and_next_note_is_immediate(self):
        tracker = TunerTracker()
        tracker.update(440, 0.02)
        self.assertIsNone(tracker.update(None, 0.02))
        reading = tracker.update(523.2511306, 0.02)
        self.assertEqual(reading.note, "C5")
        self.assertAlmostEqual(reading.cents, 0, delta=0.0001)

    def test_new_notes_are_not_delayed_by_smoothing(self):
        tracker = TunerTracker()
        tracker.update(440, 0.02)
        self.assertEqual(tracker.update(880, 0.02).note, "A5")

    def test_note_hysteresis_prevents_boundary_flicker(self):
        tracker = TunerTracker(smoothing_seconds=0)
        for cents in (49, 51, 49, 51, 54):
            reading = tracker.update(440 * 2 ** (cents / 1200), 0.02)
            self.assertEqual(reading.note, "A4")
        reading = tracker.update(440 * 2 ** (56 / 1200), 0.02)
        self.assertEqual(reading.note, "A♯4")
        self.assertAlmostEqual(reading.cents, -44, places=6)

    def test_smoothing_depends_on_audio_time(self):
        results = []
        for steps in (1, 4):
            tracker = TunerTracker()
            tracker.update(440, 0.02)
            for _ in range(steps):
                reading = tracker.update(440 * 2 ** (20 / 1200), 0.08 / steps)
            results.append(reading.cents)
        self.assertAlmostEqual(*results, places=8)
        self.assertGreater(results[0], 0)
        self.assertLess(results[0], 20)

    def test_audio_detection_feeds_tuner_with_resonance_and_detuning(self):
        time = np.arange(8192) / 48000
        for cents in (-29.4, 0, 29.4):
            frequency = 440 * 2 ** (cents / 1200)
            samples = (0.5 * np.sin(2 * np.pi * frequency * time)
                       + 0.08 * np.sin(2 * np.pi * 4613 * time))
            _, detected = trigger_waveform(samples)
            reading = pitch_to_note(detected)
            self.assertEqual(reading.note, "A4")
            self.assertAlmostEqual(reading.cents, cents, delta=0.01)
        _, detected = trigger_waveform(np.zeros(8192))
        self.assertIsNone(pitch_to_note(detected))


if __name__ == "__main__":
    unittest.main()
