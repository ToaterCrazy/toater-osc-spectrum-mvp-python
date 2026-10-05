import unittest

import numpy as np

from utils import SpectrumAnalyzer, trigger_waveform


SAMPLE_RATE = 48000
SAMPLE_COUNT = 8192
TIME = np.arange(SAMPLE_COUNT) / SAMPLE_RATE


class WaveformTests(unittest.TestCase):
    def test_two_sine_cycles_across_pitches_and_input_phases(self):
        expected = 0.7 * np.sin(2 * np.pi * np.linspace(0, 2, 1024))
        for frequency in (20, 27.5, 55, 110, 440, 1000, 4186, 8000, 20000):
            for phase in (0.1, 2.8):
                with self.subTest(frequency=frequency, phase=phase):
                    samples = 0.7 * np.sin(2 * np.pi * frequency * TIME + phase)
                    cycle, detected = trigger_waveform(samples)
                    self.assertEqual(cycle.shape, (1024,))
                    self.assertAlmostEqual(detected / frequency, 1, delta=0.002)
                    # Low notes have enough input samples for linear interpolation
                    # to reproduce the sine as well as its two-cycle duration.
                    if frequency <= 1000:
                        np.testing.assert_allclose(cycle, expected, atol=0.006)

    def test_bandlimited_saw_and_square_keep_the_fundamental(self):
        for frequency in (55, 440, 1760, 4186, 10000):
            phase = 2 * np.pi * frequency * TIME + 0.4
            harmonics = range(1, min(40, int(SAMPLE_RATE / 2 / frequency)) + 1)
            saw = sum(0.5 / h * np.sin(h * phase) for h in harmonics)
            square = sum(0.5 / h * np.sin(h * phase) for h in harmonics if h % 2)
            for name, samples in (("saw", saw), ("square", square)):
                with self.subTest(frequency=frequency, wave=name):
                    cycle, detected = trigger_waveform(samples)
                    self.assertIsNotNone(cycle)
                    self.assertAlmostEqual(detected / frequency, 1, delta=0.002)

    def test_extra_zero_crossings_do_not_change_pitch_or_trigger_phase(self):
        cycles = []
        for shift in (0, 0.1, 0.4, 1.3, 2.8, 5.1):
            phase = 2 * np.pi * 440 * TIME + shift
            samples = (0.15 * np.sin(phase + 0.4)
                       + 0.5 * np.sin(2 * phase + 0.2)
                       + 0.25 * np.sin(3 * phase + 0.9))
            cycle, detected = trigger_waveform(samples)
            self.assertAlmostEqual(detected, 440, delta=0.2)
            cycles.append(cycle)
        for cycle in cycles[1:]:
            np.testing.assert_allclose(cycle, cycles[0], atol=0.006)

    def test_dc_offset_and_noisy_tone(self):
        rng = np.random.default_rng(42)
        samples = 0.3 + 0.5 * np.sin(2 * np.pi * 220 * TIME)
        samples += rng.normal(0, 0.01, SAMPLE_COUNT)
        cycle, detected = trigger_waveform(samples)
        self.assertAlmostEqual(detected, 220, delta=0.5)
        self.assertAlmostEqual(np.mean(cycle), 0.3, delta=0.015)

    def test_resonant_ringing_does_not_move_the_note_or_trigger(self):
        for frequency in (55, 125.42, 440, 1000):
            phases = []
            for block in range(30):
                time = TIME + block * 1024 / SAMPLE_RATE
                samples = (0.5 * np.sin(2 * np.pi * frequency * time + 0.3)
                           + 0.08 * np.sin(2 * np.pi * 4613 * time + 0.9))
                waveform, detected = trigger_waveform(samples)
                with self.subTest(frequency=frequency, block=block):
                    self.assertAlmostEqual(detected, frequency, delta=0.01)
                    angle = (2 * np.pi * np.linspace(0, 2, len(waveform))
                             * frequency / detected)
                    basis = np.column_stack((np.cos(angle), np.sin(angle),
                                             np.ones(len(waveform))))
                    weights = np.sqrt(np.hanning(len(waveform)))
                    coefficients = np.linalg.lstsq(
                        basis * weights[:, None], waveform * weights, rcond=None
                    )[0]
                    phases.append(np.arctan2(coefficients[0], coefficients[1]))
                    self.assertAlmostEqual(np.hypot(*coefficients[:2]), 0.5, delta=0.005)
                    # The visible trace must still contain the resonant ripple.
                    residual = waveform - basis @ coefficients
                    self.assertGreater(np.std(residual), 0.04)
            self.assertLess(np.std(np.unwrap(phases)), 0.005)

    def test_missing_fundamental_still_displays_a_complete_period(self):
        phase = 2 * np.pi * 220 * TIME + 0.4
        samples = 0.5 * np.sin(2 * phase) + 0.4 * np.sin(3 * phase)
        waveform, detected = trigger_waveform(samples)
        self.assertIsNotNone(waveform)
        self.assertAlmostEqual(detected, 220, delta=0.1)

    def test_silence_and_nonperiodic_noise_have_no_cycle(self):
        rng = np.random.default_rng(42)
        for samples in (np.zeros(SAMPLE_COUNT), np.full(SAMPLE_COUNT, 0.2),
                        rng.normal(0, 1e-5, SAMPLE_COUNT),
                        rng.normal(0, 0.1, SAMPLE_COUNT)):
            self.assertEqual(trigger_waveform(samples), (None, None))


class SpectrumTests(unittest.TestCase):
    def analyzer(self, **kwargs):
        return SpectrumAnalyzer(SAMPLE_COUNT, SAMPLE_RATE, **kwargs)

    def test_silence_and_dc_are_at_floor(self):
        analyzer = self.analyzer(smoothing_seconds=0)
        for samples in (np.zeros(SAMPLE_COUNT), np.full(SAMPLE_COUNT, 0.25)):
            np.testing.assert_allclose(analyzer.process(samples, 0.02), -100)

    def test_absolute_level_is_not_normalized_to_the_loudest_bin(self):
        analyzer = self.analyzer(smoothing_seconds=0)
        frequency = analyzer.frequencies[100]
        for amplitude in (1, 0.5, 0.01):
            samples = amplitude * np.sin(2 * np.pi * frequency * TIME)
            db = analyzer.process(samples, 0.02)
            self.assertAlmostEqual(db[100], 20 * np.log10(amplitude), places=8)
        rng = np.random.default_rng(42)
        db = analyzer.process(rng.normal(0, 0.001, SAMPLE_COUNT), 0.02)
        self.assertLess(np.max(db), -70)

    def test_off_bin_tone_has_low_leakage_away_from_peak(self):
        analyzer = self.analyzer(smoothing_seconds=0)
        db = analyzer.process(0.5 * np.sin(2 * np.pi * 440 * TIME), 0.02)
        far = ((np.abs(analyzer.frequencies - 440) > 8 * SAMPLE_RATE / SAMPLE_COUNT)
               & (analyzer.frequencies >= 20))
        self.assertLess(np.max(db[far]), -90)

    def test_averaging_reduces_noise_flicker(self):
        rng = np.random.default_rng(42)
        raw = self.analyzer(smoothing_seconds=0)
        smooth = self.analyzer(smoothing_seconds=0.15)
        raw_levels, smooth_levels = [], []
        for _ in range(120):
            samples = rng.normal(0, 0.1, SAMPLE_COUNT)
            raw_levels.append(raw.process(samples, 1024 / SAMPLE_RATE)[200])
            smooth_levels.append(smooth.process(samples, 1024 / SAMPLE_RATE)[200])
        self.assertLess(np.std(smooth_levels[20:]), np.std(raw_levels[20:]) * 0.5)

    def test_release_depends_on_audio_time_instead_of_refresh_rate(self):
        tone = 0.5 * np.sin(2 * np.pi * (100 * SAMPLE_RATE / SAMPLE_COUNT) * TIME)
        results = []
        for step_count in (1, 10):
            analyzer = self.analyzer()
            analyzer.process(tone, 0.02)
            for _ in range(step_count):
                db = analyzer.process(np.zeros(SAMPLE_COUNT), 0.15 / step_count)
            results.append(db)
        np.testing.assert_allclose(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
