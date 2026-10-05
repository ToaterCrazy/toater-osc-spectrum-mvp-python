import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main
from tuner import TunerWindow


class AppTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")
        plt.ioff()

    def test_three_windows_follow_audio_and_closing_tuner_stops_stream(self):
        self.check_app(tuner_enabled=True)

    def test_default_two_windows_show_note_and_cents_without_tuner(self):
        self.check_app(tuner_enabled=False)

    def check_app(self, tuner_enabled):
        windows = []
        streams = []
        rendered = []
        frequency = 440 * 2 ** ((71 - 69 + 29.4 / 100) / 12)

        class FakeInputStream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
                self.closed = False
                streams.append(self)

            def feed(self, hz):
                time = np.arange(main.BUFFER_SAMPLES) / main.SAMPLE_RATE
                samples = 0.5 * np.sin(2 * np.pi * hz * time)
                self.callback(samples[:, None], len(samples), None, None)

            def __enter__(self):
                self.feed(frequency)
                return self

            def __exit__(self, *args):
                self.closed = True

        def make_tuner(**kwargs):
            window = TunerWindow(**kwargs)
            windows.append(window)
            return window

        def pause(_):
            self.assertEqual(len(plt.get_fignums()), 3 if tuner_enabled else 2)
            self.assertEqual(len(windows), 1 if tuner_enabled else 0)
            wave = plt.figure(plt.get_fignums()[0])
            title = wave.axes[0].get_title()
            rendered.append(title)
            window = windows[0] if tuner_enabled else None
            if len(rendered) == 1:
                self.assertTrue(title.endswith(" | B4 +29.4 cents"), title)
                if window is not None:
                    self.assertEqual(window.note_labels[1].get_text(), "B4")
                    self.assertEqual(window.cents_text.get_text(), "+29.4 ct")
                    self.assertEqual(window.cents_text.get_color(), window.OUT_OF_TUNE)
                    self.assertTrue(window.trace.get_xdata().size)
                streams[0].feed(440)
            elif len(rendered) == 2:
                self.assertTrue(title.endswith(" | A4 +0.0 cents"), title)
                if window is not None:
                    self.assertEqual(window.note_labels[1].get_text(), "A4")
                    self.assertEqual(window.cents_text.get_text(), "+0.0 ct")
                    self.assertEqual(window.cents_text.get_color(), window.IN_TUNE)
                streams[0].feed(440 * 2 ** ((72 - 69 - 29.4 / 100) / 12))
            elif len(rendered) == 3:
                self.assertTrue(title.endswith(" | C5 -29.4 cents"), title)
                streams[0].feed(0)
            else:
                self.assertEqual(title, "Oscilloscope — no stable periodic signal")
                if window is not None:
                    self.assertEqual(window.note_labels[1].get_text(), "—")
                    self.assertEqual(window.frequency_text.get_text(), "No stable pitch")
                    self.assertEqual(len(window.trace.get_xdata()), 0)
                plt.close(window.figure if window is not None else wave)

        with (patch.object(main.sd, "InputStream", FakeInputStream),
              patch.object(main, "TunerWindow", make_tuner),
              patch.object(main.plt, "pause", pause)):
            main.main(["-tuner"] if tuner_enabled else [])
        self.assertEqual(len(rendered), 4)
        self.assertTrue(streams[0].closed)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
