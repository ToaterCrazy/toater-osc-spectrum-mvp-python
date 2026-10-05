from collections import deque
from dataclasses import dataclass
from math import exp, floor, isfinite, log2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


NOTE_NAMES = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")


def note_name(midi_note):
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


@dataclass(frozen=True)
class PitchReading:
    note: str
    midi_note: int
    midi_pitch: float
    cents: float
    frequency: float
    target_frequency: float


def _reading(midi_pitch, midi_note, reference_hz):
    return PitchReading(
        note=note_name(midi_note),
        midi_note=midi_note,
        midi_pitch=midi_pitch,
        cents=100 * (midi_pitch - midi_note),
        frequency=reference_hz * 2 ** ((midi_pitch - 69) / 12),
        target_frequency=reference_hz * 2 ** ((midi_note - 69) / 12),
    )


def pitch_to_note(frequency, reference_hz=440):
    """Map a frequency to the nearest equal-tempered note, with A4 as reference."""
    if not isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError("Tuner reference frequency must be positive and finite")
    if frequency is None or not isfinite(frequency) or frequency <= 0:
        return None
    midi_pitch = 69 + 12 * log2(frequency / reference_hz)
    return _reading(midi_pitch, floor(midi_pitch + 0.5), reference_hz)


class TunerTracker:
    """Smooth small pitch fluctuations and avoid note flicker near boundaries."""

    def __init__(self, reference_hz=440, smoothing_seconds=0.08):
        pitch_to_note(None, reference_hz)  # Validate the configured reference.
        self.reference_hz = reference_hz
        self.smoothing_seconds = smoothing_seconds
        self.midi_pitch = None
        self.midi_note = None

    def update(self, frequency, elapsed_seconds):
        reading = pitch_to_note(frequency, self.reference_hz)
        if reading is None:
            self.midi_pitch = self.midi_note = None
            return None

        # Respond immediately to new notes; smooth fine tuning within a note.
        if (self.midi_pitch is None or self.smoothing_seconds <= 0
                or abs(reading.midi_pitch - self.midi_pitch) > 0.5):
            self.midi_pitch = reading.midi_pitch
        else:
            weight = 1 - exp(-max(elapsed_seconds, 0) / self.smoothing_seconds)
            self.midi_pitch += weight * (reading.midi_pitch - self.midi_pitch)

        # Five cents of hysteresis keeps a boundary from alternating labels.
        if self.midi_note is None or abs(self.midi_pitch - self.midi_note) > 0.55:
            self.midi_note = floor(self.midi_pitch + 0.5)
        return _reading(self.midi_pitch, self.midi_note, self.reference_hz)


class TunerWindow:
    """A compact note and cents display with a scrolling pitch history."""

    BACKGROUND = "#191b1e"
    GRID = "#35383d"
    MUTED = "#8c949f"
    TEXT = "#e5e9ee"
    IN_TUNE = "#70d6a1"
    OUT_OF_TUNE = "#ff696e"

    def __init__(self, reference_hz=440, history_seconds=5, smoothing_seconds=0.08):
        self.tracker = TunerTracker(reference_hz, smoothing_seconds)
        self.history_seconds = history_seconds
        self.time = 0.0
        self.history = deque()
        self.figure = plt.figure(figsize=(6.4, 3.2), facecolor=self.BACKGROUND)
        self.figure.canvas.manager.set_window_title("Tuner")
        self.figure.text(0.05, 0.93, "TUNER", color=self.TEXT, fontsize=11, weight="bold")
        self.figure.text(0.95, 0.93, "AUTO · CENTS", ha="right", color=self.MUTED, fontsize=9)

        self.axes = self.figure.add_axes([0.05, 0.22, 0.77, 0.64], facecolor=self.BACKGROUND)
        self.axes.set_xlim(-history_seconds, 0)
        self.axes.set_ylim(-1.5, 1.5)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for spine in self.axes.spines.values():
            spine.set_color(self.GRID)
        for level in (-0.5, 0.5):
            self.axes.axhline(level, color=self.GRID, linewidth=0.8)
        self.axes.axhline(0, color=self.MUTED, linewidth=0.8, alpha=0.5)
        self.axes.axhspan(-0.05, 0.05, color=self.IN_TUNE, alpha=0.07)
        self.trace, = self.axes.plot([], [], color=self.TEXT, linewidth=1.4)
        self.marker, = self.axes.plot([], [], "o", color=self.MUTED, markersize=5)
        self.cents_text = self.axes.text(
            0.55, 0.26, "— ct", transform=self.axes.transAxes,
            ha="center", color=self.MUTED, fontsize=16, weight="bold",
        )

        self.note_axes = self.figure.add_axes([0.83, 0.22, 0.12, 0.64],
                                             facecolor=self.BACKGROUND)
        self.note_axes.set_xlim(0, 1)
        self.note_axes.set_ylim(-1.5, 1.5)
        self.note_axes.set_xticks([])
        self.note_axes.set_yticks([])
        for spine in self.note_axes.spines.values():
            spine.set_color(self.GRID)
        self.note_highlight = Rectangle((0, -0.5), 1, 1, color=self.GRID)
        self.note_axes.add_patch(self.note_highlight)
        self.note_labels = [
            self.note_axes.text(0.5, level, "—", ha="center", va="center",
                                color=self.MUTED, fontsize=12)
            for level in (1, 0, -1)
        ]
        self.note_labels[1].set_fontsize(20)
        self.note_labels[1].set_weight("bold")
        self.frequency_text = self.figure.text(
            0.05, 0.09, "Play a single note", color=self.MUTED, fontsize=10
        )
        self.figure.text(0.95, 0.09, f"Reference A4  {reference_hz:g} Hz",
                         ha="right", color=self.MUTED, fontsize=10)

    def update(self, frequency, elapsed_seconds):
        self.time += max(elapsed_seconds, 0)
        reading = self.tracker.update(frequency, elapsed_seconds)
        self.history.append((self.time, np.nan if reading is None else reading.midi_pitch))
        while self.history and self.history[0][0] < self.time - self.history_seconds:
            self.history.popleft()

        if reading is None:
            self.trace.set_data([], [])
            self.marker.set_data([], [])
            self.cents_text.set_text("— ct")
            self.cents_text.set_color(self.MUTED)
            self.frequency_text.set_text("No stable pitch")
            self.note_highlight.set_color(self.GRID)
            for label in self.note_labels:
                label.set_text("—")
                label.set_color(self.MUTED)
        else:
            color = self.IN_TUNE if abs(reading.cents) <= 5 else self.OUT_OF_TUNE
            times, pitches = np.asarray(self.history).T
            self.trace.set_data(times - self.time, pitches - reading.midi_note)
            self.marker.set_data([0], [reading.cents / 100])
            self.marker.set_color(color)
            cents = 0.0 if abs(reading.cents) < 0.05 else reading.cents
            self.cents_text.set_text(f"{cents:+.1f} ct")
            self.cents_text.set_color(color)
            self.frequency_text.set_text(f"{reading.note}  ·  {reading.frequency:.2f} Hz")
            self.note_highlight.set_color(color)
            for label, offset in zip(self.note_labels, (1, 0, -1)):
                label.set_text(note_name(reading.midi_note + offset))
                label.set_color(self.BACKGROUND if offset == 0 else self.MUTED)

        self.figure.canvas.draw_idle()
        return reading
