# Synth oscilloscope, spectrum, and tuner

Run with the existing environment:

```sh
.venv/bin/python main.py
```

By default this opens the oscilloscope and spectrum windows. Add `-tuner` to
also open the tuner (`--tuner` is supported too):

```sh
.venv/bin/python main.py -tuner
```

Set `INPUT_DEVICE` in `main.py` to your synth's audio input/loopback device.
List devices with `.venv/bin/python -c 'import sounddevice; print(sounddevice.query_devices())'`.
The analyzer reads the first input channel at 48 kHz. Close any window or press
Ctrl+C to stop.

The oscilloscope detects a periodic, monophonic signal and displays two cycles
on a phase axis from 0 to 2, preserving the input amplitude. Detection covers
20 Hz to 20 kHz with an 8192-sample analysis buffer. Silence and input without a
stable period clear the trace. Chords, noise, and fast pitch/envelope changes
may not produce a stable cycle. Very high notes have only a few captured samples
per cycle, so their drawn shape is less detailed.
When a note is detected, the oscilloscope title ends with its note and signed
tuning offset, for example `| A4 +12.3 cents`, even without the tuner window.

For resonant filters, the scope refines the period around the detected note and
uses its fundamental component as a smooth phase reference. Fast ringing no
longer supplies competing trigger edges. The displayed trace still comes from
the original input samples, so resonance and waveform detail remain visible.
When the fundamental is absent, the scope falls back to waveform edges.
Independent filter self-oscillation or modulation can still move the ripples
relative to the note; that motion is present in the audio itself.

With `-tuner`, the tuner opens in a third window and shares the scope's pitch detection. It
shows the note and octave, its frequency, and signed cents from the target:
positive is sharp, negative is flat. A scrolling trace shows recent pitch with
the current note in the center and neighboring notes at the right. Green means
within ±5 cents; red means out of tune. Small fluctuations are smoothed and a
5-cent margin prevents note labels from flickering at a semitone boundary.
Silence or an unstable pitch clears the current reading. The tuner is intended
for single notes. Change `TUNER_REFERENCE_HZ` in `main.py` to adjust the reference
(default A4 = 440 Hz). Octaves use scientific pitch notation: middle C is C4.
The layout follows the note/cents view of [Ableton's Tuner](https://www.ableton.com/en/live-manual/11/live-audio-effect-reference/#tuner).

The spectrum uses an absolute peak-amplitude dBFS scale: a full-scale sine
centered on an FFT bin reads 0 dBFS, and a half-amplitude sine reads about
-6 dBFS. Quiet input is no longer boosted to 0 dB. A periodic four-term
Blackman-Harris window reduces leakage, and power averaging reduces flicker.
This window broadens peaks slightly. Real input noise remains visible at its
actual level; there is no noise gate. `SPECTRUM_SMOOTHING_SECONDS` (default 0.15)
controls averaging, and `SPECTRUM_FLOOR_DB` (default -100) sets the display floor.
See the [Blackman-Harris window documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackmanharris.html)
and [NumPy real FFT conventions](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
for the window and frequency-bin definitions.

Run synthetic signal regression checks:

```sh
.venv/bin/python -B -m unittest -v
```
