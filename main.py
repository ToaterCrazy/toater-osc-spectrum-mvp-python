import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from collections import deque
import utils as utils

# print(sd.query_devices())

SAMPLE_RATE = 48000
BLOCK_SIZE = 1024

# How much audio to show in oscilloscope
DISPLAY_SAMPLES = 2048*4

buffer = deque(
    np.zeros(DISPLAY_SAMPLES),
    maxlen=DISPLAY_SAMPLES
)


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    # Take channel 1
    samples = indata[:, 0]

    buffer.extend(samples)


# ----------------------------
# Plot setup
# ----------------------------

plt.ion()

fig_wave, ax_wave = plt.subplots()
fig_wave.canvas.manager.set_window_title("Waveform")

x_wave = np.arange(DISPLAY_SAMPLES)

wave_line, = ax_wave.plot(
    x_wave,
    np.zeros(DISPLAY_SAMPLES)
)

ax_wave.set_ylim(-1, 1)
ax_wave.set_xlim(0, DISPLAY_SAMPLES)
ax_wave.set_title("Oscilloscope")


fig_fft, ax_fft = plt.subplots()
fig_fft.canvas.manager.set_window_title("Spectrum")

freqs = np.fft.rfftfreq(
    DISPLAY_SAMPLES,
    1 / SAMPLE_RATE
)

fft_line, = ax_fft.plot(
    freqs,
    np.zeros(len(freqs))
)

ax_fft.set_xlim(20, 20000)
ax_fft.set_xscale("log")

ax_fft.set_ylim(-100, 0)

ax_fft.set_title("Spectrum")
ax_fft.set_xlabel("Frequency [Hz]")
ax_fft.set_ylabel("Magnitude [dB]")


# ----------------------------
# Audio stream
# ----------------------------

with sd.InputStream(
    device=1,
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCK_SIZE,
    callback=audio_callback
):

    while True:

        samples = np.array(buffer)


        # ----------------
        # Waveform
        # ----------------

        # display_samples = utils.trigger_waveform(samples)
        # wave_line.set_ydata(display_samples)

        wave_line.set_ydata(samples)

        fig_wave.canvas.draw_idle()
        fig_wave.canvas.flush_events()


        # ----------------
        # FFT
        # ----------------

        # Apply window before FFT
        window = np.hanning(len(samples))

        spectrum = np.fft.rfft(
            samples * window
        )

        magnitude = np.abs(spectrum)

        # Avoid log(0)
        magnitude = np.maximum(
            magnitude,
            1e-10
        )

        magnitude_db = (
            20 * np.log10(magnitude)
        )

        # Normalize display
        magnitude_db -= np.max(magnitude_db)

        fft_line.set_ydata(
            magnitude_db
        )

        fig_fft.canvas.draw_idle()
        fig_fft.canvas.flush_events()

        plt.pause(0.01)