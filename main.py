import argparse
from collections import deque
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

import utils
from tuner import TunerTracker, TunerWindow


SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
BUFFER_SAMPLES = 8192
DISPLAY_SAMPLES = 1024
INPUT_DEVICE = 1
SPECTRUM_SMOOTHING_SECONDS = 0.15
SPECTRUM_FLOOR_DB = -100
TUNER_REFERENCE_HZ = 440


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synth oscilloscope and spectrum")
    parser.add_argument("-tuner", "--tuner", action="store_true",
                        help="also open the note and cents tuner window")
    args = parser.parse_args(argv)

    buffer = deque(np.zeros(BUFFER_SAMPLES), maxlen=BUFFER_SAMPLES)
    buffer_lock = Lock()
    samples_received = 0

    def audio_callback(indata, frames, time, status):
        nonlocal samples_received
        if status:
            print(status)
        with buffer_lock:
            buffer.extend(indata[:, 0])
            samples_received += frames

    plt.ion()
    fig_wave, ax_wave = plt.subplots()
    fig_wave.canvas.manager.set_window_title("Waveform")
    phase = np.linspace(0, 2, DISPLAY_SAMPLES)
    wave_line, = ax_wave.plot(phase, np.zeros(DISPLAY_SAMPLES))
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_xlim(0, 2)
    for position in (0.5, 1.5):
        ax_wave.axvline(position, color="grey", linestyle=":", linewidth=1, zorder=1)
    ax_wave.set_title("Oscilloscope — waiting for a periodic signal")
    ax_wave.set_xlabel("Phase [cycles]")
    ax_wave.set_ylabel("Amplitude")

    analyzer = utils.SpectrumAnalyzer(
        BUFFER_SAMPLES, SAMPLE_RATE,
        smoothing_seconds=SPECTRUM_SMOOTHING_SECONDS,
        floor_db=SPECTRUM_FLOOR_DB,
    )
    fig_fft, ax_fft = plt.subplots()
    fig_fft.canvas.manager.set_window_title("Spectrum")
    fft_line, = ax_fft.plot(
        analyzer.frequencies,
        np.full(len(analyzer.frequencies), SPECTRUM_FLOOR_DB),
    )
    ax_fft.set_xlim(20, 20000)
    ax_fft.set_xscale("log")
    ax_fft.set_ylim(SPECTRUM_FLOOR_DB, 0)
    ax_fft.set_title("Spectrum")
    ax_fft.set_xlabel("Frequency [Hz]")
    ax_fft.set_ylabel("Magnitude [dBFS]")

    tuner = TunerWindow(reference_hz=TUNER_REFERENCE_HZ) if args.tuner else None
    pitch_tracker = TunerTracker(reference_hz=TUNER_REFERENCE_HZ)

    last_samples_received = 0
    try:
        with sd.InputStream(
            device=INPUT_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
        ):
            while (plt.fignum_exists(fig_wave.number)
                   and plt.fignum_exists(fig_fft.number)
                   and (tuner is None or plt.fignum_exists(tuner.figure.number))):
                with buffer_lock:
                    received = samples_received
                    samples = (
                        np.array(buffer)
                        if received != last_samples_received else None
                    )

                if samples is not None:
                    cycle, frequency = utils.trigger_waveform(
                        samples, SAMPLE_RATE, DISPLAY_SAMPLES
                    )
                    elapsed_seconds = (received - last_samples_received) / SAMPLE_RATE
                    reading = (
                        tuner.update(frequency, elapsed_seconds) if tuner is not None
                        else pitch_tracker.update(frequency, elapsed_seconds)
                    )
                    if cycle is None:
                        wave_line.set_ydata(np.full(DISPLAY_SAMPLES, np.nan))
                        ax_wave.set_title("Oscilloscope — no stable periodic signal")
                    else:
                        wave_line.set_ydata(cycle)
                        cents = 0.0 if abs(reading.cents) < 0.05 else reading.cents
                        ax_wave.set_title(
                            f"Oscilloscope — two cycles, {frequency:.1f} Hz"
                            f" | {reading.note} {cents:+.1f} cents"
                        )

                    fft_line.set_ydata(analyzer.process(samples, elapsed_seconds))
                    last_samples_received = received
                    fig_wave.canvas.draw_idle()
                    fig_fft.canvas.draw_idle()

                fig_wave.canvas.flush_events()
                fig_fft.canvas.flush_events()
                if tuner is not None:
                    tuner.figure.canvas.flush_events()
                plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig_wave)
        plt.close(fig_fft)
        if tuner is not None:
            plt.close(tuner.figure)


if __name__ == "__main__":
    main()
