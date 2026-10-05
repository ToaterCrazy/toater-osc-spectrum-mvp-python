import numpy as np


def estimate_period(samples, sample_rate, min_frequency=20, max_frequency=20000):
    """Estimate a monophonic period in samples; return None for silence/noise.

    Use an FFT-accelerated, normalized difference function (YIN-style) so
    extra zero crossings in harmonically rich waves do not shorten the cycle.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.size < 8:
        return None
    centered = samples - np.mean(samples)
    if np.sqrt(np.mean(centered ** 2)) < 1e-4:
        return None

    # Fractional delays matter for high notes: their periods span few samples.
    # Band-limited interpolation avoids picking a later multiple just because
    # the true period falls between two integer lags.
    upsampling = 4
    spectrum = np.fft.rfft(centered)
    if len(centered) % 2 == 0:
        spectrum[-1] *= 0.5
    centered = np.fft.irfft(spectrum, n=len(centered) * upsampling) * upsampling
    sample_rate *= upsampling

    min_lag = max(2, int(sample_rate / max_frequency))
    max_lag = min(int(np.ceil(sample_rate / min_frequency)), len(centered) // 2 - 1)
    if max_lag <= min_lag:
        return None

    # Zero padding makes this a linear correlation, without wrapping the end.
    fft_size = 1 << (2 * len(centered) - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size)
    correlation = np.fft.irfft(np.abs(transform) ** 2, n=fft_size)
    energy = np.concatenate(([0.0], np.cumsum(centered ** 2)))
    # Include a neighbour beyond the lowest supported pitch for refinement.
    lags = np.arange(1, max_lag + 2)
    difference = np.maximum(
        energy[len(centered) - lags] + energy[-1] - energy[lags]
        - 2 * correlation[lags],
        0.0,
    ) / (len(centered) - lags)
    normalized = np.ones(max_lag + 2)
    normalized[1:] = difference * lags / np.maximum(
        np.cumsum(difference), np.finfo(float).tiny
    )

    # Take the first convincing trough, not the deepest later multiple.
    candidates = np.flatnonzero(normalized[min_lag:max_lag + 1] < 0.15)
    if not candidates.size:
        return None
    lag = int(candidates[0] + min_lag)
    while lag < max_lag and normalized[lag + 1] < normalized[lag]:
        lag += 1
    if normalized[lag + 1] < normalized[lag]:
        return None

    left, middle, right = difference[lag - 2:lag + 1]
    curvature = left - 2 * middle + right
    offset = 0.5 * (left - right) / curvature if curvature > 0 else 0.0
    return (lag + float(np.clip(offset, -0.5, 0.5))) / upsampling


def _refine_fundamental_period(samples, period, sample_rate):
    """Refine the detected note without following faster resonant ringing."""
    # With only a few bass cycles, the difference estimate is more reliable.
    if len(samples) < 6 * period:
        return period
    window = np.hanning(len(samples))
    fft_size = 1 << (8 * len(samples) - 1).bit_length()
    magnitude = np.abs(np.fft.rfft(
        (samples - np.mean(samples)) * window, n=fft_size
    ))
    fundamental_bin = fft_size / period
    first = max(1, int(np.ceil(0.8 * fundamental_bin)))
    last = min(len(magnitude) - 2, int(np.floor(1.2 * fundamental_bin)))
    if last < first:
        return period
    peak = first + int(np.argmax(magnitude[first:last + 1]))
    # A small, fast ringing component can repeat before a bass note has moved
    # appreciably. Prefer a substantially stronger lower tone in that case.
    lowest_bin = max(1, int(np.floor(20 * fft_size / sample_rate)))
    strongest = lowest_bin + int(np.argmax(magnitude[lowest_bin:-1]))
    if (strongest < 0.75 * fundamental_bin
            and magnitude[strongest] > 1.5 * magnitude[peak]):
        peak = strongest
    amplitude = 2 * magnitude[peak] / np.sum(window)
    if amplitude < max(1e-4, 0.05 * np.std(samples)):
        return period
    left, middle, right = np.log(np.maximum(magnitude[peak - 1:peak + 2], 1e-20))
    curvature = left - 2 * middle + right
    offset = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
    return fft_size / (peak + float(np.clip(offset, -0.5, 0.5)))


def _fundamental_start(samples, period, latest_start):
    """Fit a smooth fundamental for triggering, independent of harmonic edges."""
    # Use recent complete periods to keep phase responsive as the filter moves.
    count = min(len(samples), int(np.ceil(4 * period)))
    positions = np.arange(len(samples) - count, len(samples))
    angle = 2 * np.pi * positions / period
    basis = np.column_stack((np.cos(angle), np.sin(angle), np.ones(count)))
    weights = np.sqrt(np.hanning(count))
    cosine, sine, _ = np.linalg.lstsq(
        basis * weights[:, None], samples[-count:] * weights, rcond=None
    )[0]
    amplitude = np.hypot(cosine, sine)
    if amplitude < max(1e-4, 0.05 * np.std(samples)):
        return None
    phase = np.arctan2(-sine, cosine)
    first = (-np.pi / 2 - phase) * period / (2 * np.pi)
    start = first + np.floor((latest_start - first) / period) * period
    return float(start) if start >= 0 else None


def trigger_waveform(samples, sample_rate=48000, display_samples=1024):
    """Return (two phase-aligned cycles, frequency), or (None, None).

    The display spans phase 0..2 with a fixed number of interpolated points.
    Silence and nonperiodic input have no meaningful period.
    """
    samples = np.asarray(samples, dtype=float)
    period = estimate_period(samples, sample_rate)
    if period is None:
        return None, None

    period = _refine_fundamental_period(samples, period, sample_rate)

    display_span = 2 * period
    start = _fundamental_start(samples, period, len(samples) - 1 - display_span)
    if start is not None:
        positions = start + np.linspace(0, display_span, display_samples)
        waveform = np.interp(positions, np.arange(len(samples)), samples)
        return waveform, sample_rate / period

    # Missing fundamentals still need a trigger: fall back to waveform edges.
    # Estimate the trigger's DC level over whole periods. A plain buffer mean
    # varies with phase, especially when low notes fit only a few cycles.
    end = len(samples) - 1
    span = np.floor(end / period) * period
    start = end - span
    integration_positions = np.concatenate((
        [start], np.arange(np.ceil(start), end), [end]
    ))
    values = np.interp(integration_positions, np.arange(len(samples)), samples)
    dc_level = np.trapezoid(values, integration_positions) / span
    centered = samples - dc_level
    crossings = np.flatnonzero((centered[:-1] < 0) & (centered[1:] >= 0))
    slopes = centered[crossings + 1] - centered[crossings]
    starts = crossings - centered[crossings] / slopes
    complete = starts + display_span <= len(samples) - 1
    starts, slopes = starts[complete], slopes[complete]
    if not starts.size:
        return None, None

    # Rich waves can cross zero several times per cycle. Lock to their steepest
    # rising edge so successive buffers do not switch between different phases.
    starts = starts[slopes >= 0.95 * np.max(slopes)]

    # Retain the original amplitude and DC offset; centering is for detection.
    positions = starts[-1] + np.linspace(0, display_span, display_samples)
    cycle = np.interp(positions, np.arange(len(samples)), samples)
    return cycle, sample_rate / period


class SpectrumAnalyzer:
    """Windowed, power-averaged spectrum in dBFS (peak amplitude reference)."""

    def __init__(self, sample_count, sample_rate, smoothing_seconds=0.15,
                 floor_db=-100):
        self.sample_count = sample_count
        self.smoothing_seconds = smoothing_seconds
        self.floor_db = floor_db
        self.frequencies = np.fft.rfftfreq(sample_count, 1 / sample_rate)
        phase = 2 * np.pi * np.arange(sample_count) / sample_count
        # Periodic four-term Blackman-Harris: suppress off-bin tone leakage.
        self.window = (
            0.35875 - 0.48829 * np.cos(phase)
            + 0.14128 * np.cos(2 * phase) - 0.01168 * np.cos(3 * phase)
        )
        self.window_gain = np.sum(self.window)
        self.averaged_power = None

    def process(self, samples, elapsed_seconds):
        samples = np.asarray(samples, dtype=float)
        if samples.shape != (self.sample_count,):
            raise ValueError("Spectrum input must match the configured sample count")
        centered = samples - np.mean(samples)
        amplitude = np.abs(np.fft.rfft(centered * self.window)) / self.window_gain
        # Only interior bins have a separate negative-frequency counterpart.
        amplitude[1:-1 if self.sample_count % 2 == 0 else None] *= 2
        power = amplitude ** 2
        if self.averaged_power is None or self.smoothing_seconds <= 0:
            self.averaged_power = power
        else:
            weight = -np.expm1(-max(elapsed_seconds, 0) / self.smoothing_seconds)
            self.averaged_power += weight * (power - self.averaged_power)

        # Absolute scale: quiet noise stays quiet instead of being boosted to 0 dB.
        return 10 * np.log10(np.maximum(
            self.averaged_power, 10 ** (self.floor_db / 10)
        ))
