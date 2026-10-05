import numpy as np

def trigger_waveform(samples):

    crossings = np.where(
        (samples[:-1] < 0) &
        (samples[1:] >= 0)
    )[0]

    if len(crossings) == 0:
        return samples

    start = crossings[0]

    result = np.roll(
        samples,
        -start
    )

    return result

if __name__ == '__main__':
    pass