import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

EEG = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 31),
    'gamma': (31, 51)
}

SAMPLE_RATE = 512

time = np.arange(0, 1, 1 / SAMPLE_RATE)


def raw_to_microvolt(raw: list):
    return [round(((x * (1.8 / 4096)) / 2000) * 1e6, 3) for x in raw] # Neurosky


def microvolt_to_bands(microvolt: list):
    data_fft = rfft(microvolt)
    freq = rfftfreq(len(microvolt), 1 / SAMPLE_RATE)
    bands = {}
    for band, (low, high) in EEG.items():
        idx = np.where(np.logical_or(freq < low,
                                     freq >= high))
        tmp = data_fft.copy()
        tmp[idx] = 0
        inverse = irfft(tmp)
        bands[band] = inverse.tolist()
    return bands


def bands_to_waves(bands: dict):
    waves = {}
    for band, data in bands.items():
        waves[band] = [(x, y) for x, y in zip(time, data)]
    return waves


def raw_to_waves(raw: list):
    return bands_to_waves(microvolt_to_bands(raw_to_microvolt(raw)))


def calculate_sum_waves(waves_1: list, waves_2: list):
    x1, y1 = zip(*waves_1)
    x2, y2 = zip(*waves_2)
    waves_1 = np.array(y1)
    waves_2 = np.array(y2)
    resonance = (waves_1 + waves_2).tolist()
    return [(x, y) for x, y in zip(x1, resonance)]

def wave_fft(wave):
    data_fft = rfft(wave)
    freq = rfftfreq(len(wave), 1/SAMPLE_RATE)
    return freq, np.abs(data_fft)[:len(freq)//2]
