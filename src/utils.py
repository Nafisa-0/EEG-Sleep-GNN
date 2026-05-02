import logging
import numpy as np
from scipy.signal import welch
from config import BANDS, LOG_PATH


def setup_logger():
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

def log(msg):
    print(msg)
    logging.info(msg)


def map_label(desc):
    if   "Sleep stage W" in desc:                             return 0
    elif "Sleep stage 1" in desc:                             return 1
    elif "Sleep stage 2" in desc:                             return 2
    elif "Sleep stage 3" in desc or "Sleep stage 4" in desc: return 3
    elif "Sleep stage R" in desc:                             return 4
    else:                                                     return None


# 10 features per channel
def extract_features(signal, fs=100):
    freqs, psd = welch(signal, fs=fs, nperseg=fs * 2)

    band_powers = []
    for _, lo, hi in BANDS:
        idx = (freqs >= lo) & (freqs <= hi)
        bp  = np.mean(psd[idx]) if idx.any() else 1e-10
        band_powers.append(bp)

    total      = sum(band_powers) + 1e-10
    log_ratios = [np.log(bp / total + 1e-10) for bp in band_powers]

    psd_norm     = psd / (psd.sum() + 1e-10)
    spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    diff1      = np.diff(signal)
    diff2      = np.diff(diff1)
    activity   = np.var(signal) + 1e-10
    mobility   = np.sqrt(np.var(diff1) / activity)
    complexity = (np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))) / (mobility + 1e-10)

    rms = np.sqrt(np.mean(signal ** 2))

    return np.array(log_ratios + [spec_entropy, activity, mobility, complexity, rms])


def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum() / max(len(signal), 1)

def eog_extra_features(raw_eog):
    zcr      = zero_crossing_rate(raw_eog)
    peak_amp = np.max(np.abs(raw_eog))
    return np.array([zcr, peak_amp])