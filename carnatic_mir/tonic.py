"""Tonic (Sa) identification."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def tonic_essentia(mix_wav) -> float:
    """Essentia's multipitch/drone-based tonic estimator (Salamon et al. / Gulati et al.).
    Run on the FULL MIX: it relies on the tanpura drone, which separation removes from vocals."""
    import essentia.standard as es

    audio = es.MonoLoader(filename=str(mix_wav), sampleRate=44100)()
    return float(es.TonicIndianArtMusic()(audio))


def tonic_from_pitch(f0_hz: np.ndarray) -> float:
    """Heuristic fallback from the vocal pitch: Sa and Pa are the two most sustained svaras,
    so pick the pitch class c maximising hist(c) + hist(c + 702 cents)."""
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    cents = 1200 * np.log2(f0 / 55.0)
    hist, _ = np.histogram(np.mod(cents, 1200), bins=240, range=(0, 1200))   # 5-cent bins
    hist = gaussian_filter1d(hist.astype(float), 2, mode="wrap")
    score = hist + np.roll(hist, -int(round(702 / 5)))
    pc = np.argmax(score) * 5.0
    # choose the octave: Sa usually sits roughly a fifth below the singer's median pitch
    target = np.median(cents) - 600
    k = np.round((target - pc) / 1200)
    return float(55.0 * 2 ** ((pc + 1200 * k) / 1200))


def fit_tonic_octave(tonic_hz: float, f0_hz: np.ndarray) -> float:
    """Move a tonic estimate to the octave that best matches the singer's register."""
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if len(f0) == 0:
        return tonic_hz
    med = np.median(1200 * np.log2(f0 / tonic_hz))
    if -300 <= med <= 1500:          # plausible already; trust the estimator's octave
        return float(tonic_hz)
    k = np.round((med - 600) / 1200)
    return float(tonic_hz * 2 ** k)
