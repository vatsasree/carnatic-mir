"""Frame-level vocal pitch (f0) + loudness / timbre descriptors at a fixed 10 ms hop."""
import numpy as np
import pandas as pd
import librosa

from .config import FMAX, FMIN, HOP_S, SR_ANALYSIS


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


N_BINS, CENTS_PER_BIN, CENTS_BIN0 = 360, 20.0, 1997.3794084376191   # CREPE bin grid (cents re 10 Hz)
_MAX_JUMP = 12                                                     # bins; CREPE's triangular transition width


def _crepe_salience(y: np.ndarray, sr: int, hop: int, model: str, device: str) -> np.ndarray:
    """CREPE network outputs (frames x 360 sigmoid saliences); fp16 autocast on CUDA."""
    import torch
    import torchcrepe

    audio = torch.tensor(y, dtype=torch.float32)[None]
    bs = 2048 if device == "cuda" else 512
    out = []
    with torch.inference_mode():
        for frames in torchcrepe.preprocess(audio, sr, hop, bs, device, pad=True):
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torchcrepe.infer(frames, model, device)
            else:
                p = torchcrepe.infer(frames, model, device)
            out.append(p.float().cpu())
    return torch.cat(out).numpy()


def _mlx_ok() -> bool:
    import os
    if os.environ.get("CARNATIC_MIR_MLX", "1") != "1":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _crepe_salience_mlx(y: np.ndarray, hop: int, model: str, skip_below_db: float = -50.0) -> np.ndarray:
    """Same frames and normalisation as torchcrepe.preprocess. Frames too quiet to ever count as voiced
    (the voicing gate is -45 dB) are skipped and get zero salience; vocal stems are often silent for long stretches."""
    from numpy.lib.stride_tricks import sliding_window_view
    from .crepe_mlx import salience

    win = 1024
    n = 1 + len(y) // hop
    frames = sliding_window_view(np.pad(y, win // 2).astype(np.float32), win)[::hop][:n]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(1))
    active = np.flatnonzero(20 * np.log10(np.maximum(rms, 1e-10) / (rms.max() + 1e-10)) > skip_below_db)
    out = np.zeros((n, N_BINS), np.float32)
    step = 8192                                                     # bound the normalised copy's memory
    for i in range(0, len(active), step):
        idx = active[i:i + step]
        f = frames[idx] - frames[idx].mean(1, keepdims=True)
        f /= np.maximum(f.std(1, ddof=1, keepdims=True), 1e-10)
        out[idx] = salience(f, model, batch=64, dtype="float32")
    return out


def _banded_viterbi(sal: np.ndarray) -> np.ndarray:
    """Viterbi path over the 360 CREPE bins with CREPE's triangular transition (|jump| < 12 bins).
    Exploits the band structure: O(frames x 360 x 23) instead of O(frames x 360^2) -> ~100x faster
    than torchcrepe's librosa-based decoder, and runs over the whole track without batch restarts."""
    n, S = sal.shape
    K = _MAX_JUMP - 1
    offs = np.arange(-K, K + 1)
    w = (_MAX_JUMP - np.abs(offs)).astype(float)
    prev = np.arange(S)[None, :] - offs[:, None]                    # (23, S): previous state for (offset, current)
    valid = (prev >= 0) & (prev < S)
    prev_c = np.clip(prev, 0, S - 1)
    # row-normalised transition from each previous state (edge rows have fewer neighbours)
    rowsum = np.array([(w * ((j + offs >= 0) & (j + offs < S))).sum() for j in range(S)])
    logT = np.where(valid, np.log(w)[:, None] - np.log(rowsum[prev_c]), -np.inf)
    emis = np.log(sal / (sal.sum(1, keepdims=True) + 1e-12) + 1e-12)
    score = emis[0].copy()
    back = np.empty((n, S), np.int16)
    for t in range(1, n):
        cand = score[prev_c] + logT                                 # (23, S)
        k = cand.argmax(0)
        score = cand[k, np.arange(S)] + emis[t]
        back[t] = prev_c[k, np.arange(S)]
    path = np.empty(n, np.int64)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _local_average_cents(sal: np.ndarray, path: np.ndarray, r: int = 4) -> np.ndarray:
    """Sub-bin pitch: salience-weighted mean of the bins within +-r of the path (as in the original CREPE),
    instead of torchcrepe's 20-cent bins + random dither."""
    n = len(path)
    idx = np.clip(path[:, None] + np.arange(-r, r + 1)[None, :], 0, N_BINS - 1)
    w = sal[np.arange(n)[:, None], idx]
    return CENTS_BIN0 + CENTS_PER_BIN * (w * idx).sum(1) / (w.sum(1) + 1e-12)


def f0_crepe(y: np.ndarray, sr: int, hop: int, model: str = "full"):
    from scipy.ndimage import median_filter

    dev = _device()
    try:
        sal = (_crepe_salience_mlx(y, hop, model) if dev == "mps" and _mlx_ok()
               else _crepe_salience(y, sr, hop, model, dev))
    except Exception:          # fall back to CPU if an op is unsupported on this device
        sal = _crepe_salience(y, sr, hop, model, "cpu")
    # restrict to the vocal search range
    bin_cents = CENTS_BIN0 + CENTS_PER_BIN * np.arange(N_BINS)
    bin_hz = 10 * 2 ** (bin_cents / 1200)
    sal = np.where(((bin_hz >= FMIN) & (bin_hz <= FMAX))[None, :], sal, 0.0)
    path = _banded_viterbi(sal)
    f0 = 10 * 2 ** (_local_average_cents(sal, path) / 1200)
    conf = median_filter(sal[np.arange(len(path)), path], size=3, mode="nearest")
    return f0, conf


def f0_pyin(y: np.ndarray, sr: int, hop: int):
    f0, _, prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=2048, hop_length=hop)
    return np.nan_to_num(f0), prob


def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    mask = mask.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < min_len:
            mask[s:e] = False
    return mask


def _fix_octave_jumps(f0: np.ndarray, voiced: np.ndarray, win: int = 50) -> np.ndarray:
    """Shift isolated octave errors back toward the local (0.5 s) median contour."""
    cents = np.where(voiced, 1200 * np.log2(np.maximum(f0, 1e-6) / 55.0), np.nan)
    ref = pd.Series(cents).rolling(win, center=True, min_periods=5).median().to_numpy().copy()
    d = cents - ref
    shift = np.where(d > 900, -1200, np.where(d < -900, 1200, 0))
    shift = np.nan_to_num(shift)
    return np.where(voiced, f0 * 2.0 ** (shift / 1200), 0.0)


def extract_frame_features(vocals_wav, method: str = "crepe", voicing_threshold: float = 0.5,
                           silence_db: float = -45.0) -> pd.DataFrame:
    """Return a DataFrame with one row per 10 ms frame."""
    y, sr = librosa.load(str(vocals_wav), sr=SR_ANALYSIS, mono=True)
    hop = int(round(SR_ANALYSIS * HOP_S))

    if method == "crepe":
        f0, conf = f0_crepe(y, sr, hop)
    elif method == "crepe-tiny":
        f0, conf = f0_crepe(y, sr, hop, model="tiny")
    elif method == "pyin":
        f0, conf = f0_pyin(y, sr, hop)
    else:
        raise ValueError(f"unknown pitch method {method}")

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) + 1e-9)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=hop)[0]

    n = min(len(f0), len(rms))
    f0, conf, rms_db, centroid = f0[:n], conf[:n], rms_db[:n], centroid[:n]

    voiced = (conf >= voicing_threshold) & (rms_db > silence_db) & (f0 > 0)
    voiced = _remove_short_runs(voiced, min_len=6)          # drop < 60 ms blips
    f0 = _fix_octave_jumps(f0, voiced)

    return pd.DataFrame({
        "time": np.arange(n) * HOP_S,
        "f0_hz": np.where(voiced, f0, np.nan),
        "f0_raw_hz": f0,
        "confidence": conf,
        "voiced": voiced,
        "rms_db": rms_db,
        "spectral_centroid_hz": centroid,
    })
