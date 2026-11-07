"""Vocal timbre / voice-quality / softness descriptors (tonic-independent).

Frame level (10 ms, aligned with frames.csv), from the 44.1 kHz vocal stem:
    rms_dbfs            absolute level (only meaningful *within* a recording)
    centroid_hz         brightness (inflated by breath noise; read together with harmonic tilt)
    harmonic_tilt_db_per_oct  slope of harmonic peaks up to 5 kHz - steeper (more negative) = darker/softer
    flatness            noisiness (breathy / airy voices are flatter)
    alpha_ratio_db      E(1-5 kHz) / E(50 Hz-1 kHz)            - vocal effort, higher = pressed/loud
    hammarberg_db       max dB(0-2 kHz) - max dB(2-5 kHz)        - higher = softer / less high energy
    tilt_db_per_khz     slope of the 50 Hz-5 kHz spectrum          - steeper (more negative) = softer
    h1h2_db             1st minus 2nd harmonic amplitude (uncorrected) - higher = breathier / laxer
    cpp_db              cepstral peak prominence                   - lower = breathier / less periodic
    hnr_db              harmonics-to-noise ratio (Praat, cc)
    mfcc_1..13          timbre fingerprint
Segment level (held notes >= 0.3 s, Praat):  jitter_local, shimmer_local.

Caveat: separation residue (violin, tanpura) lowers HNR/CPP and raises jitter/shimmer;
compare singers on recordings with similar accompaniment, or validate on clean vocal tracks.
"""
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from .config import FMAX, FMIN, HOP_S

SR = 44100
HOP = int(SR * HOP_S)
NFFT = 2048


def _band(freqs, lo, hi):
    return (freqs >= lo) & (freqs < hi)


def extract_voice_frames(vocals_wav, frames: pd.DataFrame) -> pd.DataFrame:
    y, sr = sf.read(str(vocals_wav), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    n = len(frames)
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, center=True))[:, :n]
    if S.shape[1] < n:
        S = np.pad(S, ((0, 0), (0, n - S.shape[1])))
    P = S ** 2 + 1e-12
    dB = 10 * np.log10(P)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    f0 = frames["f0_hz"].to_numpy(float)
    voiced = frames["voiced"].to_numpy(bool)

    rms = librosa.feature.rms(S=S, frame_length=NFFT)[0]
    out = {"rms_dbfs": 20 * np.log10(rms + 1e-9)}
    out["centroid_hz"] = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    out["flatness"] = librosa.feature.spectral_flatness(S=S)[0]
    lo, hi = _band(freqs, 50, 1000), _band(freqs, 1000, 5000)
    out["alpha_ratio_db"] = 10 * np.log10(P[hi].sum(0) / P[lo].sum(0))
    out["hammarberg_db"] = dB[_band(freqs, 0, 2000)].max(0) - dB[_band(freqs, 2000, 5000)].max(0)
    tb = _band(freqs, 50, 5000)
    x = freqs[tb] / 1000.0
    xc = x - x.mean()
    out["tilt_db_per_khz"] = (xc[:, None] * (dB[tb] - dB[tb].mean(0))).sum(0) / (xc ** 2).sum()

    # harmonic amplitudes on voiced frames -> H1-H2 and harmonic spectral tilt (dB / octave, up to 5 kHz)
    df_ = freqs[1]
    h1h2 = np.full(n, np.nan)
    htilt = np.full(n, np.nan)
    for i in np.flatnonzero(voiced & np.isfinite(f0)):
        w = max(1, int(0.15 * f0[i] / df_))
        H = np.arange(1, int(5000 // f0[i]) + 1)
        if len(H) < 3:
            continue
        ks = np.round(H * f0[i] / df_).astype(int)
        amps = np.array([dB[max(0, k - w):k + w + 1, i].max() for k in ks])
        h1h2[i] = amps[0] - amps[1]
        htilt[i] = np.polyfit(np.log2(H), amps, 1)[0]
    out["h1h2_db"] = h1h2
    out["harmonic_tilt_db_per_oct"] = htilt

    # cepstral peak prominence on voiced frames (chunked to bound memory)
    cpp = np.full(n, np.nan)
    qlo, qhi = int(SR / FMAX), int(SR / FMIN)
    q = np.arange(qlo, qhi) / SR
    vidx = np.flatnonzero(voiced)
    for c0 in range(0, len(vidx), 4000):
        idx = vidx[c0:c0 + 4000]
        spec_db = 20 * np.log10(S[:, idx].T + 1e-9)
        ceps = 20 * np.log10(np.abs(np.fft.irfft(spec_db, axis=1))[:, qlo:qhi] + 1e-9)
        A = np.vstack([q, np.ones_like(q)]).T
        coef, *_ = np.linalg.lstsq(A, ceps.T, rcond=None)
        k = ceps.argmax(1)
        cpp[idx] = ceps[np.arange(len(idx)), k] - (coef[0] * q[k] + coef[1])
    out["cpp_db"] = cpp

    # HNR via Praat (22.05 kHz is plenty)
    try:
        import parselmouth
        y22 = librosa.resample(y, orig_sr=SR, target_sr=22050)
        harm = parselmouth.Sound(y22, 22050).to_harmonicity_cc(time_step=HOP_S, minimum_pitch=FMIN)
        hv = harm.values[0]
        hv[hv < -100] = np.nan
        out["hnr_db"] = np.interp(np.arange(n) * HOP_S, harm.xs(), hv, left=np.nan, right=np.nan)
    except Exception:  # noqa: BLE001
        out["hnr_db"] = np.full(n, np.nan)

    mel = librosa.feature.melspectrogram(S=P, sr=SR, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    for k in range(13):
        out[f"mfcc_{k + 1}"] = mfcc[k]

    df = pd.DataFrame({k: np.asarray(v)[:n] for k, v in out.items()})
    for c in df.columns:
        if c not in ("rms_dbfs",):
            df.loc[~voiced, c] = np.nan
    return df


def jitter_shimmer(vocals_wav, segments: list[dict], frames: pd.DataFrame, min_dur: float = 0.3) -> pd.DataFrame:
    """Praat local jitter / shimmer on held notes (sustained phonation is where they are valid)."""
    import parselmouth
    from parselmouth.praat import call

    y, sr = sf.read(str(vocals_wav), dtype="float32", always_2d=True)
    snd = parselmouth.Sound(y.mean(axis=1), sr)
    f0 = frames["f0_hz"].to_numpy(float)
    rows = []
    for s in segments:
        if s["duration"] < min_dur:
            continue
        a, b = int(s["start"] / HOP_S), int(s["end"] / HOP_S)
        fm = np.nanmedian(f0[a:b])
        if not np.isfinite(fm):
            continue
        try:
            part = snd.extract_part(from_time=s["start"], to_time=s["end"], preserve_times=False)
            pp = call(part, "To PointProcess (periodic, cc)", 0.7 * fm, 1.4 * fm)
            jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shim = call([part, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception:  # noqa: BLE001
            continue
        rows.append({"start": s["start"], "end": s["end"], "jitter_local": jit, "shimmer_local": shim})
    return pd.DataFrame(rows, columns=["start", "end", "jitter_local", "shimmer_local"])


def _modulation(x: np.ndarray, lo=2.0, hi=12.0):
    """Depth (std) and dominant rate (Hz) of the detrended signal x (10 ms frames)."""
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return np.nan, np.nan
    tr = pd.Series(x).rolling(15, center=True, min_periods=1).mean().to_numpy()
    d = x - tr
    spec = np.abs(np.fft.rfft(d * np.hanning(len(d)), n=max(256, len(d))))
    fr = np.fft.rfftfreq(max(256, len(d)), HOP_S)
    band = (fr >= lo) & (fr <= hi)
    return float(np.std(d)), float(fr[band][np.argmax(spec[band])]) if band.any() else np.nan


def summarize(vf: pd.DataFrame, js: pd.DataFrame, frames: pd.DataFrame, mel: dict,
              accomp_dbfs: np.ndarray | None) -> tuple[dict, list]:
    """Combine frame/segment voice features with melodic segmentation into a summary group."""
    from .melodic import _r

    voiced = frames["voiced"].to_numpy(bool)
    stable = mel["timeseries"]["stable"].to_numpy(bool)
    cents = mel["timeseries"]["cents"].to_numpy(float)
    med = lambda c, m: _r(np.nanmedian(vf.loc[m, c]), 2) if m.any() else None  # noqa: E731

    # micro-modulation inside held notes (vibrato-like flutter, amplitude tremolo)
    pm_d, pm_r, am_d, am_r = [], [], [], []
    for s in mel["events"]["stable_notes"]:
        a, b = int(s["start"] / HOP_S), int(s["end"] / HOP_S)
        if b - a < 30:
            continue
        d, r = _modulation(cents[a:b]); pm_d.append(d); pm_r.append(r)
        d, r = _modulation(vf["rms_dbfs"].to_numpy()[a:b]); am_d.append(d); am_r.append(r)

    # attack time at phrase onsets: onset -> 90 % of peak RMS within 400 ms
    lin = 10 ** (vf["rms_dbfs"].to_numpy() / 20)
    attacks, slopes = [], []
    for ph in mel["events"]["phrases"]:
        a = int(ph["start"] / HOP_S)
        seg = lin[a:a + 40]
        if len(seg) < 10:
            continue
        k90 = int(np.argmax(seg >= 0.9 * seg.max()))
        attacks.append(k90 * HOP_S * 1000)
        db = 20 * np.log10(seg + 1e-9)
        slopes.append((db[k90] - db[0]) / max(k90 * HOP_S, HOP_S))

    lvl = vf.loc[voiced, "rms_dbfs"]
    summary = {
        "brightness_centroid_hz": med("centroid_hz", voiced),
        "brightness_variability_hz": _r(np.nanstd(vf.loc[voiced, "centroid_hz"]), 1) if voiced.any() else None,
        "spectral_flatness": _r(np.nanmedian(vf.loc[voiced, "flatness"]), 4) if voiced.any() else None,
        "alpha_ratio_db": med("alpha_ratio_db", voiced),
        "hammarberg_db": med("hammarberg_db", voiced),
        "harmonic_tilt_db_per_oct": med("harmonic_tilt_db_per_oct", voiced),
        "spectral_tilt_db_per_khz": med("tilt_db_per_khz", voiced),
        "h1h2_db_held_notes": med("h1h2_db", stable),
        "cpp_db_held_notes": med("cpp_db", stable),
        "hnr_db_held_notes": med("hnr_db", stable),
        "jitter_local_pct": _r(100 * js["jitter_local"].median(), 3) if len(js) else None,
        "shimmer_local_pct": _r(100 * js["shimmer_local"].median(), 3) if len(js) else None,
        "held_note_pitch_flutter_cents": _r(np.nanmedian(pm_d), 1) if pm_d else None,
        "held_note_pitch_flutter_rate_hz": _r(np.nanmedian(pm_r), 2) if pm_r else None,
        "held_note_tremolo_db": _r(np.nanmedian(am_d), 2) if am_d else None,
        "loudness_range_db_p5_p95": _r(np.percentile(lvl, 95) - np.percentile(lvl, 5), 1) if len(lvl) else None,
        "loudness_std_db": _r(lvl.std(), 2) if len(lvl) else None,
        "phrase_attack_time_ms": _r(np.median(attacks), 0) if attacks else None,
        "phrase_attack_slope_db_s": _r(np.median(slopes), 1) if slopes else None,
        "vocal_to_accomp_db": _r(np.nanmedian(vf["rms_dbfs"].to_numpy()[voiced] - accomp_dbfs[:len(vf)][voiced]), 1)
        if accomp_dbfs is not None and voiced.any() else None,
    }
    mfcc_mean = [_r(np.nanmean(vf.loc[voiced, f"mfcc_{k}"]), 3) for k in range(1, 14)] if voiced.any() else []
    return summary, mfcc_mean
