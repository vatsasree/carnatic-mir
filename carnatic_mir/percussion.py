"""Percussion (mridangam / ghatam / kanjira) tempo and 'softness' descriptors.

The accompaniment stem still contains violin + tanpura, so we isolate the percussive part with
harmonic-percussive source separation (HPSS, median filtering). Mridangam's tonal strokes ring at
the tonic and are partly assigned to the harmonic part; their *attacks* stay in the percussive part,
which is what the softness measures need.

Tempo: Carnatic tempo is ambiguous by factors of 2/4 (strokes subdivide the akshara), so we report a
local tempo curve, the global median, and the top tempogram candidates. Sections without percussion
(alapana, viruttam, tanam) are masked out via `active`.
"""
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import find_peaks

from .config import HOP_S

SR = 32000
HOP = int(SR * HOP_S)


def extract_percussion(accomp_wav, out_dir, n_frames: int) -> dict:
    y, _ = librosa.load(str(accomp_wav), sr=SR, mono=True)
    D = librosa.stft(y, n_fft=2048, hop_length=HOP)
    _, Pc = librosa.decompose.hpss(D, kernel_size=(17, 31), margin=1.0)
    yp = librosa.istft(Pc, hop_length=HOP, length=len(y))
    sf.write(str(out_dir / "percussion.wav"), yp, SR, subtype="PCM_16")

    def fit(a):
        a = np.asarray(a)[:n_frames]
        return np.pad(a, (0, n_frames - len(a)), constant_values=a[-1] if len(a) else 0)

    p_db = fit(20 * np.log10(librosa.feature.rms(y=yp, frame_length=1024, hop_length=HOP)[0] + 1e-9))
    a_db = fit(20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0] + 1e-9))
    env = fit(librosa.onset.onset_strength(y=yp, sr=SR, hop_length=HOP))

    # activity: percussion energy (2 s smoothing) close to its loud level and not buried in the accompaniment
    k = int(2.0 / HOP_S)
    sm = lambda d: 10 * np.log10(np.convolve(10 ** (d / 10), np.ones(k) / k, mode="same") + 1e-12)  # noqa: E731
    ps, as_ = sm(p_db), sm(a_db)
    active = (ps > np.percentile(ps, 95) - 25) & (ps > as_ - 25)
    active = np.convolve(active, np.ones(k), mode="same") > k / 2

    local = fit(librosa.feature.tempo(onset_envelope=env, sr=SR, hop_length=HOP, aggregate=None, ac_size=8.0))
    local = np.where(active, local, np.nan)
    win = 384
    tg = librosa.feature.tempogram(onset_envelope=env, sr=SR, hop_length=HOP, win_length=win)
    bpms = librosa.tempo_frequencies(win, hop_length=HOP, sr=SR)
    m = active[:tg.shape[1]] if active.any() else np.ones(tg.shape[1], bool)
    prof = tg[:, m].mean(1)
    ok = (bpms >= 30) & (bpms <= 320)
    pk, pr = find_peaks(np.where(ok, prof, 0), prominence=0.01)
    cands = [round(float(bpms[i]), 1) for i in pk[np.argsort(-prof[pk])][:3]]

    # strokes
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="frames")
    on = on[active[np.clip(on, 0, n_frames - 1)]]
    rows = []
    w2 = int(0.002 * SR)
    freqs = np.fft.rfftfreq(int(0.1 * SR), 1 / SR)
    n_att = int(0.03 * SR)
    afreqs = np.fft.rfftfreq(n_att, 1 / SR)
    for f in on:
        s0, s1 = int(f * HOP - 0.03 * SR), int(f * HOP + 0.12 * SR)
        if s0 < 0 or s1 > len(yp):
            continue
        seg = yp[s0:s1]
        e = np.sqrt(np.convolve(seg ** 2, np.ones(w2) / w2, mode="same"))
        pk_i = int(np.argmax(e))
        peak = e[pk_i]
        pre = e[:pk_i + 1]
        below = np.flatnonzero(pre < 0.1 * peak)
        i10 = below[-1] if len(below) else 0
        i90 = i10 + int(np.argmax(e[i10:pk_i + 1] >= 0.9 * peak))
        body = yp[s0 + pk_i:s0 + pk_i + int(0.1 * SR)]
        if len(body) < int(0.1 * SR):
            continue
        spec = np.abs(np.fft.rfft(body * np.hanning(len(body)))) ** 2
        tot = spec.sum() + 1e-12
        att = yp[s0 + i10:s0 + i10 + n_att]                  # first 30 ms: where 'hardness' is heard
        aspec = np.abs(np.fft.rfft(att * np.hanning(len(att)), n=n_att)) ** 2
        rows.append({
            "time": round((s0 + pk_i) / SR, 3),
            "peak_dbfs": 20 * np.log10(peak + 1e-9),
            "attack_ms": (i90 - i10) / SR * 1000,
            "centroid_hz": float((afreqs * aspec).sum() / (aspec.sum() + 1e-12)),
            "crest_db": 20 * np.log10(np.abs(body).max() / (np.sqrt((body ** 2).mean()) + 1e-12)),
            "low_ratio": float(spec[freqs < 250].sum() / tot),
            "high_ratio": float(spec[freqs > 2000].sum() / tot),
        })
    strokes = pd.DataFrame(rows, columns=["time", "peak_dbfs", "attack_ms", "centroid_hz", "crest_db",
                                          "low_ratio", "high_ratio"])
    if len(strokes):   # drop junk onsets (drone plucks, ringing tails) far below the loud strokes
        p90 = strokes["peak_dbfs"].quantile(0.9)
        strokes = strokes[strokes["peak_dbfs"] >= p90 - 25].reset_index(drop=True)
        strokes["principal"] = strokes["peak_dbfs"] >= p90 - 15
    frames = pd.DataFrame({"perc_dbfs": p_db, "accomp_dbfs": a_db, "onset_env": env / (env.max() + 1e-9),
                           "local_tempo_bpm": local, "perc_active": active})
    frames.to_csv(out_dir / "percussion_frames.csv", index=False, float_format="%.3f")
    strokes.to_csv(out_dir / "strokes.csv", index=False, float_format="%.4f")
    return {"tempo_candidates_bpm": cands}


def summarize(pf: pd.DataFrame, strokes: pd.DataFrame, info: dict, vocal_dbfs: np.ndarray, voiced: np.ndarray) -> dict:
    from .melodic import _r

    act = pf["perc_active"].to_numpy(bool)
    at = act.sum() * HOP_S
    lt = pf["local_tempo_bpm"].to_numpy(float)
    lt = lt[np.isfinite(lt)]
    both = act & voiced[:len(act)]
    allst = strokes
    s = strokes[strokes["principal"].astype(bool)] if "principal" in strokes and len(strokes) else strokes
    out = {
        "active_ratio": _r(act.mean()),
        "tempo_bpm": _r(np.median(lt), 1) if len(lt) else None,
        "tempo_candidates_bpm": ", ".join(map(str, info.get("tempo_candidates_bpm", []))),
        "tempo_stability_iqr_rel": _r((np.percentile(lt, 75) - np.percentile(lt, 25)) / np.median(lt)) if len(lt) else None,
        "stroke_rate_per_s": _r(len(allst) / at, 2) if at > 0 else None,
        "principal_stroke_rate_per_s": _r(len(s) / at, 2) if at > 0 else None,
        "stroke_attack_ms_median": _r(s["attack_ms"].median(), 2) if len(s) else None,
        "stroke_centroid_hz_median": _r(s["centroid_hz"].median(), 0) if len(s) else None,
        "stroke_crest_db_median": _r(s["crest_db"].median(), 2) if len(s) else None,
        "stroke_dynamic_range_db_p10_p90": _r(allst["peak_dbfs"].quantile(0.9) - allst["peak_dbfs"].quantile(0.1), 2) if len(allst) else None,
        "stroke_low_band_share": _r(s["low_ratio"].median()) if len(s) else None,
        "stroke_high_band_share": _r(s["high_ratio"].median()) if len(s) else None,
        "perc_to_vocal_db": _r(np.median(pf["perc_dbfs"].to_numpy()[both] - vocal_dbfs[:len(act)][both]), 1) if both.any() else None,
        "perc_to_accomp_db": _r(np.median((pf["perc_dbfs"] - pf["accomp_dbfs"]).to_numpy()[act]), 1) if act.any() else None,
    }
    return out
