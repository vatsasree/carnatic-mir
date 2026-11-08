"""H3: do lullabies differ acoustically from their matched non-lullabies?

Measured twice: on the snippets participants heard (from the 10 ms time series) and on the whole songs
(pre-registered summary features from features.json). With 10 pairs the test is the exact paired sign-flip
permutation test over pairs (2^10 = 1024 flips) plus the paired effect size dz and a bootstrap 95 % CI.
Tempo is reported but flagged: estimates can be off by a factor of 2 until checked by ear.
"""
import itertools
import json

import numpy as np
import pandas as pd

from carnatic_mir.config import OUTPUTS
from .common import rng, snippet_frames

# feature -> predicted sign of (lullaby - non-lullaby)
SNIPPET_FEATURES = {
    "held_note_share": +1, "plain_steady_share": +1, "oscillation_share": -1, "kampita_share": -1,
    "glide_share": None, "range_p5_p95_cents": -1, "tara_share": -1, "pitch_speed_cents_s": -1,
    "percussion_active_share": -1, "perc_minus_voice_db": -1, "sung_share": None}
SONG_FEATURES = {
    "elongation.share_of_voiced_time": +1, "range.range_p5_p95_octaves": -1, "range.share_tara": -1,
    "voice.phrase_attack_time_ms": +1, "voice.harmonic_tilt_db_per_oct": -1, "percussion.stroke_rate_per_s": -1,
    "percussion.perc_to_vocal_db": -1, "melody.turning_points_per_voiced_s": -1, "tempo.tempo_bpm": -1}


def snippet_profile(track_id: str, start: float, end: float) -> dict:
    ts = snippet_frames(track_id, start, end)
    v = ts["voiced"].to_numpy(); n = max(v.sum(), 1); ev = ts["event"].to_numpy()[v]
    c = ts["cents"].to_numpy()[v]; c = c[np.isfinite(c)]
    voice = ts["rms_dbfs"].to_numpy(); perc = ts["perc_dbfs"].to_numpy()
    sung = v & (voice > -60)
    return {"held_note_share": ts["elongation"].to_numpy()[v].sum() / n,
            "plain_steady_share": (ev == 1).sum() / n, "oscillation_share": np.isin(ev, (2, 3)).sum() / n,
            "kampita_share": (ev == 3).sum() / n, "glide_share": (ev == 4).sum() / n,
            "range_p5_p95_cents": float(np.subtract(*np.percentile(c, [95, 5]))) if c.size > 10 else np.nan,
            "tara_share": float((c >= 1200).mean()) if c.size else np.nan,
            "pitch_speed_cents_s": float(np.nanmedian(np.abs(ts["pitch_velocity_cents_s"].to_numpy()[v]))),
            "percussion_active_share": float(ts["perc_active"].mean()),
            "perc_minus_voice_db": float(np.median(perc[sung] - voice[sung])) if sung.any() else np.nan,
            "sung_share": float(v.mean())}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def song_profile(track_id: str) -> dict:
    s = _flat(json.loads((OUTPUTS / track_id / "features.json").read_text())["summary"])
    return {k: (float(s[k]) if isinstance(s.get(k), (int, float)) else np.nan) for k in SONG_FEATURES}


def paired_test(diff: np.ndarray, n_boot: int = 5000) -> dict:
    d = diff[np.isfinite(diff)]; n = len(d)
    if n < 2:
        return dict(n_pairs=n)
    flips = np.array(list(itertools.product([1, -1], repeat=n))) if n <= 14 else rng.choice([1, -1], (20000, n))
    null = (flips * d).mean(1)
    p = (np.abs(null) >= abs(d.mean()) - 1e-12).mean()
    boots = rng.choice(d, (n_boot, n)).mean(1)
    return dict(n_pairs=n, mean_diff=d.mean(), ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan, p_perm=p,
                pairs_lullaby_higher=int((d > 0).sum()))


def compare(table: pd.DataFrame, features: dict, level: str) -> pd.DataFrame:
    rows = []
    w = table.groupby(["pair", "condition"])[list(features)].mean().unstack("condition")   # versions averaged per song
    for f, sign in features.items():
        L, N = w[(f, "lullaby")].to_numpy(float), w[(f, "non_lullaby")].to_numpy(float)
        r = dict(level=level, feature=f, predicted=("higher" if sign == 1 else "lower" if sign == -1 else "none"),
                 lullaby_mean=np.nanmean(L), non_lullaby_mean=np.nanmean(N), **paired_test(L - N))
        if sign and "mean_diff" in r:
            r["direction_as_predicted"] = bool(np.sign(r["mean_diff"]) == sign)
        if f == "tempo.tempo_bpm":
            r["note"] = "unverified: may be off by x2"
        rows.append(r)
    return pd.DataFrame(rows)


def run(snippets: pd.DataFrame, snippet_means: pd.DataFrame | None = None) -> dict:
    sn = snippets[snippets["pair"] != "PRACTICE"]
    snip = pd.DataFrame([{"pair": r.pair, "condition": r.condition, "version": getattr(r, "version", 1),
                          "track_id": r.track_id, "token": r.token,
                          **snippet_profile(r.track_id, float(r.start_s), float(r.end_s))} for r in sn.itertuples()])
    song = pd.DataFrame([{"pair": r.pair, "condition": r.condition, "track_id": r.track_id,
                          **song_profile(r.track_id)} for r in sn.drop_duplicates("track_id").itertuples()])
    res = {"snippet_features": snip, "song_features": song,
           "h3": pd.concat([compare(snip, SNIPPET_FEATURES, "snippet"), compare(song, SONG_FEATURES, "whole song")])}
    if snippet_means is not None and len(snippet_means):     # exploratory link H1 <-> H3 (20 snippets)
        m = snip.merge(snippet_means[["token", "soothing", "arousal"]], on="token")
        rows = []
        for f in SNIPPET_FEATURES:
            for o in ("soothing", "arousal"):
                x = m[[f, o]].dropna()
                if len(x) > 4:
                    rho = x[f].rank().corr(x[o].rank())
                    rows.append(dict(feature=f, outcome=o, spearman_rho=rho, n_snippets=len(x)))
        res["feature_rating_link"] = pd.DataFrame(rows)
    return res
