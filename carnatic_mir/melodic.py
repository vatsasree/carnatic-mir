"""Melodic / ornamentation analysis of the vocal pitch contour.

All events are detected on the tonic-normalised, 50 ms median-smoothed contour (cents re: Sa).
Building blocks:
  * turning points  - zig-zag extrema: a reversal of >= `turn_delta_cents` confirms a peak/trough
  * oscillation episodes - >= `osc_min_extrema` consecutive turning points spaced 40-350 ms apart
        -> 'simple' (extent < kampita_min_extent_cents) or 'heavy_kampita'
  * stable notes - <= stable_range_cents excursion over 150 ms, >= 200 ms, outside oscillations
  * glides  - monotonic movement >= glide_min_extent_cents between turning points / stable notes,
              10-90 % rise time >= glide_min_dur_s, outside oscillations
  * sphurita - an isolated dip from below: steady -> brief dip (60-350 c, <= 150 ms) -> SAME svara
  * elongations - stable notes (plain) or oscillation episodes centred on one svara (ornamented)
              lasting >= elongation_min_s; also reported in beats when tempo is known
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from .config import HOP_S, SVARASTHANAS

SVARA_NAMES = [n for _, n in SVARASTHANAS]

# frame event codes (timeseries column `event`)
EV_NONE, EV_STABLE, EV_SIMPLE_OSC, EV_KAMPITA, EV_GLIDE = 0, 1, 2, 3, 4

PARAMS = dict(
    smooth_frames=5,
    stable_window_s=0.15, stable_range_cents=35.0, stable_min_dur_s=0.20,
    turn_delta_cents=30.0,
    osc_min_half_period_s=0.04, osc_max_half_period_s=0.35, osc_min_extrema=4, osc_max_swing_cents=600.0,
    kampita_min_extent_cents=150.0,
    glide_min_extent_cents=150.0, glide_min_dur_s=0.08,
    sphurita_min_depth_cents=60.0, sphurita_max_depth_cents=350.0, sphurita_max_width_s=0.15,
    sphurita_ctx_s=0.10, sphurita_ctx_range_cents=60.0, sphurita_return_tol_cents=50.0,
    elongation_min_s=0.6, elongation_merge_gap_s=0.08,
    phrase_gap_s=0.25, phrase_min_dur_s=0.30,
)


# ----------------------------------------------------------------------------- helpers
def runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def svara_label(cents: float) -> str:
    """Nearest svarasthana with octave mark: '.P' = mandra, 'P' = madhya, "P'" = tara."""
    if cents is None or not np.isfinite(cents):
        return ""
    k = int(np.round(cents / 100.0))
    octave, idx = divmod(k, 12)
    name = SVARA_NAMES[idx]
    return ("." * -octave + name) if octave < 0 else (name + "'" * octave)


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def turning_points(x: np.ndarray, delta: float):
    """Zig-zag turning points of a finite 1-D array: list of (index, +1 peak / -1 trough)."""
    n = len(x)
    out = []
    if n < 3:
        return out
    hi = lo = x[0]
    hi_i = lo_i = 0
    d = 0
    for i in range(1, n):
        v = x[i]
        if d == 0:
            if v > hi:
                hi, hi_i = v, i
            if v < lo:
                lo, lo_i = v, i
            if hi - lo >= delta:
                d = 1 if hi_i > lo_i else -1
        elif d == 1:
            if v > hi:
                hi, hi_i = v, i
            elif hi - v >= delta:
                out.append((hi_i, 1))
                d, lo, lo_i = -1, v, i
        else:
            if v < lo:
                lo, lo_i = v, i
            elif v - lo >= delta:
                out.append((lo_i, -1))
                d, hi, hi_i = 1, v, i
    return out


# ----------------------------------------------------------------------------- main analysis
def analyze(frames: pd.DataFrame, tonic_hz: float, tempo_bpm: float | None = None,
            params: dict | None = None) -> dict:
    p = {**PARAMS, **(params or {})}
    n = len(frames)
    voiced = frames["voiced"].to_numpy(bool).copy()
    f0 = frames["f0_hz"].to_numpy(float).copy()

    cents = np.full(n, np.nan)
    cents[voiced] = 1200 * np.log2(f0[voiced] / tonic_hz)
    sm = pd.Series(cents).rolling(p["smooth_frames"], center=True, min_periods=1).median().to_numpy().copy()
    sm[~voiced] = np.nan
    vel = np.full(n, np.nan)
    for a, b in runs(voiced):
        if b - a >= 3:
            vel[a:b] = np.gradient(gaussian_filter1d(sm[a:b], 1.0)) / HOP_S

    vruns = [(a, b) for a, b in runs(voiced) if b - a >= 3]
    tps = []                                    # global (index, sign)
    for a, b in vruns:
        tps += [(a + i, s) for i, s in turning_points(sm[a:b], p["turn_delta_cents"])]
    tp_run = np.zeros(len(tps), int)            # which voiced run each tp belongs to
    run_id = np.full(n, -1)
    for k, (a, b) in enumerate(vruns):
        run_id[a:b] = k
    for j, (i, _) in enumerate(tps):
        tp_run[j] = run_id[i]

    # --- oscillation episodes ---------------------------------------------------------
    hp_lo, hp_hi = p["osc_min_half_period_s"], p["osc_max_half_period_s"]
    groups, cur = [], [0] if tps else []
    for j in range(1, len(tps)):
        dt = (tps[j][0] - tps[j - 1][0]) * HOP_S
        sw = abs(sm[tps[j][0]] - sm[tps[j - 1][0]])
        if tp_run[j] == tp_run[j - 1] and hp_lo <= dt <= hp_hi and sw <= p["osc_max_swing_cents"]:
            cur.append(j)
        else:
            groups.append(cur)
            cur = [j]
    if cur:
        groups.append(cur)

    osc_mask = np.zeros(n, bool)
    osc_id = np.full(n, -1)
    oscillations, osc_extrema = [], []
    for g in groups:
        if len(g) < p["osc_min_extrema"]:
            continue
        idx = np.array([tps[j][0] for j in g])
        vals = sm[idx]
        swings = np.abs(np.diff(vals))
        span = (idx[-1] - idx[0]) * HOP_S                 # extremum-to-extremum (used for rate)
        ext_k = int(round(np.median(np.diff(idx)) / 2))   # the episode extends ~half a half-period past
        ra, rb = vruns[run_id[idx[0]]]                    # its outer extrema (clipped to the voiced run)
        a, b = max(ra, idx[0] - ext_k), min(rb, idx[-1] + 1 + ext_k)
        dur = (b - a) * HOP_S
        osc_extrema.append((idx[0], idx[-1]))
        mids = (vals[:-1] + vals[1:]) / 2
        peaks = vals[[tps[j][1] == 1 for j in g]]
        troughs = vals[[tps[j][1] == -1 for j in g]]
        extent = float(np.median(swings))
        half = len(mids) // 2
        drift = abs(np.median(mids[half:]) - np.median(mids[:half])) if half else 0.0
        hps = np.diff(idx) * HOP_S
        e = {
            "start": _r(a * HOP_S), "end": _r(b * HOP_S), "duration": _r(dur),
            "type": "heavy_kampita" if extent >= p["kampita_min_extent_cents"] else "simple",
            "extent_cents": _r(extent, 1), "max_extent_cents": _r(swings.max(), 1),
            "rate_hz": _r((len(idx) - 1) / 2 / span if span > 0 else np.nan, 2),
            "n_cycles": _r((len(idx) - 1) / 2, 1),
            "regularity_cv": _r(np.std(hps) / np.mean(hps), 3),
            "centre_cents": _r(np.median(mids), 1), "centre_svara": svara_label(np.median(mids)),
            "upper_svara": svara_label(np.median(peaks)) if len(peaks) else "",
            "lower_svara": svara_label(np.median(troughs)) if len(troughs) else "",
            "centre_drift_cents": _r(drift, 1),
        }
        osc_id[a:b] = len(oscillations)
        osc_mask[a:b] = True
        oscillations.append(e)

    # --- stable notes ---------------------------------------------------------------------
    w = max(3, int(round(p["stable_window_s"] / HOP_S)))
    s = pd.Series(sm)
    rng = (s.rolling(w, center=True, min_periods=w).max() - s.rolling(w, center=True, min_periods=w).min()).to_numpy()
    centre_ok = np.nan_to_num(rng, nan=1e9) <= p["stable_range_cents"]
    stable_raw = (np.convolve(centre_ok, np.ones(w), mode="same") > 0) & voiced & ~osc_mask
    min_len = int(round(p["stable_min_dur_s"] / HOP_S))
    stable_mask = np.zeros(n, bool)
    stable = []
    for a, b in runs(stable_raw):
        if b - a >= min_len:
            stable_mask[a:b] = True
            med = float(np.nanmedian(sm[a:b]))
            stable.append({"start": _r(a * HOP_S), "end": _r(b * HOP_S), "duration": _r((b - a) * HOP_S),
                           "cents": _r(med, 1), "svara": svara_label(med),
                           "deviation_cents": _r(med - 100 * np.round(med / 100), 1)})

    # --- sphurita ----------------------------------------------------------------------------
    sph = []
    G = int(round(p["sphurita_max_width_s"] / HOP_S))
    W = int(round(p["sphurita_ctx_s"] / HOP_S))
    for i, sign in tps:
        if sign != -1 or osc_mask[i] or i - G - W < 0 or i + G + W >= n:
            continue
        pre, post = sm[i - G - W:i - G], sm[i + G:i + G + W]
        if np.isfinite(pre).mean() < 0.8 or np.isfinite(post).mean() < 0.8:
            continue
        if np.nanmax(pre) - np.nanmin(pre) > p["sphurita_ctx_range_cents"] or \
           np.nanmax(post) - np.nanmin(post) > p["sphurita_ctx_range_cents"]:
            continue
        lp, lq = np.nanmedian(pre), np.nanmedian(post)
        if abs(lp - lq) > p["sphurita_return_tol_cents"]:
            continue
        level = (lp + lq) / 2
        depth = level - sm[i]
        if not (p["sphurita_min_depth_cents"] <= depth <= p["sphurita_max_depth_cents"]):
            continue
        thr = sm[i] + depth / 2
        l = i
        while l > i - G - W and np.isfinite(sm[l - 1]) and sm[l - 1] < thr:
            l -= 1
        r = i
        while r < i + G + W and np.isfinite(sm[r + 1]) and sm[r + 1] < thr:
            r += 1
        width = (r - l + 1) * HOP_S
        if width > p["sphurita_max_width_s"]:
            continue
        sph.append({"time": _r(i * HOP_S), "start": _r(l * HOP_S), "end": _r((r + 1) * HOP_S),
                    "svara": svara_label(level), "touched_svara": svara_label(sm[i]),
                    "depth_cents": _r(depth, 1), "width_s": _r(width), "trough_cents": _r(sm[i], 1)})

    sph_mask = np.zeros(n, bool)
    for x in sph:
        sph_mask[max(0, int(round(x["start"] / HOP_S)) - 2):int(round(x["end"] / HOP_S)) + 2] = True
    osc_first = {i0 for i0, _ in osc_extrema}
    osc_last = {i1 for _, i1 in osc_extrema}
    max_hp = int(round(p["osc_max_half_period_s"] / HOP_S))

    # --- glides ------------------------------------------------------------------------------
    glide_mask = np.zeros(n, bool)
    glides, n_transitions = [], 0
    stab_edges = set()
    for a, b in runs(stable_mask):
        stab_edges.update((a, b - 1))
    for k, (a, b) in enumerate(vruns):
        bnd = sorted({a, b - 1} | {i for i, _ in tps if a <= i < b} | {i for i in stab_edges if a <= i < b})
        for i0, i1 in zip(bnd[:-1], bnd[1:]):
            mid = (i0 + i1) // 2
            if osc_mask[mid] or sph_mask[mid] or (stable_mask[mid] and stable_mask[i0] and stable_mask[i1]):
                continue
            if (i0 in osc_last or i1 in osc_first) and i1 - i0 <= max_hp:   # exit/approach half-swing of an oscillation
                continue
            delta = sm[i1] - sm[i0]
            if abs(delta) < p["glide_min_extent_cents"]:
                continue
            n_transitions += 1
            prog = (sm[i0:i1 + 1] - sm[i0]) / delta
            t10 = int(np.argmax(prog >= 0.1))
            t90 = int(np.argmax(prog >= 0.9))
            dur = (t90 - t10) * HOP_S / 0.8
            if dur < p["glide_min_dur_s"]:
                continue
            ga, gb = i0 + t10, i0 + t90 + 1
            glide_mask[ga:gb] = True
            glides.append({
                "start": _r(ga * HOP_S), "end": _r(gb * HOP_S), "duration": _r(dur),
                "direction": "up" if delta > 0 else "down",
                "extent_cents": _r(abs(delta), 1), "slope_cents_s": _r(abs(delta) / dur, 1),
                "from_cents": _r(sm[i0], 1), "to_cents": _r(sm[i1], 1),
                "from_svara": svara_label(sm[i0]), "to_svara": svara_label(sm[i1]),
            })

    # --- elongations -------------------------------------------------------------------------
    merged = []
    for x in stable:
        if merged and x["start"] - merged[-1]["end"] <= p["elongation_merge_gap_s"] \
                and abs(x["cents"] - merged[-1]["cents"]) <= 30:
            m = merged[-1]
            m["cents"] = (m["cents"] * m["duration"] + x["cents"] * x["duration"]) / (m["duration"] + x["duration"])
            m["end"], m["duration"] = x["end"], round(x["end"] - m["start"], 3)
        else:
            merged.append(dict(x))
    beat = 60.0 / tempo_bpm if tempo_bpm else None
    elong = []
    for x in merged:
        if x["duration"] >= p["elongation_min_s"]:
            elong.append({"start": x["start"], "end": x["end"], "duration": x["duration"], "kind": "plain",
                          "svara": svara_label(x["cents"]), "cents": _r(x["cents"], 1)})
    for o in oscillations:
        if o["duration"] >= p["elongation_min_s"] and o["centre_drift_cents"] <= 60:
            elong.append({"start": o["start"], "end": o["end"], "duration": o["duration"],
                          "kind": f"ornamented ({o['type']})", "svara": o["centre_svara"], "cents": o["centre_cents"]})
    elong.sort(key=lambda e: e["start"])
    for e in elong:
        e["beats"] = _r(e["duration"] / beat, 2) if beat else None
    elong_mask = np.zeros(n, bool)
    for e in elong:
        elong_mask[int(round(e["start"] / HOP_S)):int(round(e["end"] / HOP_S))] = True

    # --- phrases ------------------------------------------------------------------------------
    gap = int(round(p["phrase_gap_s"] / HOP_S))
    filled = voiced.copy()
    for a, b in runs(~voiced):
        if b - a < gap and a > 0 and b < n:
            filled[a:b] = True
    phrases = []
    for a, b in runs(filled):
        seg = sm[a:b]
        fin = seg[np.isfinite(seg)]
        if (b - a) * HOP_S < p["phrase_min_dur_s"] or len(fin) == 0:
            continue
        phrases.append({"start": _r(a * HOP_S), "end": _r(b * HOP_S), "duration": _r((b - a) * HOP_S),
                        "min_cents": _r(fin.min(), 1), "max_cents": _r(fin.max(), 1),
                        "start_svara": svara_label(fin[0]), "end_svara": svara_label(fin[-1]),
                        "stable_ratio": _r(stable_mask[a:b].sum() / max(1, voiced[a:b].sum()))})

    # --- frame event labels -------------------------------------------------------------------
    event = np.zeros(n, np.int8)
    event[stable_mask] = EV_STABLE
    event[glide_mask] = EV_GLIDE
    for k, o in enumerate(oscillations):
        event[osc_id == k] = EV_KAMPITA if o["type"] == "heavy_kampita" else EV_SIMPLE_OSC

    # --- summaries ---------------------------------------------------------------------------
    vc = sm[np.isfinite(sm)]
    vt = voiced.sum() * HOP_S
    vmin = vt / 60 if vt > 0 else np.nan
    pct = (lambda q: float(np.percentile(vc, q))) if len(vc) else (lambda q: np.nan)

    def stats(xs, key):
        v = np.array([x[key] for x in xs], float)
        return (_r(np.median(v), 2), _r(np.mean(v), 2)) if len(v) else (None, None)

    simple = [o for o in oscillations if o["type"] == "simple"]
    heavy = [o for o in oscillations if o["type"] == "heavy_kampita"]
    up = [g for g in glides if g["direction"] == "up"]

    summary = {
        "range": {
            "lowest_svara_p1": svara_label(pct(1)), "highest_svara_p99": svara_label(pct(99)),
            "p1_cents": _r(pct(1), 1), "p99_cents": _r(pct(99), 1),
            "p5_cents": _r(pct(5), 1), "p95_cents": _r(pct(95), 1),
            "range_p1_p99_octaves": _r((pct(99) - pct(1)) / 1200),
            "range_p5_p95_octaves": _r((pct(95) - pct(5)) / 1200),
            "lowest_hz_p1": _r(tonic_hz * 2 ** (pct(1) / 1200), 1), "highest_hz_p99": _r(tonic_hz * 2 ** (pct(99) / 1200), 1),
            "share_mandra": _r((vc < 0).mean()) if len(vc) else None,
            "share_madhya": _r(((vc >= 0) & (vc < 1200)).mean()) if len(vc) else None,
            "share_tara": _r((vc >= 1200).mean()) if len(vc) else None,
        },
        "elongation": {
            "count": len(elong), "per_voiced_min": _r(len(elong) / vmin, 2),
            "plain_count": sum(e["kind"] == "plain" for e in elong),
            "ornamented_count": sum(e["kind"] != "plain" for e in elong),
            "mean_duration_s": stats(elong, "duration")[1], "max_duration_s": _r(max([e["duration"] for e in elong], default=np.nan), 2),
            "mean_duration_beats": _r(np.mean([e["beats"] for e in elong]), 2) if beat and elong else None,
            "share_of_voiced_time": _r(elong_mask.sum() * HOP_S / vt) if vt else None,
            "top_svaras": _top([e["svara"] for e in elong]),
        },
        "glides": {
            "count": len(glides), "per_voiced_min": _r(len(glides) / vmin, 2),
            "share_up": _r(len(up) / len(glides)) if glides else None,
            "median_extent_cents": stats(glides, "extent_cents")[0],
            "median_duration_s": stats(glides, "duration")[0],
            "median_slope_cents_s": stats(glides, "slope_cents_s")[0],
            "glide_share_of_transitions": _r(len(glides) / n_transitions) if n_transitions else None,
            "share_of_voiced_time": _r(glide_mask.sum() * HOP_S / vt) if vt else None,
        },
        "sphurita": {
            "count": len(sph), "per_voiced_min": _r(len(sph) / vmin, 2),
            "median_depth_cents": stats(sph, "depth_cents")[0], "median_width_s": stats(sph, "width_s")[0],
            "top_svaras": _top([x["svara"] for x in sph]),
        },
        "oscillation": {
            "simple_count": len(simple), "simple_per_voiced_min": _r(len(simple) / vmin, 2),
            "simple_median_extent_cents": stats(simple, "extent_cents")[0], "simple_median_rate_hz": stats(simple, "rate_hz")[0],
            "simple_share_of_voiced_time": _r(sum(o["duration"] for o in simple) / vt) if vt else None,
            "kampita_count": len(heavy), "kampita_per_voiced_min": _r(len(heavy) / vmin, 2),
            "kampita_median_extent_cents": stats(heavy, "extent_cents")[0], "kampita_median_rate_hz": stats(heavy, "rate_hz")[0],
            "kampita_share_of_voiced_time": _r(sum(o["duration"] for o in heavy) / vt) if vt else None,
            "kampita_top_svaras": _top([o["centre_svara"] for o in heavy]),
        },
        "melody": {
            "voiced_ratio": _r(voiced.mean()), "stable_ratio": _r(stable_mask.sum() / max(1, voiced.sum())),
            "n_stable_notes": len(stable), "n_phrases": len(phrases),
            "mean_phrase_s": _r(np.mean([x["duration"] for x in phrases]), 2) if phrases else None,
            "turning_points_per_voiced_s": _r(len(tps) / vt, 2) if vt else None,
            "mean_abs_pitch_velocity_cents_s": _r(np.nanmean(np.abs(vel)), 1) if len(vc) else None,
        },
        "intonation": {},
    }

    # --- intonation (on stable frames) -------------------------------------------------------------
    sv = np.mod(sm[stable_mask], 1200)
    per_svara = {}
    for c0, name in SVARASTHANAS:
        d = (sv - c0 + 600) % 1200 - 600          # signed distance to svara centre (circular)
        x = d[np.abs(d) <= 50]
        if len(x) < 50:
            continue
        h, e = np.histogram(x, bins=50, range=(-50, 50))
        h = gaussian_filter1d(h.astype(float), 1.5)
        per_svara[name] = {"share": _r(len(x) / len(sv)), "peak_dev_cents": _r((e[np.argmax(h)] + 1), 1),
                           "mean_dev_cents": _r(x.mean(), 1), "std_cents": _r(x.std(), 1),
                           "skew": _r(skew(x), 2), "kurtosis": _r(kurtosis(x), 2)}
    if per_svara:
        wts = np.array([v["share"] for v in per_svara.values()])
        summary["intonation"] = {
            "mean_abs_dev_from_et_cents": _r(np.average([abs(v["mean_dev_cents"]) for v in per_svara.values()], weights=wts), 1),
            "mean_spread_cents": _r(np.average([v["std_cents"] for v in per_svara.values()], weights=wts), 1),
        }

    # --- histograms -----------------------------------------------------------------------------
    fold, fe = np.histogram(np.mod(vc, 1200), bins=240, range=(0, 1200))
    fold = gaussian_filter1d(fold.astype(float), 1.5, mode="wrap")
    fold /= max(fold.sum(), 1)
    ext = np.concatenate([fold[-20:], fold, fold[:20]])
    pk, _ = find_peaks(ext, prominence=fold.max() * 0.03, distance=10)
    pk = pk[(pk >= 20) & (pk < 260)] - 20
    hist_peaks = [{"cents": float(c), "svara": svara_label(c), "weight": _r(fold[i], 5),
                   "deviation_from_et": _r(c - 100 * np.round(c / 100), 1)} for i, c in zip(pk, pk * 5.0 + 2.5)]

    def class_dist(values, weights=None):
        if len(values) == 0:
            return {}
        idx = np.mod(np.round(np.asarray(values) / 100).astype(int), 12)
        dd = np.bincount(idx, weights=weights, minlength=12)
        return {SVARA_NAMES[i]: _r(dd[i] / max(dd.sum(), 1e-9), 4) for i in range(12)}

    ts = pd.DataFrame({"cents": sm, "pitch_velocity_cents_s": vel, "event": event,
                       "stable": stable_mask, "elongation": elong_mask, "svara": [svara_label(c) for c in sm]})
    return {
        "timeseries": ts, "summary": summary, "params": p,
        "events": {"stable_notes": stable, "elongations": elong, "glides": glides, "sphurita": sph,
                   "oscillations": oscillations, "phrases": phrases},
        "intonation_per_svara": per_svara,
        "histogram_folded": {"bin_centers_cents": ((fe[:-1] + fe[1:]) / 2).tolist(),
                             "density": np.round(fold, 6).tolist(), "peaks": hist_peaks},
        "svara_distribution_all_frames": class_dist(vc),
        "svara_distribution_stable_notes": class_dist([x["cents"] for x in stable], [x["duration"] for x in stable]),
    }


def _top(labels, k=3):
    if not labels:
        return ""
    v, c = np.unique(labels, return_counts=True)
    o = np.argsort(-c)[:k]
    return ", ".join(f"{v[i]}({c[i]})" for i in o)
