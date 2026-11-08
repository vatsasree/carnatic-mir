"""H2: which musical moments are followed by "calm or soothing" presses?

1. Press curves: for each snippet, the share of listeners holding the button every STEP seconds
   (hold times shifted by each trial's output latency, so they refer to when the sound reached the ear).
2. Event-locked change: for each feature event, mean press share in the POST s after the event minus the PRE s
   before it, averaged over all events of that type; compared with the same measure at random times in the same
   snippets (permutation, N_PERM draws).
3. Lagged regression (exploratory): press share ~ frame features at lag L, snippet fixed effects, cluster-robust
   SE by snippet; the lag with the best fit estimates the response delay.
"""
import warnings

import numpy as np
import pandas as pd

from .common import rng, snippet_frames

STEP, PRE, POST, N_PERM = 0.1, 1.0, 3.0, 2000
HOP = 0.01                              # timeseries frame step
EVENT_TYPES = ["elongation_plain", "elongation_ornamented", "glide", "simple_oscillation", "heavy_kampita",
               "phrase_onset", "percussion_entry"]


def _runs(mask: np.ndarray, min_len: int = 1):
    """(start, end) index pairs of True runs at least min_len long."""
    e = np.diff(np.r_[0, mask.astype(int), 0])
    on, off = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    return [(a, b) for a, b in zip(on, off) if b - a >= min_len]


def events(ts: pd.DataFrame) -> pd.DataFrame:
    """Feature events in a snippet (time in snippet seconds)."""
    t = ts["t"].to_numpy(); ev = ts["event"].to_numpy(); out = []
    for a, b in _runs(ts["elongation"].to_numpy(), int(0.3 / HOP)):
        plain = (ev[a:b] == 1).mean() >= 0.6
        out.append(("elongation_plain" if plain else "elongation_ornamented", t[a], (b - a) * HOP))
    for code, name, mn in ((4, "glide", 0.05), (2, "simple_oscillation", 0.2), (3, "heavy_kampita", 0.2)):
        out += [(name, t[a], (b - a) * HOP) for a, b in _runs(ev == code, int(mn / HOP))]
    v = ts["voiced"].to_numpy()
    last_off = -1e9
    for a, b in _runs(v, int(0.1 / HOP)):                     # phrase = sung run after a pause of >= 0.3 s
        if (a - last_off) * HOP >= 0.3:
            out.append(("phrase_onset", t[a], None))
        last_off = b
    p = ts["perc_active"].to_numpy()
    for a, b in _runs(p, int(0.5 / HOP)):                     # percussion returns after >= 1 s without it
        if a >= int(1 / HOP) and not p[a - int(1 / HOP):a].any():
            out.append(("percussion_entry", t[a], (b - a) * HOP))
    return pd.DataFrame(out, columns=["event", "t", "duration_s"])


def curves(phase2: pd.DataFrame, presses: pd.DataFrame, snippets: pd.DataFrame) -> dict:
    """token -> DataFrame(t, share, n_listeners) on a STEP grid."""
    out = {}
    for tok, trials in phase2.groupby("token"):
        dur = float(trials["duration_s"].median())
        grid = np.arange(0, dur, STEP)
        held = np.zeros((len(trials), len(grid)), bool)
        h = presses[presses["token"] == tok]
        for k, (_, tr) in enumerate(trials.iterrows()):
            lat = tr["output_latency_s"] if pd.notna(tr["output_latency_s"]) else 0.0
            for _, r in h[h["pid"] == tr["pid"]].iterrows():
                held[k] |= (grid >= r["start_s"] - lat) & (grid < r["end_s"] - lat)
        out[tok] = pd.DataFrame({"t": grid, "share": held.mean(0), "n_listeners": len(trials)})
    return out


def _delta(c: np.ndarray, times: np.ndarray) -> np.ndarray:
    i0 = np.round(times / STEP).astype(int)
    pre, post = int(PRE / STEP), int(POST / STEP)
    cs = np.r_[0, np.cumsum(c)]
    return (cs[i0 + post] - cs[i0]) / post - (cs[i0] - cs[i0 - pre]) / pre


def event_locked(cur: dict, ev: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed vs permutation null per event type; plus mean time courses for plotting."""
    rows, courses = [], []
    for name in EVENT_TYPES:
        obs, null_parts = [], []
        per_tok = []
        for tok, c in cur.items():
            if tok not in ev:
                continue
            dur = c["t"].iloc[-1]
            e = ev[tok]; e = e[(e["event"] == name) & (e["t"] >= PRE) & (e["t"] <= dur - POST)]["t"].to_numpy()
            if len(e):
                per_tok.append((c["share"].to_numpy(), e, dur))
        if not per_tok:
            rows.append(dict(event=name, n_events=0)); continue
        obs = np.concatenate([_delta(s, e) for s, e, _ in per_tok]).mean()
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            null[k] = np.concatenate([_delta(s, rng.uniform(PRE, d - POST, len(e))) for s, e, d in per_tok]).mean()
        p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (N_PERM + 1)
        rows.append(dict(event=name, n_events=sum(len(e) for _, e, _ in per_tok), n_snippets=len(per_tok),
                         delta_share=obs, null_mean=null.mean(), null_sd=null.std(),
                         z=(obs - null.mean()) / null.std() if null.std() > 0 else np.nan, p_perm=p))
        lags = np.arange(-2, 5 + 1e-9, STEP)               # mean time course around the events
        acc = []
        for s, e, d in per_tok:
            for x in e:
                idx = np.round((x + lags) / STEP).astype(int)
                ok = (idx >= 0) & (idx < len(s))
                acc.append(np.where(ok, s[np.clip(idx, 0, len(s) - 1)], np.nan))
        courses.append(pd.DataFrame({"event": name, "lag_s": lags, "share": np.nanmean(acc, 0)}))
    return pd.DataFrame(rows), pd.concat(courses) if courses else pd.DataFrame()


def frame_features(ts: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Frame features averaged onto the STEP grid."""
    ev = ts["event"].to_numpy()
    f = pd.DataFrame({
        "held_note": ts["elongation"].astype(float), "steady": (ev == 1).astype(float),
        "oscillation": np.isin(ev, (2, 3)).astype(float), "kampita": (ev == 3).astype(float),
        "glide": (ev == 4).astype(float), "voiced": ts["voiced"].astype(float),
        "pitch_speed": np.nan_to_num(np.abs(ts["pitch_velocity_cents_s"].to_numpy())) / 1000,
        "percussion": ts["perc_active"].astype(float),
        "voice_level": (np.clip(ts["rms_dbfs"].to_numpy(), -60, 0) + 60) / 60})
    b = np.clip(np.floor(ts["t"].to_numpy() / STEP).astype(int), 0, len(grid) - 1)
    return f.groupby(b).mean().reindex(range(len(grid))).ffill().bfill()


def lagged_regression(cur: dict, feats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf
    names = ["held_note", "oscillation", "glide", "voiced", "pitch_speed", "percussion", "voice_level"]
    fits, best = [], None
    for lag in np.arange(0, 3.01, 0.25):
        k = int(round(lag / STEP)); parts = []
        for tok, c in cur.items():
            if tok not in feats:
                continue
            X = feats[tok][names].shift(k)               # feature at t - lag explains the press share at t
            parts.append(pd.concat([c[["share"]], X], axis=1).assign(token=tok).dropna())
        if not parts:
            break
        d = pd.concat(parts, ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.ols("share ~ " + " + ".join(names) + " + C(token)", d).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d["token"])[0]})
        fits.append(dict(lag_s=lag, r2=m.rsquared, n_bins=len(d)))
        if best is None or m.rsquared > best[0]:
            best = (m.rsquared, lag, m)
    if best is None:
        return pd.DataFrame(), pd.DataFrame()
    _, lag, m = best
    coef = pd.DataFrame({"feature": names, "lag_s": lag, "coef": [m.params[n] for n in names],
                         "se": [m.bse[n] for n in names], "p": [m.pvalues[n] for n in names]})
    return pd.DataFrame(fits), coef


def run(phase2: pd.DataFrame, presses: pd.DataFrame, snippets: pd.DataFrame) -> dict:
    if phase2.empty:
        return {}
    cur = curves(phase2, presses, snippets)
    sn = snippets.set_index("token")
    ev, feats, evrows = {}, {}, []
    for tok, c in cur.items():
        r = sn.loc[tok]
        ts = snippet_frames(r["track_id"], float(r["start_s"]), float(r["end_s"]))
        ev[tok] = events(ts)
        feats[tok] = frame_features(ts, c["t"].to_numpy())
        evrows.append(ev[tok].assign(token=tok, pair=r["pair"], condition=r["condition"]))
    locked, courses = event_locked(cur, ev)
    fits, coef = lagged_regression(cur, feats)
    curves_long = pd.concat([c.assign(token=t, pair=sn.loc[t, "pair"], condition=sn.loc[t, "condition"])
                             for t, c in cur.items()])
    return {"event_locked": locked, "event_courses": courses, "lag_scan": fits, "lagged_regression": coef,
            "press_curves": curves_long, "snippet_events": pd.concat(evrows) if evrows else pd.DataFrame(),
            "press_summary": phase2.assign(share=phase2["held_s"] / phase2["duration_s"])
                             .groupby(["pair", "condition"])
                             .agg(listeners=("pid", "nunique"), mean_holds=("n_holds", "mean"),
                                  mean_share_held=("share", "mean")).reset_index()}
