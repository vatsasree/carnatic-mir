"""Loading the export and applying the pre-registered exclusion rules.

Participant exclusions (docs/EXPERIMENT_DESIGN.md, "Quality control and exclusion rules"):
  incomplete       status != complete
  headphone        failed the headphone check
  attention        answer other than 2 on the attention row
  straight_lining  the same value on more than 80 % of rating rows (averaged over scales)
  too_fast         total time under 50 % of the median completion time
Trial exclusions: practice trials; the tab was hidden while the music played.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from carnatic_mir.config import OUTPUTS, ROOT

STUDY = ROOT / "study"
DATA = Path(os.environ.get("STUDY_DATA", STUDY / "data"))
EXPORT = DATA / "export"
RESULTS = DATA / "results"          # aggregated results stay next to the data (not in git) until you copy them out
SAME_SINGER = {"P3", "P5", "P6", "P9", "P10"}
SCALES = ["valence", "arousal", "liking"]
FAMILIAR = {"no": 0, "maybe": 1, "yes": 2, "know_well": 3}
LYRICS = {"none": 0, "some": 1, "all": 2}
STRAIGHT_LINE = 0.80
TOO_FAST = 0.50


@dataclass
class Study:
    participants: pd.DataFrame      # all, with `excluded` and `reason`
    ratings: pd.DataFrame           # kept participants and trials, with `soothing`, `lullaby`
    phase2: pd.DataFrame            # kept participants and trials, one row per listener x snippet
    presses: pd.DataFrame           # holds of the kept phase-2 trials
    snippets: pd.DataFrame          # study/snippets.csv
    gems: list                      # gems_* columns present


def _read(d: Path, name: str) -> pd.DataFrame:
    p = d / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else pd.DataFrame()


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(export: Path = EXPORT) -> Study:
    P = _read(export, "participants.csv")
    R = _read(export, "ratings.csv")
    T = _read(export, "phase2_trials.csv")
    H = _read(export, "presses.csv")
    S = pd.read_csv(STUDY / "snippets.csv")
    if P.empty:
        raise SystemExit(f"no participants in {export}; run: python -m experiment.export")
    gems = sorted(c for c in R.columns if c.startswith("gems_"))

    # ---- participant exclusions, in the pre-registered order (first reason wins) ----
    P = P.copy()
    P["reason"] = ""
    def flag(mask, why):
        P.loc[mask & (P["reason"] == ""), "reason"] = why
    flag(P["status"] != "complete", "incomplete")
    flag(~_truthy(P.get("headphone_passed", pd.Series(False, index=P.index))), "headphone")
    main = R[R["condition"] != "practice"]
    if "attention" in main:
        att = main[_truthy(main["attention_trial"])].set_index("pid")["attention"]
        flag(P["pid"].map(att).notna() & (P["pid"].map(att) != 2), "attention")
    cols = [c for c in gems + SCALES if c in main]
    if cols:
        modal = main.groupby("pid")[cols].agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else 0)
        flag(P["pid"].map(modal.mean(axis=1)) > STRAIGHT_LINE, "straight_lining")
    dur = P["finished"] - P["created"]
    med = dur[P["status"] == "complete"].median()
    flag(dur < TOO_FAST * med, "too_fast")
    P["excluded"] = P["reason"] != ""
    P["duration_min"] = (dur / 60).round(1)
    keep = set(P.loc[~P["excluded"], "pid"])

    # ---- trials ----
    R = R[R["pid"].isin(keep) & (R["condition"] != "practice")].copy()
    if "tab_hidden_during_audio" in R:
        R = R[R["tab_hidden_during_audio"].fillna(0) == 0]
    R["lullaby"] = (R["condition"] == "lullaby").astype(int)
    R["soothing"] = R[["gems_peacefulness", "gems_tenderness"]].mean(axis=1)
    R["familiar_n"] = R["familiar"].map(FAMILIAR)
    R["lyrics_n"] = R["lyrics"].map(LYRICS)
    R["same_singer"] = R["pair"].isin(SAME_SINGER)
    prof = P.set_index("pid")
    for c in ("age", "years_indian_training", "mood_valence", "mood_arousal", "lullaby_heard", "lullaby_sung"):
        if f"profile.{c}" in prof:
            R[c] = R["pid"].map(prof[f"profile.{c}"])

    if not T.empty:
        T = T[T["pid"].isin(keep) & (T["condition"] != "practice")].copy()
        T = T[T["tab_hidden_during_audio"].fillna(0) == 0]
        H = H.merge(T[["pid", "token"]], on=["pid", "token"]) if not H.empty else H
    return Study(P, R, T, H, S, gems)


def snippet_frames(track_id: str, start_s: float, end_s: float) -> pd.DataFrame:
    """The track's 10 ms time series cut to the snippet, with time relative to the snippet start."""
    ts = pd.read_csv(OUTPUTS / track_id / "timeseries.csv")
    ts = ts[(ts["time"] >= start_s) & (ts["time"] < end_s)].copy()
    ts["t"] = ts["time"] - start_s
    for c in ("voiced", "elongation", "perc_active", "stable"):
        if c in ts:
            ts[c] = _truthy(ts[c])
    return ts.reset_index(drop=True)


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    p = p.astype(float)
    order = p.sort_values().index
    m = len(p)
    adj, running = pd.Series(index=p.index, dtype=float), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def save(df: pd.DataFrame, name: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False)
    return df


rng = np.random.default_rng(20261109)       # fixed seed: permutation p-values are reproducible
