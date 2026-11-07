"""Feature stage: combine cached frame tables (pitch, voice, percussion) + tonic into
`timeseries.csv` and `features.json`. Cheap (~1 s), so it reruns on every tonic/param change."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import melodic, percussion, voice
from .config import HOP_S


def _read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def compute(out: Path, meta: dict, params: dict | None = None) -> dict:
    frames = pd.read_csv(out / "frames.csv")
    n = len(frames)
    vf, js = _read(out / "voice_frames.csv"), _read(out / "voice_segments.csv")
    pf, st = _read(out / "percussion_frames.csv"), _read(out / "strokes.csv")
    if pf is not None:
        pf = pf.iloc[:n].reset_index(drop=True)

    perc = percussion.summarize(pf, st, meta.get("percussion_info", {}),
                                vf["rms_dbfs"].to_numpy() if vf is not None else np.full(n, np.nan),
                                frames["voiced"].to_numpy(bool)) if pf is not None else {}
    tempo = perc.get("tempo_bpm")
    mel = melodic.analyze(frames, meta["tonic_hz"], tempo_bpm=tempo, params=params)

    summary = {"general": {
        "duration_s": round(n * HOP_S, 2), "tonic_hz": round(float(meta["tonic_hz"]), 2),
        "tonic_source": meta.get("tonic_source"), "separation": meta.get("separation"),
        "pitch_method": meta.get("pitch_method"),
    }}
    summary["range"] = mel["summary"]["range"]
    tempo_keys = ["tempo_bpm", "tempo_candidates_bpm", "tempo_stability_iqr_rel", "active_ratio"]
    summary["tempo"] = {("percussion_active_ratio" if k == "active_ratio" else k): perc.get(k) for k in tempo_keys}
    summary["tempo"]["melodic_turning_points_per_s"] = mel["summary"]["melody"]["turning_points_per_voiced_s"]
    for g in ("elongation", "glides", "sphurita", "oscillation", "melody", "intonation"):
        summary[g] = mel["summary"][g]
    mfcc = []
    if vf is not None:
        summary["voice"], mfcc = voice.summarize(
            vf, js if js is not None else pd.DataFrame(columns=["jitter_local", "shimmer_local"]),
            frames, mel, pf["accomp_dbfs"].to_numpy() if pf is not None else None)
    summary["percussion"] = {k: v for k, v in perc.items() if k not in tempo_keys}

    ts = pd.concat([frames[["time", "f0_hz", "confidence", "voiced"]], mel["timeseries"]], axis=1)
    if vf is not None:
        ts = pd.concat([ts, vf[[c for c in vf.columns if not c.startswith("mfcc")]]], axis=1)
    if pf is not None:
        ts = pd.concat([ts, pf], axis=1)
    ts.to_csv(out / "timeseries.csv", index=False, float_format="%.4f")

    feats = {
        "summary": summary, "params": mel["params"], "events": mel["events"],
        "intonation_per_svara": mel["intonation_per_svara"],
        "histogram_folded": mel["histogram_folded"],
        "svara_distribution_all_frames": mel["svara_distribution_all_frames"],
        "svara_distribution_stable_notes": mel["svara_distribution_stable_notes"],
        "voice_mfcc_mean": mfcc,
        "meta": meta,
    }
    (out / "features.json").write_text(json.dumps(feats, indent=1, default=_json_default))
    return feats


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def flatten_summary(summary: dict) -> dict:
    return {f"{g}.{k}": v for g, d in summary.items() for k, v in d.items()}
