"""Choose and cut the listening snippets.

    python -m experiment.snippets suggest     # writes study/snippets.csv with a proposed window per song
    (check / edit start_s and end_s in the viewer, e.g. http://127.0.0.1:8765/#<track_id>)
    python -m experiment.snippets cut         # writes study/snippets/<token>.mp3 (loudness-normalised)

A suggested window starts just before a sung phrase and ends just after one, lasts MIN..MAX seconds and has
the highest share of sung (voiced) frames. File names are random tokens, so the URL never reveals the condition.
"""
import argparse
import csv
import json
import secrets
import subprocess
import sys

import numpy as np
import pandas as pd

from .paths import PAIRS, SNIPPET_DIR, SNIPPETS_CSV, TRACKS

MIN_S, MAX_S, TARGET_S = 30.0, 45.0, 40.0
PAD_S = 0.3            # keep the breath before / decay after a phrase
GAP_S = 0.3            # silence that separates two phrases
LUFS, TRUE_PEAK = -23.0, -2.0
FADE_S = 0.5
PRACTICE_TRACK = "Moksamu_Galada_Bhuvilo-_Saramati-_Adi-_Thyagaraja-_ML_Vasanthaku_600-900s"
FIELDS = ["pair", "condition", "track_id", "start_s", "end_s", "duration_s", "voiced_ratio", "token", "note"]


def phrases(track_id: str, gap_s: float = GAP_S) -> tuple[np.ndarray, np.ndarray, float]:
    """(onsets, offsets, duration) of sung phrases from the track's pitch time series."""
    ts = pd.read_csv(TRACKS / track_id / "timeseries.csv", usecols=["time", "voiced"])
    t = ts["time"].to_numpy()
    v = ts["voiced"].astype(str).str.lower().eq("true").to_numpy()
    hop = float(np.median(np.diff(t)))
    edges = np.diff(np.r_[0, v.astype(int), 0])
    on, off = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)        # off is exclusive
    keep_on, keep_off = [], []
    for a, b in zip(on, off):                                                  # merge runs split by short gaps
        if keep_off and (a - keep_off[-1]) * hop < gap_s:
            keep_off[-1] = b
        else:
            keep_on.append(a); keep_off.append(b)
    return t[np.array(keep_on, int)], t[np.array(keep_off, int) - 1] + hop, float(t[-1] + hop), (t, v)


def suggest(track_id: str) -> dict:
    """Best window by voiced share, starting/ending at phrase boundaries."""
    for gap in (GAP_S, 0.15, 0.08):                       # fall back to shorter pauses for continuous singing
        on, off, dur, (t, v) = phrases(track_id, gap)
        cv = np.r_[0, np.cumsum(v)]
        best = None
        for a in on:
            s = max(0.0, a - PAD_S)
            for b in off[off > a]:
                e = min(dur, b + PAD_S)
                L = e - s
                if L > MAX_S:
                    break
                if L < MIN_S:
                    continue
                i, j = np.searchsorted(t, [s, e])
                ratio = (cv[j] - cv[i]) / max(j - i, 1)
                score = ratio - 0.002 * abs(L - TARGET_S)
                if best is None or score > best[0]:
                    best = (score, s, e, ratio)
        if best:
            _, s, e, ratio = best
            return dict(start_s=round(s, 2), end_s=round(e, 2), duration_s=round(e - s, 2),
                        voiced_ratio=round(ratio, 3), note="auto" if gap == GAP_S else f"auto (gap {gap}s)")
    s = max(0.0, min(dur - TARGET_S, dur / 3))            # no phrase structure found: fixed window
    return dict(start_s=round(s, 2), end_s=round(s + TARGET_S, 2), duration_s=TARGET_S, voiced_ratio=None,
                note="fixed window - CHECK")


def read_rows() -> list[dict]:
    with open(SNIPPETS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict]):
    with open(SNIPPETS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)


def cmd_suggest(force: bool):
    old = {(r["pair"], r["condition"]): r for r in read_rows()} if SNIPPETS_CSV.exists() else {}
    with open(PAIRS, newline="") as f:
        jobs = [(r["pair"], r["condition"], r["track_id"]) for r in csv.DictReader(f)]
    jobs.append(("PRACTICE", "practice", PRACTICE_TRACK))
    rows = []
    for pair, cond, tid in jobs:
        prev = old.get((pair, cond))
        if prev and prev["track_id"] == tid and not force:   # keep hand-edited windows
            rows.append(prev); continue
        if not (TRACKS / tid / "timeseries.csv").exists():
            sys.exit(f"{pair} {cond}: no features for {tid!r}")
        r = dict(pair=pair, condition=cond, track_id=tid, token=(prev or {}).get("token") or secrets.token_hex(6))
        r.update(suggest(tid))
        rows.append(r)
        print(f"{pair:>8} {cond:<11} {r['start_s']:>7.1f}-{r['end_s']:<7.1f} ({r['duration_s']:.1f}s, "
              f"voiced {r['voiced_ratio']})  {tid[:50]}")
    write_rows(rows)
    print(f"wrote {SNIPPETS_CSV}; check the windows in the viewer, edit start_s/end_s if needed, then run 'cut'")


def _loudnorm_args(src, s, d) -> str:
    """Two-pass EBU R128 normalisation: measure first, then apply with the measured values."""
    flt = f"loudnorm=I={LUFS}:TP={TRUE_PEAK}:LRA=20:print_format=json"
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(s), "-t", str(d), "-i", str(src),
                        "-af", flt, "-f", "null", "-"], capture_output=True, text=True, check=True)
    m = json.loads(p.stderr[p.stderr.rindex("{"):p.stderr.rindex("}") + 1])
    return (f"loudnorm=I={LUFS}:TP={TRUE_PEAK}:LRA=20:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
            f"linear=true")


def cmd_cut():
    SNIPPET_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    for r in rows:
        src = TRACKS / r["track_id"] / "mix.wav"
        if not src.exists():
            src = TRACKS / r["track_id"] / "mix.m4a"
        s, e = float(r["start_s"]), float(r["end_s"])
        d = e - s
        out = SNIPPET_DIR / f"{r['token']}.mp3"
        af = f"{_loudnorm_args(src, s, d)},afade=t=in:d={FADE_S},afade=t=out:st={d - FADE_S}:d={FADE_S},aresample=44100"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(s), "-t", str(d), "-i",
                        str(src), "-af", af, "-ac", "2", "-c:a", "libmp3lame", "-b:a", "192k", str(out)], check=True)
        r["duration_s"] = round(d, 2)
        print(f"{r['pair']:>8} {r['condition']:<11} -> {out.name}  ({d:.1f}s)")
    write_rows(rows)
    print(f"{len(rows)} snippets in {SNIPPET_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("suggest"); sp.add_argument("--force", action="store_true", help="overwrite edited windows")
    sub.add_parser("cut")
    a = ap.parse_args()
    cmd_suggest(a.force) if a.cmd == "suggest" else cmd_cut()


if __name__ == "__main__":
    main()
