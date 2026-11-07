"""Score detected ornaments against synthetic ground truth, and check softness feature directions.

    python scripts/eval_synthetic.py            # expects outputs/synthetic_soft, outputs/synthetic_hard
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def match(gt, det, key_gt=("start", "end"), key_det=("start", "end"), tol=0.0):
    """Greedy match by time overlap; returns (tp, fn list, fp list)."""
    used, tp, fn = set(), 0, []
    for g in gt:
        best = None
        for j, d in enumerate(det):
            if j in used:
                continue
            if overlap(g[key_gt[0]] - tol, g[key_gt[1]] + tol, d[key_det[0]], d[key_det[1]]) > 0:
                best = j
                break
        if best is None:
            fn.append(g)
        else:
            used.add(best)
            tp += 1
    fp = [d for j, d in enumerate(det) if j not in used]
    return tp, fn, fp


def evaluate(track):
    gt = json.loads((ROOT / "data/raw" / f"{track}.gt.json").read_text())
    f = json.loads((Path(os.environ.get("CARNATIC_MIR_OUTPUTS", ROOT / "outputs")) / track / "features.json").read_text())
    ev, G = f["events"], gt["events"]
    rows = []
    osc = ev["oscillations"]
    for name, gts, det in [
        ("glides", [g for g in G if g["type"] == "glide"], ev["glides"]),
        ("simple oscillation", [g for g in G if g["type"] == "simple_oscillation"], [o for o in osc if o["type"] == "simple"]),
        ("heavy kampita", [g for g in G if g["type"] == "heavy_kampita"], [o for o in osc if o["type"] == "heavy_kampita"]),
        ("elongations", [g for g in G if g["type"] == "elongation"], ev["elongations"]),
    ]:
        tp, fn, fp = match(gts, det)
        rows.append((name, len(gts), tp, len(fp), fn, fp))
    sg = [dict(g, start=g["time"] - 0.05, end=g["time"] + 0.05) for g in G if g["type"] == "sphurita"]
    tp, fn, fp = match(sg, ev["sphurita"])
    rows.append(("sphurita", len(sg), tp, len(fp), fn, fp))

    print(f"\n== {track}  (tonic est {f['summary']['general']['tonic_hz']} Hz, true {gt['tonic_hz']})")
    print(f"{'event':20s} {'true':>4s} {'hit':>4s} {'false+':>6s}")
    for name, n, tp, nfp, fn, fp in rows:
        print(f"{name:20s} {n:4d} {tp:4d} {nfp:6d}" + (f"   missed@{[round(x.get('start', x.get('time', 0)), 2) for x in fn]}" if fn else "")
              + (f"   false@{[x.get('start', x.get('time')) for x in fp]}" if fp else ""))
    for o in osc:
        print(f"   osc {o['start']:6.2f}-{o['end']:6.2f}  {o['type']:13s} extent {o['extent_cents']:6.1f}c  rate {o['rate_hz']} Hz  "
              f"{o['lower_svara']}<->{o['upper_svara']}")
    s = f["summary"]
    print("range:", s["range"]["lowest_svara_p1"], "->", s["range"]["highest_svara_p99"],
          f"({s['range']['range_p1_p99_octaves']} oct)   tempo:", s["tempo"]["tempo_bpm"],
          "cands", s["tempo"]["tempo_candidates_bpm"], f"(true beat {gt['beat_bpm']}, strokes {gt['stroke_rate_per_s']:.2f}/s)")
    return s


if __name__ == "__main__":
    soft, hard = evaluate("synthetic_soft"), evaluate("synthetic_hard")
    print("\n== softness directions (soft vs hard; ✓ = expected direction)")
    checks = [
        ("voice", "harmonic_tilt_db_per_oct", "<"), ("voice", "alpha_ratio_db", "<"), ("voice", "hammarberg_db", ">"),
        ("voice", "h1h2_db_held_notes", ">"), ("voice", "hnr_db_held_notes", "<"), ("voice", "cpp_db_held_notes", "<"),
        ("voice", "brightness_centroid_hz", "<"), ("voice", "phrase_attack_time_ms", ">"),
        ("percussion", "stroke_attack_ms_median", ">"), ("percussion", "stroke_centroid_hz_median", "<"),
        ("percussion", "stroke_crest_db_median", "<"), ("percussion", "perc_to_vocal_db", "<"),
        ("percussion", "principal_stroke_rate_per_s", "="),
    ]
    for g, k, op in checks:
        a, b = soft[g].get(k), hard[g].get(k)
        ok = None if a is None or b is None else (a < b if op == "<" else a > b if op == ">" else abs(a - b) < 0.3)
        print(f"  {g}.{k:34s} soft {a!s:>9}  hard {b!s:>9}  {'✓' if ok else '✗' if ok is False else '?'}")
