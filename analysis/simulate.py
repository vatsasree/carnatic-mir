"""Fake export with KNOWN effects, to check that the analysis recovers them (never mix with real data).

    python -m analysis.simulate --out <dir> [--n 36] [--effect 0.6]
    python -m analysis.report --export <dir> --out <dir>/results --quick

Built in: lullaby effect on soothing = --effect scale points; listeners press 1-3 s after plain held notes;
one participant uses speakers, one fails the attention check, one straight-lines, one is incomplete.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from carnatic_mir.config import ROOT
from .common import snippet_frames
from .h2 import events

GEMS = ["wonder", "transcendence", "power", "tenderness", "nostalgia", "peacefulness", "joyful_activation",
        "tension", "sadness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--effect", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--press-after-held", type=float, default=0.6,
                    help="probability of a press 1-3 s after each plain held note (0 = presses at random only)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "study/config.json").read_text())
    sn = pd.read_csv(ROOT / "study/snippets.csv")
    main_sn = sn[sn["pair"] != "PRACTICE"]
    song_eff = {t: rng.normal(0, 0.4) for t in sn["token"]}
    ev = {r.token: events(snippet_frames(r.track_id, r.start_s, r.end_s)) for r in sn.itertuples()}

    P, R, T, H = [], [], [], []
    now = time.time()
    for i in range(a.n):
        pid = f"sim{i:03d}"; set_id = "AB"[i % 2]; age = int(rng.integers(18, 70))
        status = "started" if i == a.n - 1 else "complete"
        P.append({"pid": pid, "created": now, "finished": now + rng.normal(1500, 200), "status": status,
                  "completion_code": "X", "link": "sim", "age_group": "-", "set_id": set_id,
                  "profile.age": age, "profile.years_indian_training": int(rng.integers(0, 10)),
                  "profile.mood_valence": int(rng.integers(3, 8)),
                  "profile.headphones": "speakers" if i == 1 else "in_ear", "attention_trial": 3})
        person = rng.normal(0, 0.5)
        toks = [rng.choice(main_sn[(main_sn["pair"] == p) & (main_sn["condition"] == c)]["token"].to_numpy())
                for c, ps in cfg["sets"][set_id].items() for p in ps]
        for pos, tok in enumerate(rng.permutation(toks)):
            r = sn.set_index("token").loc[tok]
            lull = r["condition"] == "lullaby"
            base = 3 + person + song_eff[tok] + a.effect * lull
            row = {"pid": pid, "token": tok, "pair": r["pair"], "condition": r["condition"],
                   "version": r.get("version", 1), "track_id": r["track_id"],
                   "snippet_start_s": r["start_s"], "snippet_end_s": r["end_s"], "position": pos,
                   "attention_trial": pos == 3, "tab_hidden_during_audio": 0,
                   "valence": int(np.clip(round(5 + rng.normal(0, 1.5)), 1, 9)),
                   "arousal": int(np.clip(round(5 - 1.0 * lull + rng.normal(0, 1.5)), 1, 9)),
                   "liking": int(np.clip(round(3 + rng.normal(0, 1)), 1, 5)),
                   "familiar": rng.choice(["no", "maybe", "yes", "know_well"], p=[.5, .2, .2, .1]),
                   "lyrics": rng.choice(["all", "some", "none"]), "words": rng.choice(["calm", "sleepy, calm", "happy"]),
                   "rt_s": 30.0}
            for g in GEMS:
                mu = base if g in ("peacefulness", "tenderness") else 3 + rng.normal(0, 0.5)
                row[f"gems_{g}"] = int(np.clip(round(mu + rng.normal(0, 0.7)), 1, 5))
            if pos == 3:
                row["attention"] = 3 if i == 2 else 2
            if i == 3:                                   # straight-liner
                for k in [f"gems_{g}" for g in GEMS] + ["valence", "arousal", "liking"]:
                    row[k] = 3
            R.append(row)
            # phase 2: holds 1-3 s after plain held notes (60 %), plus background presses
            dur = float(r["end_s"] - r["start_s"])
            holds = []
            for e in ev[tok][ev[tok]["event"] == "elongation_plain"]["t"]:
                if rng.random() < a.press_after_held:
                    s = e + rng.uniform(0.7, 1.3); holds.append([s, min(dur, s + rng.uniform(1, 3))])
            for s in rng.uniform(0, dur, rng.poisson(0.06 * dur)):
                holds.append([s, min(dur, s + rng.uniform(0.5, 2))])
            holds.sort(); merged = []
            for h in holds:
                if merged and h[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], h[1])
                else:
                    merged.append(h)
            T.append({"pid": pid, "token": tok, "pair": r["pair"], "condition": r["condition"],
                      "track_id": r["track_id"], "label": "calm or soothing", "n_holds": len(merged),
                      "held_s": sum(b - a_ for a_, b in merged), "duration_s": dur, "position": pos,
                      "output_latency_s": 0.03, "input": "mouse/keyboard", "tab_hidden_during_audio": 0,
                      "attempts": int(rng.choice([1, 2], p=[.8, .2]))})
            for k, (s, e) in enumerate(merged):
                H.append({"pid": pid, "token": tok, "pair": r["pair"], "condition": r["condition"],
                          "label": "calm or soothing", "n": k, "start_s": round(s + 0.03, 3), "end_s": round(e + 0.03, 3),
                          "duration_s": round(e - s, 3), "output_latency_s": 0.03})
    for name, rows in (("participants", P), ("ratings", R), ("phase2_trials", T), ("presses", H)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    print(f"simulated {a.n} participants -> {out} (lullaby effect {a.effect}, presses after plain held notes)")


if __name__ == "__main__":
    main()
