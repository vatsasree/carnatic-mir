"""Export responses to tidy CSVs in study/data/export/.

    python -m experiment.export

participants.csv  one row per participant (profile flattened, assignment, status)
ratings.csv       one row per participant x snippet (pair, condition, GEMS, valence, arousal, ...)
presses.csv       one row per key hold in phase 2 (start/end in snippet seconds), final listening only
phase2_trials.csv one row per participant x snippet in phase 2 (n_holds may be 0; attempts = listenings)
events.csv        tab hidden, focus lost, audio errors, ...
"""
import csv
import json

from .paths import DATA_DIR, SNIPPETS_CSV
from .store import db

OUT = DATA_DIR / "export"


def _write(name, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = ";".join(map(str, v)) if isinstance(v, list) else v
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SNIPPETS_CSV, newline="") as f:
        sn = {r["token"]: r for r in csv.DictReader(f)}
    con = db()
    recs = [dict(r) for r in con.execute("SELECT * FROM records ORDER BY pid, server_ts")]
    hp = {r["pid"]: json.loads(r["payload"]) for r in recs if r["kind"] == "headphone" and r["item"] == "result"}

    parts = []
    for p in con.execute("SELECT * FROM participants ORDER BY created"):
        p = dict(p)
        a = json.loads(p.pop("assignment"))
        row = {k: p[k] for k in ("pid", "created", "finished", "status", "completion_code", "link", "age_group",
                                 "set_id")}
        row.update(_flat(json.loads(p["profile"]), "profile."))
        h = hp.get(p["pid"], {})                      # only for participants from before the tone check was removed
        row.update(headphone_correct=h.get("correct"), headphone_passed=h.get("passed"),
                   attention_trial=a["attention_trial"])
        parts.append(row)
    _write("participants.csv", parts)

    def snip(tok):
        s = sn.get(tok, {})
        return {"pair": s.get("pair"), "condition": s.get("condition"), "version": s.get("version") or "1",
                "track_id": s.get("track_id"),
                "snippet_start_s": s.get("start_s"), "snippet_end_s": s.get("end_s")}

    ratings, presses, phase2, events = [], [], [], []
    for r in recs:
        pl = json.loads(r["payload"])
        if r["kind"] == "rating":
            ratings.append({"pid": r["pid"], "token": r["item"], **snip(r["item"]), **_flat(pl)})
        elif r["kind"] == "presses":
            tok, _, label = r["item"].partition("|")
            phase2.append({"pid": r["pid"], "token": tok, **snip(tok), "label": label, "n_holds": len(pl["holds"]),
                           "held_s": round(sum(t1 - t0 for t0, t1 in pl["holds"]), 3),
                           "duration_s": pl.get("duration_s"), "position": pl.get("position"),
                           "attempts": pl.get("attempts", 1),
                           "earlier_holds": json.dumps([e["holds"] for e in pl.get("earlier_attempts", [])]),
                           "output_latency_s": pl.get("output_latency_s"), "input": pl.get("input"),
                           "tab_hidden_during_audio": pl.get("tab_hidden_during_audio")})
            for i, (t0, t1) in enumerate(pl["holds"]):
                presses.append({"pid": r["pid"], "token": tok, **snip(tok), "label": label, "n": i,
                                "start_s": t0, "end_s": t1, "duration_s": round(t1 - t0, 3),
                                "output_latency_s": pl.get("output_latency_s")})
        elif r["kind"] in ("event", "calibration", "final"):
            events.append({"pid": r["pid"], "kind": r["kind"], "item": r["item"], **_flat(pl)})
    _write("ratings.csv", ratings)
    _write("presses.csv", presses)
    _write("phase2_trials.csv", phase2)      # one row per listener x snippet, also when nothing was marked
    _write("events.csv", events)


if __name__ == "__main__":
    main()
