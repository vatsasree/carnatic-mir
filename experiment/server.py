"""Listening-study server.

    python -m experiment.links add pilot          # make a private study link (prints the URL path)
    python -m experiment.server                   # http://127.0.0.1:8770/?k=<link key>
    python -m experiment.export                   # responses -> study/data/export/*.csv

Participants only ever see random snippet tokens; which pair/condition a token belongs to stays on the server.
"""
import csv
import json
import os
import random
import secrets

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .links import load_links
from .paths import CONFIG, SNIPPET_DIR, SNIPPETS_CSV

STATIC = os.path.join(os.path.dirname(__file__), "static")
app = FastAPI(title="Listening study", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def config() -> dict:
    return json.loads(CONFIG.read_text())


def snippets() -> dict[tuple[str, str, str], dict]:
    """(pair, condition, version) -> row of study/snippets.csv."""
    with open(SNIPPETS_CSV, newline="") as f:
        return {(r["pair"], r["condition"], r.get("version") or "1"): r for r in csv.DictReader(f)}


def pick_version(sn: dict, pair: str, cond: str, used: dict) -> str:
    """The version of this song heard least so far (random on a tie): versions stay balanced per song."""
    toks = {k[2]: r["token"] for k, r in sn.items() if k[:2] == (pair, cond)}
    least = min(used.get(t, 0) for t in toks.values())
    return toks[random.choice([v for v, t in toks.items() if used.get(t, 0) == least])]


def _link(k: str | None) -> str:
    links = load_links()
    if not k or k not in links:
        raise HTTPException(403, "This study link is not valid. Please use the link you were given.")
    return links[k]


def _age_group(cfg: dict, age: int) -> str:
    for g in cfg["age_groups"]:
        if g["min"] <= age <= g["max"]:
            return g["id"]
    raise HTTPException(400, "Please check the age you entered.")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def public_config(k: str):
    _link(k)
    cfg = config()
    sn = snippets()["PRACTICE", "practice", "1"]          # not rated; only a 12 s excerpt for setting the volume
    return {"title": cfg["title"], "contact": cfg["contact"], "text": cfg["text"], "gems": cfg["gems"],
            "phase2_labels": cfg["phase2_labels"], "volume_sample": sn["token"]}


class StartReq(BaseModel):
    k: str
    profile: dict
    device: dict = {}


@app.post("/api/start")
def start(req: StartReq, request: Request):
    link = _link(req.k)
    cfg = config()
    try:
        age = int(req.profile.get("age"))
    except (TypeError, ValueError):
        raise HTTPException(400, "Please enter your age.")
    group = _age_group(cfg, age)
    n = store.counts(group if cfg.get("balance_sets_by_age_group", True) else None)
    least = min(n.get(s, 0) for s in cfg["sets"])
    set_id = random.choice([s for s in cfg["sets"] if n.get(s, 0) == least])      # balance A/B per age group
    sn = snippets()
    used = store.token_counts()
    items = [pick_version(sn, p, cond, used) for cond, pairs in cfg["sets"][set_id].items() for p in pairs]
    random.shuffle(items)
    order2 = items[:]
    random.shuffle(order2)
    assignment = {"phase1": items, "phase2": order2, "attention_trial": random.randrange(len(items))}
    pid = secrets.token_hex(6)
    store.add_participant(pid=pid, link=link, age_group=group, set_id=set_id, assignment=assignment,
                          profile={**req.profile, "device": req.device},
                          user_agent=request.headers.get("user-agent", ""))
    return {"pid": pid, "assignment": assignment, "age_group": group}


@app.get("/api/state")
def state(pid: str):
    p = store.get_participant(pid)
    if not p:
        raise HTTPException(404)
    return {"pid": pid, "assignment": json.loads(p["assignment"]), "status": p["status"],
            "completion_code": p["completion_code"], "saved": store.saved_keys(pid)}


class SaveReq(BaseModel):
    pid: str
    kind: str           # calibration | rating | presses | final | event
    item: str
    payload: dict | list
    client_ts: float | None = None


KINDS = {"headphone", "calibration", "rating", "presses", "final", "event"}


@app.post("/api/save")
def save(req: SaveReq):
    p = store.get_participant(req.pid)
    if not p or req.kind not in KINDS:
        raise HTTPException(400)
    store.save(req.pid, req.kind, req.item, req.payload, req.client_ts)
    return {"ok": True}


@app.post("/api/finish")
def finish(pid: str):
    p = store.get_participant(pid)
    if not p:
        raise HTTPException(404)
    code = p["completion_code"] or secrets.token_hex(4).upper()
    store.set_status(pid, "complete", code)
    return {"completion_code": code}


@app.get("/audio/{name}")
def audio(name: str):
    tokens = {r["token"] for r in snippets().values()}
    tok = name.removesuffix(".mp3")
    if tok not in tokens:
        raise HTTPException(404)
    return FileResponse(SNIPPET_DIR / f"{tok}.mp3", media_type="audio/mpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


@app.get("/admin/status")
def admin_status(x_admin_token: str = Header(default="")):
    tok = os.environ.get("STUDY_ADMIN_TOKEN")
    if not tok or not secrets.compare_digest(x_admin_token, tok):
        raise HTTPException(403)
    rows = store.db().execute("SELECT age_group, set_id, status, COUNT(*) n FROM participants "
                              "GROUP BY age_group, set_id, status").fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8770)))
