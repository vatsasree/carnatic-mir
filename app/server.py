"""Viewer backend.  Run:  python -m app.server   then open http://127.0.0.1:8765"""
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from carnatic_mir.config import OUTPUTS, SEPARATION_PRESETS, UPLOADS
from carnatic_mir.pipeline import STEMS, process, recompute_features, track_id_for

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Carnatic MIR viewer")
OUTPUTS.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUTS), name="files")

# ---- single background worker: one model on the GPU at a time ---------------------------
jobs: dict[str, dict] = {}
_q: "queue.Queue[tuple]" = queue.Queue()


def _worker():
    while True:
        jid, path, kw = _q.get()
        job = jobs[jid]
        job.update(state="running", started=time.time())
        try:
            process(path, log=lambda m: job["log"].append(f"{time.strftime('%H:%M:%S')} {m}"), **kw)
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001
            job.update(state="error", error=str(e))
            job["log"].append(traceback.format_exc())
        job["finished"] = time.time()


threading.Thread(target=_worker, daemon=True).start()


def _enqueue(path: Path, separation: str, pitch: str, force: bool = False) -> dict:
    tid = track_id_for(path)
    jobs[tid] = {"track_id": tid, "state": "queued", "log": [], "file": path.name}
    _q.put((tid, path, dict(separation=separation, pitch_method=pitch, force=force)))
    return jobs[tid]


# ---- API ----------------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"separation_presets": list(SEPARATION_PRESETS), "pitch_methods": ["crepe", "crepe-tiny", "pyin"]}


@app.get("/api/tracks")
def tracks():
    out = []
    for d in sorted(OUTPUTS.iterdir()):
        if not (d / "features.json").exists() or not (d / "timeseries.csv").exists():
            continue
        try:                            # a partly copied folder must not hide every other track
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            meta = {}
        missing = [f for f in ("meta.json", "frames.csv") if not (d / f).exists()]
        if not any((d / f"vocals{e}").exists() for e in (".wav", ".m4a")):
            missing.append("vocals")
        out.append({"track_id": d.name, "tonic_hz": meta.get("tonic_hz"),
                    "separation": meta.get("separation"), "pitch_method": meta.get("pitch_method"),
                    "missing": missing})
    return {"tracks": out, "jobs": list(jobs.values())}


@app.get("/api/jobs")
def job_list():
    return list(jobs.values())


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), separation: str = Form("bs_roformer"),
                 pitch: str = Form("crepe")):
    dst = UPLOADS / Path(file.filename).name
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return _enqueue(dst, separation, pitch)


class ProcessReq(BaseModel):
    path: str
    separation: str = "bs_roformer"
    pitch: str = "crepe"
    force: bool = False


@app.post("/api/process")
def process_local(req: ProcessReq):
    """Process a file already on disk (e.g. a dataset folder) without uploading."""
    p = Path(req.path).expanduser()
    if not p.is_file():
        raise HTTPException(404, f"no such file: {p}")
    return _enqueue(p, req.separation, req.pitch, req.force)


def _clean(a):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(x) else round(float(x), 2) for x in a]


@app.get("/api/tracks/{tid}")
def track(tid: str):
    d = OUTPUTS / tid
    if not (d / "features.json").exists():
        raise HTTPException(404)
    f = json.loads((d / "features.json").read_text())
    ts = pd.read_csv(d / "timeseries.csv")
    cols = ["time", "cents", "f0_hz", "confidence", "rms_dbfs", "perc_dbfs", "local_tempo_bpm", "centroid_hz"]
    f["timeseries"] = {c: _clean(ts[c]) for c in cols if c in ts}
    for c in ("event", "elongation"):
        if c in ts:
            f["timeseries"][c] = ts[c].astype(int).tolist()
    if (d / "strokes.csv").exists():
        st = pd.read_csv(d / "strokes.csv")
        f["strokes"] = {c: _clean(st[c]) for c in ("time", "peak_dbfs", "attack_ms", "centroid_hz")}
    f["stems"] = {}
    for s in STEMS:                     # .wav locally; .m4a for results brought back from a server
        audio = next((d / f"{s}{ext}" for ext in (".wav", ".m4a") if (d / f"{s}{ext}").exists()), None)
        if audio is not None and (d / f"peaks_{s}.json").exists():
            f["stems"][s] = {"url": f"/files/{tid}/{audio.name}", "peaks": f"/files/{tid}/peaks_{s}.json"}
    f["track_id"] = tid
    return f


@app.get("/api/summary")
def summary_all():
    """Flattened summaries of every processed track (for the compare page)."""
    from carnatic_mir.features import flatten_summary
    rows = []
    for fj in sorted(OUTPUTS.glob("*/features.json")):
        rows.append({"track_id": fj.parent.name, **flatten_summary(json.loads(fj.read_text())["summary"])})
    return rows


@app.get("/compare")
def compare_page():
    return FileResponse(STATIC / "compare.html")


class TonicReq(BaseModel):
    tonic_hz: float


@app.post("/api/tracks/{tid}/tonic")
def set_tonic(tid: str, req: TonicReq):
    d = OUTPUTS / tid
    if not (d / "frames.csv").exists():
        raise HTTPException(404)
    recompute_features(d, tonic_hz=req.tonic_hz)
    return track(tid)


@app.get("/api/tracks/{tid}/download/{name}")
def download(tid: str, name: str):
    if name not in {"timeseries.csv", "features.json", "frames.csv", "strokes.csv", "voice_frames.csv",
                    "percussion_frames.csv", "voice_segments.csv"}:
        raise HTTPException(404)
    return FileResponse(OUTPUTS / tid / name, filename=f"{tid}_{name}")


if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8765)))
