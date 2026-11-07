"""End-to-end pipeline with per-stage caching (a stage reruns when its STAGE_VERSION changes).

outputs/<track_id>/
    mix.wav, vocals.wav, accompaniment.wav, percussion.wav   audio
    peaks_<stem>.json                     waveform peaks for the viewer
    frames.csv                            pitch stage: f0, confidence, voicing, rms (vocal, 10 ms)
    voice_frames.csv, voice_segments.csv  voice stage: timbre / voice quality frames, jitter+shimmer per held note
    percussion_frames.csv, strokes.csv    percussion stage: perc level, onsets, local tempo, per-stroke softness
    timeseries.csv, features.json         feature stage (tonic-dependent, recomputed in ~1 s)
    meta.json                             provenance: source, models, tonic source, stage versions, timings
"""
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .audio import to_wav, write_peaks
from .config import DEFAULT_SEPARATION, OUTPUTS

STEMS = ("mix", "vocals", "accompaniment", "percussion")
STAGE_VERSION = {"separation": 1, "pitch": 2, "voice": 2, "percussion": 3, "tonic": 1}


def track_id_for(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_")[:80] or "track"


def _meta(out: Path) -> dict:
    f = out / "meta.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_meta(out: Path, meta: dict):
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))


def process(audio_path, separation: str = DEFAULT_SEPARATION, pitch_method: str = "crepe",
            tonic_hz: float | None = None, vocals_path=None, force: bool = False,
            out_root: Path = OUTPUTS, log=print, start: float | None = None, end: float | None = None) -> Path:
    audio_path = Path(audio_path)
    tid = track_id_for(audio_path)
    if start or end:     # excerpts get their own track id, e.g. <name>_600-900s
        tid = f"{tid[:64]}_{int(start or 0)}-{int(end) if end else 'end'}s"
    out = Path(out_root) / tid
    out.mkdir(parents=True, exist_ok=True)
    meta = _meta(out)
    ver = meta.setdefault("stage_versions", {})
    if vocals_path:
        separation = "provided vocal track"
    dirty = set(STAGE_VERSION) if force else set()
    if meta.get("separation") not in (None, separation):
        dirty |= {"separation"}
    if meta.get("pitch_method") not in (None, pitch_method):
        dirty |= {"pitch"}
    dirty |= {s for s, v in STAGE_VERSION.items() if ver.get(s) != v}

    def need(stage, *files):
        return stage in dirty or not all((out / f).exists() for f in files)

    meta.update(track_id=tid, source=str(audio_path), separation=separation, pitch_method=pitch_method,
                excerpt_s=[start, end] if (start or end) else None)
    t0 = time.time()
    timings = {}
    clock = [time.time()]

    def lap(name):                       # seconds since the previous lap, recorded under `name`
        now = time.time()
        timings[name] = round(now - clock[0], 1)
        clock[0] = now

    if force or not (out / "mix.wav").exists():
        log("decoding audio")
        to_wav(audio_path, out / "mix.wav", start=start, end=end)
    lap("decode")

    # 1. separation (or user-provided clean vocal track)
    if need("separation", "vocals.wav", "accompaniment.wav"):
        t = time.time()
        if vocals_path:
            log("using provided vocal track")
            to_wav(Path(vocals_path), out / "vocals.wav", start=start, end=end)
            mix, sr = sf.read(str(out / "mix.wav"), dtype="float32", always_2d=True)
            voc, _ = sf.read(str(out / "vocals.wav"), dtype="float32", always_2d=True)
            k = min(len(mix), len(voc))
            sf.write(str(out / "accompaniment.wav"), np.clip(mix[:k] - voc[:k], -1, 1), sr, subtype="PCM_16")
            meta["separation_info"] = {"model": "provided", "vocals": str(vocals_path)}
        else:
            from .separation import separate_vocals
            meta["separation_info"] = separate_vocals(out / "mix.wav", out, separation, log=log)
        meta["separation_seconds"] = round(time.time() - t, 1)
        ver["separation"] = STAGE_VERSION["separation"]
        dirty |= {"pitch", "voice", "percussion"}
    lap("separation")

    # 2. pitch
    if need("pitch", "frames.csv"):
        from .pitch import extract_frame_features
        log(f"pitch: {pitch_method}")
        t = time.time()
        extract_frame_features(out / "vocals.wav", method=pitch_method).to_csv(out / "frames.csv", index=False)
        meta["pitch_seconds"] = round(time.time() - t, 1)
        ver["pitch"] = STAGE_VERSION["pitch"]
        dirty |= {"voice", "tonic"}
    frames = pd.read_csv(out / "frames.csv")
    lap("pitch")

    # 3. voice quality
    if need("voice", "voice_frames.csv", "voice_segments.csv"):
        from . import melodic, voice
        log("voice timbre / quality")
        t = time.time()
        voice.extract_voice_frames(out / "vocals.wav", frames).to_csv(out / "voice_frames.csv", index=False, float_format="%.4f")
        held = melodic.analyze(frames, 100.0)["events"]["stable_notes"]      # stability is tonic-invariant
        voice.jitter_shimmer(out / "vocals.wav", held, frames).to_csv(out / "voice_segments.csv", index=False)
        meta["voice_seconds"] = round(time.time() - t, 1)
        ver["voice"] = STAGE_VERSION["voice"]
    lap("voice")

    # 4. percussion
    if need("percussion", "percussion.wav", "percussion_frames.csv", "strokes.csv"):
        from .percussion import extract_percussion
        log("percussion: HPSS, tempo, strokes")
        t = time.time()
        meta["percussion_info"] = extract_percussion(out / "accompaniment.wav", out, len(frames))
        meta["percussion_seconds"] = round(time.time() - t, 1)
        ver["percussion"] = STAGE_VERSION["percussion"]
        (out / "peaks_percussion.json").unlink(missing_ok=True)
    lap("percussion")

    for stem in STEMS:
        if (out / f"{stem}.wav").exists() and (force or not (out / f"peaks_{stem}.json").exists()
                                                or (out / f"peaks_{stem}.json").stat().st_mtime < (out / f"{stem}.wav").stat().st_mtime):
            write_peaks(out / f"{stem}.wav", out / f"peaks_{stem}.json")
    lap("peaks")

    # 5. tonic
    if tonic_hz:
        meta.update(tonic_hz=float(tonic_hz), tonic_source="manual")
    elif "tonic_hz" not in meta or ("tonic" in dirty and meta.get("tonic_source") != "manual"):
        from .tonic import fit_tonic_octave, tonic_essentia, tonic_from_pitch
        log("tonic identification")
        try:
            t_hz, src = tonic_essentia(out / "mix.wav"), "essentia TonicIndianArtMusic (mix)"
        except Exception as e:  # noqa: BLE001
            log(f"essentia tonic failed ({e}); using pitch-histogram heuristic")
            t_hz, src = tonic_from_pitch(frames["f0_hz"].to_numpy()), "pitch-histogram heuristic"
        meta.update(tonic_hz=fit_tonic_octave(t_hz, frames["f0_hz"].to_numpy()), tonic_source=src, tonic_raw_hz=t_hz)
    ver["tonic"] = STAGE_VERSION["tonic"]
    _save_meta(out, meta)
    lap("tonic")

    # 6. features
    log("features")
    recompute_features(out)
    lap("features")
    meta = _meta(out)
    meta["total_seconds_last_run"] = round(time.time() - t0, 1)
    meta["timings_last_run"] = timings
    _save_meta(out, meta)
    shutil.rmtree(out / "_sep_tmp", ignore_errors=True)
    log("timings: " + " | ".join(f"{k} {v:.0f}s" for k, v in timings.items() if v >= 0.5)
        + f" | total {meta['total_seconds_last_run']:.0f}s")
    log(f"done -> {out}")
    return out


def recompute_features(out: Path, tonic_hz: float | None = None, params: dict | None = None) -> dict:
    from . import features
    out = Path(out)
    meta = _meta(out)
    if tonic_hz:
        meta.update(tonic_hz=float(tonic_hz), tonic_source="manual")
        _save_meta(out, meta)
    return features.compute(out, meta, params=params)
