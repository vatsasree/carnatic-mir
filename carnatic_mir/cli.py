"""Batch CLI.

    python -m carnatic_mir.cli song1.mp3 song2.wav --separation bs_roformer --pitch crepe
    python -m carnatic_mir.cli data/raw/*.mp3                       # also writes outputs/summary.csv
    python -m carnatic_mir.cli mix.wav --vocals vocal_mic.wav       # skip separation (e.g. Saraga multitrack)
    python -m carnatic_mir.cli song.mp3 --start 10:00 --end 15:00   # analyse an excerpt only
    python -m carnatic_mir.cli outputs/my_track --tonic 146.8       # only re-tune tonic (~1 s)
    python -m carnatic_mir.cli --summary                            # all tracks -> outputs/summary.csv
    python -m carnatic_mir.cli "https://youtu.be/<id>" --start 10:00 --end 15:00   # YouTube -> data/raw -> analyse
    python -m carnatic_mir.cli --urls urls.txt                      # batch: 'URL [start end]' per line
    python -m carnatic_mir.cli --urls urls.txt --workers 4          # 4 tracks at a time (big GPU + many cores)
"""
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

from .config import OUTPUTS, SEPARATION_PRESETS
from .features import flatten_summary
from .pipeline import process, recompute_features
from .youtube import RAW, download, is_url, read_list


def _ts(s: str) -> float:
    """'600', '10:00' or '1:02:03' -> seconds."""
    sec = 0.0
    for part in s.split(":"):
        sec = sec * 60 + float(part)
    return sec


def collect_summary() -> Path:
    rows = []
    for f in sorted(OUTPUTS.glob("*/features.json")):
        d = json.loads(f.read_text())
        rows.append({"track_id": f.parent.name, **flatten_summary(d["summary"]),
                     **{f"svara_share.{k}": v for k, v in d["svara_distribution_all_frames"].items()}})
    dst = OUTPUTS / "summary.csv"
    pd.DataFrame(rows).to_csv(dst, index=False)
    return dst


def _local(src: str) -> Path:
    """An audio path as given, or a bare file name looked up in data/raw/."""
    p = Path(src).expanduser()
    if not p.exists() and (RAW / p.name).exists():
        return RAW / p.name
    if not p.exists():
        raise FileNotFoundError(src)
    return p


def _run_workers(n: int, n_jobs: int) -> int:
    """Re-launch this command n times, each on every n-th job; CPU threads are split between them."""
    n = min(n, n_jobs)
    argv = [x for x in sys.argv[1:]]
    for flag in ("--workers",):                     # drop '--workers N' / '--workers=N'
        argv = [x for j, x in enumerate(argv) if not (x == flag or x.startswith(flag + "=")
                                                      or (j > 0 and argv[j - 1] == flag))]
    logs = OUTPUTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    threads = str(max(1, (os.cpu_count() or n) // n))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env.setdefault(v, threads)
    procs = []
    for i in range(n):
        log = open(logs / f"worker{i}.log", "a")
        procs.append((subprocess.Popen([sys.executable, "-m", "carnatic_mir.cli", *argv, "--shard", f"{i}/{n}"],
                                       stdout=log, stderr=subprocess.STDOUT, env=env), log))
    print(f"{n} workers, {threads} CPU threads each; progress: tail -f {logs}/worker*.log")
    codes = [p.wait() for p, _ in procs]
    print("worker exit codes:", codes)
    print("summary ->", collect_summary())
    return max(codes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="audio files, YouTube URLs, or outputs/<track_id> dirs with --tonic")
    ap.add_argument("--separation", default="bs_roformer",
                    help=f"preset {list(SEPARATION_PRESETS)} or any audio-separator model filename")
    ap.add_argument("--pitch", default="crepe", choices=["crepe", "crepe-tiny", "pyin"])
    ap.add_argument("--vocals", help="clean vocal track to use instead of separation (single input only)")
    ap.add_argument("--tonic", type=float, help="override tonic (Sa) in Hz")
    ap.add_argument("--start", type=_ts, help="excerpt start, seconds or mm:ss")
    ap.add_argument("--end", type=_ts, help="excerpt end, seconds or mm:ss")
    ap.add_argument("--force", action="store_true", help="recompute every stage")
    ap.add_argument("--summary", action="store_true", help="write outputs/summary.csv for all tracks")
    ap.add_argument("--urls", "--list", dest="urls",
                    help="job list: one 'URL-or-audio-file [start [end]]' per line, # for comments")
    ap.add_argument("--workers", type=int, default=1,
                    help="process this many tracks in parallel (separate processes; for a big GPU with many CPU cores)")
    ap.add_argument("--shard", help=argparse.SUPPRESS)          # 'i/n', set by --workers
    ap.add_argument("--playlist", action="store_true", help="for a playlist URL, process every video")
    a = ap.parse_args()

    jobs = [(s, a.start, a.end) for s in a.inputs]          # (input, start s, end s)
    if a.urls:
        jobs += [(u, _ts(st) if st else a.start, _ts(en) if en else a.end) for u, st, en in read_list(a.urls)]
    if not jobs and not a.summary:
        ap.error("give inputs, --urls or --summary")
    if a.workers > 1 and len(jobs) > 1:
        return _run_workers(a.workers, len(jobs))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        jobs = jobs[i::n]

    failed, n_done = [], 0
    for i, (src, start, end) in enumerate(jobs, 1):
        if len(jobs) > 1:
            print(f"\n[{i}/{len(jobs)}] {src}")
        try:
            paths = download(src, playlist=a.playlist) if is_url(src) else [_local(src)]
            for p in paths:
                if p.is_dir() and (p / "frames.csv").exists():
                    recompute_features(p, tonic_hz=a.tonic)
                    print(f"recomputed {p}")
                else:
                    process(p, separation=a.separation, pitch_method=a.pitch, tonic_hz=a.tonic,
                            vocals_path=a.vocals, force=a.force, start=start, end=end)
                n_done += 1
        except Exception:
            if len(jobs) == 1:
                raise
            traceback.print_exc()
            failed.append(src)
    if len(jobs) > 1:
        print(f"\ndone: {len(jobs) - len(failed)} ok, {len(failed)} failed" +
              "".join(f"\n  failed: {s}" for s in failed))
    if a.shard:
        return 1 if failed else 0
    if a.summary or n_done > 1:
        print("summary ->", collect_summary())


if __name__ == "__main__":
    sys.exit(main())
