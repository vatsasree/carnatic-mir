"""Audio I/O helpers."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SR_MIX


def to_wav(src: Path, dst: Path, sr: int = SR_MIX, start: float | None = None, end: float | None = None) -> Path:
    """Decode any format ffmpeg understands (mp3, m4a, flac, webm, ...) to 16-bit stereo WAV,
    optionally cutting the excerpt [start, end) in seconds."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found (conda env `carnatic-mir`, or apt/brew install ffmpeg)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cut = (["-ss", str(start)] if start else []) + (["-to", str(end)] if end else [])
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), *cut, "-ac", "2", "-ar", str(sr),
         "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )
    return dst


def write_peaks(wav: Path, dst: Path, per_second: int = 100) -> None:
    """Precompute waveform peaks so the browser viewer never has to decode the full file."""
    y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    y = np.abs(y).max(axis=1)
    hop = max(1, sr // per_second)
    n = int(np.ceil(len(y) / hop))
    y = np.pad(y, (0, n * hop - len(y)))
    peaks = y.reshape(n, hop).max(axis=1)
    dst.write_text(json.dumps({"duration": len(y) / sr, "peaks": np.round(peaks, 3).tolist()}))
