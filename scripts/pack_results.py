"""Make a results folder small enough for git: encode each stem .wav as AAC .m4a next to it.

    python scripts/pack_results.py results/h100

The viewer plays .m4a when the .wav is absent, and .gitignore keeps the .wav files out of git.
Features (CSV/JSON) and waveform peaks are already small and are committed as they are.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BITRATE = {"mix": "128k", "vocals": "128k", "accompaniment": "96k", "percussion": "96k"}
LIMIT = 95 * 1024 * 1024


def encode(wav: Path):
    dst = wav.with_suffix(".m4a")
    if dst.exists() and dst.stat().st_mtime >= wav.stat().st_mtime:
        return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac",
                    "-b:a", BITRATE[wav.stem], "-movflags", "+faststart", str(dst)], check=True)


def main(root: Path):
    wavs = [w for w in root.glob("*/*.wav") if w.stem in BITRATE]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(encode, wavs))
    big = [p for p in root.rglob("*") if p.is_file() and p.suffix != ".wav" and p.stat().st_size > LIMIT]
    for p in big:
        print(f"WARNING over GitHub's limit, shard it: python scripts/shard.py split '{p}'")
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p.suffix != ".wav")
    print(f"{len(wavs)} stems encoded; {size / 1e6:.0f} MB to commit in {root}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/h100"))
