"""Download audio from YouTube (or any site yt-dlp supports) into data/raw/.

    python -m carnatic_mir.youtube <url> [<url> ...]          # download only
    python -m carnatic_mir.youtube --file urls.txt            # batch download from a list
    python -m carnatic_mir.cli <url> --start 10:00 --end 15:00   # download + analyse
    python -m carnatic_mir.cli --urls urls.txt                # batch download + analyse

urls.txt: one URL per line, optionally followed by an excerpt start and end (seconds or mm:ss);
blank lines and lines starting with # are ignored. Example:
    # MLV
    https://www.youtube.com/watch?v=XXXX  10:00  15:00
    https://youtu.be/YYYY

Keeps the original audio stream (usually Opus/AAC, no re-encoding); ffmpeg reads it directly.
Already-downloaded videos (matched by video id) are not fetched again.
Only download recordings you have the right to use.
"""
import os
import re
import sys
from pathlib import Path

from .config import ROOT

RAW = Path(os.environ.get("CARNATIC_MIR_RAW", ROOT / "data" / "raw"))
_URL = re.compile(r"^https?://", re.I)


def is_url(s) -> bool:
    return bool(_URL.match(str(s)))


def _safe(title: str) -> str:
    return re.sub(r"[^\w\-]+", "_", title).strip("_")[:80] or "track"


def read_list(path) -> list[tuple[str, str | None, str | None]]:
    """Parse a URL list file -> [(url, start, end)]; start/end are raw strings or None."""
    jobs = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if not is_url(parts[0]):
            raise ValueError(f"{path}:{n}: not a URL: {parts[0]}")
        if len(parts) not in (1, 2, 3):
            raise ValueError(f"{path}:{n}: expected 'URL [start [end]]', got: {line}")
        jobs.append((parts[0], *(parts[1:] + [None, None])[:2]))
    return jobs


_VID = re.compile(r"(?:youtu\.be/|[?&]v=|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})")


def local_copy(url: str, dest: Path = RAW) -> Path | None:
    """Already-downloaded file for a single-video URL (no network; works on servers YouTube blocks)."""
    m = _VID.search(url)
    if not m:
        return None
    hits = sorted(Path(dest).glob(f"*[[]{m.group(1)}[]].*"))
    return hits[0] if hits else None


def download(url: str, dest: Path = RAW, playlist: bool = False, log=print) -> list[Path]:
    """Download the best audio stream(s); returns the local file paths."""
    dest = Path(dest)
    if not playlist and (p := local_copy(url, dest)):
        log(f"already downloaded: {p.name}")
        return [p]
    import yt_dlp

    dest.mkdir(parents=True, exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "noplaylist": not playlist, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [e for e in info.get("entries", [info]) if e]

    paths = []
    for e in entries:
        vid = e.get("id", "")
        existing = sorted(dest.glob(f"*[[]{vid}[]].*"))
        if vid and existing:
            log(f"already downloaded: {existing[0].name}")
            paths.append(existing[0])
            continue
        stem = f"{_safe(e.get('title', 'track'))} [{vid}]"
        log(f"downloading: {e.get('title')} ({e.get('duration_string', '?')})")
        dl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
                   "format": "bestaudio/best", "outtmpl": str(dest / f"{stem}.%(ext)s")}
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            r = ydl.extract_info(e.get("webpage_url") or e.get("url") or url, download=True)
            p = Path(ydl.prepare_filename(r))
        log(f"saved: {p}")
        paths.append(p)
    return paths


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*")
    ap.add_argument("-f", "--file", help="text file with one URL per line (times in it are ignored here)")
    ap.add_argument("--dest", default=str(RAW), help="folder to save into (default data/raw)")
    ap.add_argument("--playlist", action="store_true", help="download every video of a playlist URL")
    a = ap.parse_args()
    urls = list(a.urls) + ([u for u, _, _ in read_list(a.file)] if a.file else [])
    if not urls:
        ap.error("give URLs or --file")
    failed = []
    for i, u in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {u}")
        try:
            download(u, Path(a.dest), playlist=a.playlist)
        except Exception as e:            # keep going; report at the end
            print(f"  FAILED: {e}")
            failed.append(u)
    print(f"done: {len(urls) - len(failed)} ok, {len(failed)} failed")
    for u in failed:
        print("  failed:", u)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
