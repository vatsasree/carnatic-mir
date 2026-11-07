"""Split files bigger than GitHub's 100 MB limit into 90 MB pieces, and join them back.

    python scripts/shard.py split models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt
    python scripts/shard.py join            # rebuild every '*.shard000' under the repo, checking sha256

Pieces are '<file>.shard000', '.shard001', ... plus '<file>.sha256'. Git tracks the pieces, never the original.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIECE = 90 * 1024 * 1024


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split(p: Path):
    for old in p.parent.glob(p.name + ".shard*"):
        old.unlink()
    n = 0
    with open(p, "rb") as f:
        while b := f.read(PIECE):
            (p.parent / f"{p.name}.shard{n:03d}").write_bytes(b)
            n += 1
    (p.parent / f"{p.name}.sha256").write_text(sha256(p) + "\n")
    print(f"{p} -> {n} pieces")


def join(root: Path = ROOT):
    for first in sorted(root.rglob("*.shard000")):
        dst = first.with_name(first.name[:-len(".shard000")])
        want = (dst.parent / (dst.name + ".sha256")).read_text().split()[0]
        if dst.exists() and sha256(dst) == want:
            print(f"ok      {dst}")
            continue
        pieces = sorted(dst.parent.glob(dst.name + ".shard[0-9][0-9][0-9]"))
        with open(dst, "wb") as out:
            for piece in pieces:
                out.write(piece.read_bytes())
        if sha256(dst) != want:
            dst.unlink()
            sys.exit(f"checksum mismatch for {dst}: a piece is missing or corrupt")
        print(f"joined  {dst} ({len(pieces)} pieces)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "split":
        for f in sys.argv[2:]:
            split(Path(f))
    elif len(sys.argv) == 2 and sys.argv[1] == "join":
        join()
    else:
        sys.exit(__doc__)
