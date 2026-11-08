"""Private study links, one per recruitment channel. Stored in study/data/links.json (not in git).

    python -m experiment.links add pilot        # prints ?k=<random key>
    python -m experiment.links list
"""
import json
import secrets
import sys

from .paths import DATA_DIR

LINKS = DATA_DIR / "links.json"


def load_links() -> dict[str, str]:
    return json.loads(LINKS.read_text()) if LINKS.exists() else {}


def main():
    args = sys.argv[1:]
    links = load_links()
    if args[:1] == ["add"] and len(args) == 2:
        k = secrets.token_urlsafe(9)
        links[k] = args[1]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        LINKS.write_text(json.dumps(links, indent=1))
        print(f"{args[1]}: /?k={k}")
    elif args[:1] == ["list"]:
        for k, label in links.items():
            print(f"{label}: /?k={k}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
