"""Where the study files live. Everything under study/snippets and study/data stays out of git."""
import os
from pathlib import Path

from carnatic_mir.config import OUTPUTS, ROOT

STUDY = Path(os.environ.get("STUDY_DIR", ROOT / "study"))
CONFIG = STUDY / "config.json"
PAIRS = STUDY / "pairs.csv"
SNIPPETS_CSV = STUDY / "snippets.csv"
SNIPPET_DIR = STUDY / "snippets"          # audio served to participants (copyrighted: never commit)
DATA_DIR = Path(os.environ.get("STUDY_DATA", STUDY / "data"))   # responses (personal data: never commit)
DB = DATA_DIR / "responses.sqlite"
TRACKS = OUTPUTS
