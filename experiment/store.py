"""SQLite storage. One row per participant, one row per saved record (rating, presses, check, ...).

Records are keyed (pid, kind, item) and upserted, so a client retry never creates duplicates.
"""
import json
import sqlite3
import threading
import time

from .paths import DATA_DIR, DB

_lock = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
  pid TEXT PRIMARY KEY, created REAL, link TEXT, age_group TEXT, set_id TEXT,
  assignment TEXT, profile TEXT, status TEXT, completion_code TEXT, finished REAL, user_agent TEXT);
CREATE TABLE IF NOT EXISTS records (
  pid TEXT, kind TEXT, item TEXT, payload TEXT, client_ts REAL, server_ts REAL,
  PRIMARY KEY (pid, kind, item));
"""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, check_same_thread=False, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    con.row_factory = sqlite3.Row
    return con


_con = None


def db() -> sqlite3.Connection:
    global _con
    if _con is None:
        _con = connect()
    return _con


def counts(age_group: str | None = None) -> dict[str, int]:
    """Participants per set (in one age group, or overall when None), ignoring headphone-check exclusions."""
    where, args = ("AND age_group=?", (age_group,)) if age_group else ("", ())
    rows = db().execute(f"SELECT set_id, COUNT(*) n FROM participants WHERE status!='excluded' {where} "
                        "GROUP BY set_id", args).fetchall()
    return {r["set_id"]: r["n"] for r in rows}


def add_participant(**p):
    with _lock:
        db().execute("INSERT INTO participants (pid, created, link, age_group, set_id, assignment, profile, status, "
                     "user_agent) VALUES (?,?,?,?,?,?,?,?,?)",
                     (p["pid"], time.time(), p["link"], p["age_group"], p["set_id"], json.dumps(p["assignment"]),
                      json.dumps(p["profile"]), "started", p["user_agent"]))


def get_participant(pid: str):
    r = db().execute("SELECT * FROM participants WHERE pid=?", (pid,)).fetchone()
    return dict(r) if r else None


def set_status(pid: str, status: str, code: str | None = None):
    with _lock:
        if code:
            db().execute("UPDATE participants SET status=?, completion_code=?, finished=? WHERE pid=?",
                         (status, code, time.time(), pid))
        else:
            db().execute("UPDATE participants SET status=? WHERE pid=?", (status, pid))


def save(pid: str, kind: str, item: str, payload, client_ts: float | None):
    with _lock:
        db().execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)",
                     (pid, kind, item, json.dumps(payload), client_ts, time.time()))


def saved_keys(pid: str) -> list[str]:
    return [f"{r['kind']}:{r['item']}" for r in
            db().execute("SELECT kind, item FROM records WHERE pid=?", (pid,)).fetchall()]
