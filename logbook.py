import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_DIR, LOG_MAX_ENTRIES

LOG_FILE = LOG_DIR / "requests.jsonl"


def _now_iso() -> str:
    """Return the current time as an ISO-8601 string in local time."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_entries() -> list[dict]:
    """Read all entries from the logbook file (oldest first)."""
    if not LOG_FILE.exists():
        return []
    entries = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip a malformed/corrupted line rather than losing the whole file.
            continue
    return entries


def _write_entries(entries: list[dict]) -> None:
    """Atomically write entries to the logbook file, keeping the newest last."""
    content = "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
    if content:
        content += "\n"
    LOG_FILE.write_text(content, encoding="utf-8")


def start_request(tool_name: str, params: dict) -> str:
    """
    Record a request that is about to run, before generation starts.

    Writes an entry with status "pending". Returns the entry id so the caller
    can later pass it to finish_request() to update the same entry.

    Args:
        tool_name: Name of the tool being invoked.
        params: Keyword arguments the tool was called with.
    """
    entry_id = uuid.uuid4().hex
    entry = {
        "id": entry_id,
        "time": _now_iso(),
        "tool": tool_name,
        "params": params,
        "status": "pending",
    }

    entries = _read_entries()
    entries.append(entry)
    if len(entries) > LOG_MAX_ENTRIES:
        entries = entries[-LOG_MAX_ENTRIES:]
    _write_entries(entries)
    return entry_id


def finish_request(entry_id: str, status: str, output_paths: list[str] | None = None) -> None:
    """
    Update an existing entry after generation completes (or fails).

    The entry is located by its id and its status/end-time/output paths are
    set, keeping the file trimmed to LOG_MAX_ENTRIES entries.

    Args:
        entry_id: Id returned by start_request().
        status: "success" or "error".
        output_paths: Absolute paths of the files produced by the request.
    """
    entries = _read_entries()
    for entry in entries:
        if entry.get("id") == entry_id:
            entry["status"] = status
            entry["end_time"] = _now_iso()
            if output_paths is not None:
                entry["output_paths"] = output_paths
            break
    if len(entries) > LOG_MAX_ENTRIES:
        entries = entries[-LOG_MAX_ENTRIES:]
    _write_entries(entries)
