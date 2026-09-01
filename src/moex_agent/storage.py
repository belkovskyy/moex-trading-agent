from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.logs_dir = data_dir / "logs"
        self.state_dir = data_dir / "state"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        path = self.logs_dir / f"{name}.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
            file.write("\n")

    def write_state(self, name: str, payload: dict[str, Any]) -> None:
        path = self.state_dir / f"{name}.json"
        tmp_path = path.with_suffix(".json.tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        # flush + fsync before the atomic rename so an abrupt container/host
        # crash can't leave a zero-length or torn state file on /data — the
        # bot must reload clean state across the 2-week autonomous run.
        with tmp_path.open("w", encoding="utf-8") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        tmp_path.replace(path)

    def read_state(self, name: str) -> dict[str, Any] | None:
        path = self.state_dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            # A corrupt/torn state file must NEVER crash the autonomous loop:
            # treat it as missing and let the caller fall back to its default.
            # This defends every read_state caller at once (incl. the main-loop
            # scheduler block, which is not wrapped in run_once's try/except).
            return None


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return str(value)

