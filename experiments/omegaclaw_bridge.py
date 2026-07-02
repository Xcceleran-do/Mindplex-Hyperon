from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_QUEUE_DIR = Path(os.environ.get("OMEGACLAW_MINDPLEX_QUEUE_DIR", "/tmp/omegaclaw-mindplex"))
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("OMEGACLAW_MINDPLEX_RESPONSE_TIMEOUT", "120"))
DEFAULT_POLL_SECONDS = float(os.environ.get("OMEGACLAW_MINDPLEX_POLL_INTERVAL", "0.25"))
FORWARD_HISTORY = os.environ.get("OMEGACLAW_MINDPLEX_FORWARD_HISTORY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class OmegaClawBridgeTimeout(TimeoutError):
    pass


def is_omegaclaw_chat_enabled() -> bool:
    backend = os.environ.get("MINDPLEX_CHAT_BACKEND", "").strip().lower()
    enabled = os.environ.get("OMEGACLAW_CHAT_ENABLED", "").strip().lower()
    return backend == "omegaclaw" or enabled in {"1", "true", "yes", "on"}


def _queue_dirs(queue_dir: Path = DEFAULT_QUEUE_DIR) -> tuple[Path, Path, Path]:
    return queue_dir / "requests", queue_dir / "processing", queue_dir / "responses"


def _ensure_dirs(queue_dir: Path = DEFAULT_QUEUE_DIR) -> None:
    for path in _queue_dirs(queue_dir):
        path.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dirs(path.parent.parent)
    tmp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def send_chat_to_omegaclaw(
    message: str,
    *,
    session_id: str = "default",
    history: list[dict[str, Any]] | None = None,
    timeout_seconds: float | None = None,
    queue_dir: str | Path | None = None,
) -> dict[str, Any]:
    queue_root = Path(queue_dir) if queue_dir else DEFAULT_QUEUE_DIR
    requests_dir, _, responses_dir = _queue_dirs(queue_root)
    _ensure_dirs(queue_root)

    request_id = uuid.uuid4().hex
    payload = {
        "id": request_id,
        "session_id": session_id or "default",
        "message": message,
        "history": (history or []) if FORWARD_HISTORY else [],
        "created_at": time.time(),
    }
    _write_json_atomic(requests_dir / f"{request_id}.json", payload)

    timeout = DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    deadline = time.monotonic() + timeout
    response_path = responses_dir / f"{request_id}.json"

    while time.monotonic() < deadline:
        if response_path.exists():
            response = _read_json(response_path)
            try:
                response_path.unlink()
            except OSError:
                pass
            response.setdefault("id", request_id)
            response.setdefault("session_id", session_id or "default")
            response.setdefault("backend", "omegaclaw")
            return response
        time.sleep(DEFAULT_POLL_SECONDS)

    raise OmegaClawBridgeTimeout(
        f"Timed out waiting {timeout:g}s for OmegaClaw response. "
        "Start OmegaClaw with commchannel=mindplex and the same OMEGACLAW_MINDPLEX_QUEUE_DIR."
    )
