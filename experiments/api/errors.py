from __future__ import annotations

import uuid
from typing import Any

from flask import jsonify


def error_payload(code: str, message: str, *, error_id: str | None = None) -> dict[str, Any]:
    error: dict[str, str] = {"code": code, "message": message}
    if error_id:
        error["id"] = error_id
    return {"status": "error", "message": message, "error": error}


def public_error(code: str, message: str, status: int):
    return jsonify(error_payload(code, message)), status


def unexpected_error(logger: Any, context: str, message: str, *, code: str = "internal_error", status: int = 500):
    error_id = uuid.uuid4().hex[:12]
    logger.exception("%s [error_id=%s]", context, error_id)
    return jsonify(error_payload(code, message, error_id=error_id)), status
