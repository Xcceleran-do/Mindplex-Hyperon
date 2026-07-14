from __future__ import annotations

import hashlib
from typing import Any

import requests


class PeTTaChainerError(RuntimeError):
    pass


class PeTTaChainerClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise TimeoutError("PeTTaChainer request timed out") from exc
        except requests.RequestException as exc:
            raise PeTTaChainerError(f"PeTTaChainer is unavailable: {exc}") from exc
        if not response.ok:
            try:
                detail = response.json().get("error", {}).get("message")
            except ValueError:
                detail = None
            raise PeTTaChainerError(
                detail or f"PeTTaChainer returned HTTP {response.status_code}"
            )
        return response.json() if response.content else None

    def health(self) -> dict[str, Any]:
        self._request("GET", "/v1/knowledge-bases", params={"limit": 1})
        return {"status": "ready"}

    def ensure_knowledge_base(self, name: str) -> str:
        offset = 0
        while True:
            page = self._request(
                "GET",
                "/v1/knowledge-bases",
                params={"limit": 500, "offset": offset},
            )["items"]
            for knowledge_base in page:
                if knowledge_base["name"] == name:
                    return knowledge_base["id"]
            if len(page) < 500:
                break
            offset += len(page)
        return self._request(
            "POST",
            "/v1/knowledge-bases",
            json={"name": name},
        )["id"]

    def add_statements(self, kb_id: str, sources: list[str]) -> dict[str, Any]:
        if not sources:
            return {"items": []}
        last_result: dict[str, Any] = {"items": []}
        for start in range(0, len(sources), 250):
            statements = [
                {
                    "source": source,
                    "idempotency_key": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                }
                for source in sources[start : start + 250]
            ]
            last_result = self._request(
                "POST",
                f"/v1/knowledge-bases/{kb_id}/statements/bulk",
                json={"statements": statements},
            )
        return last_result

    def backward(self, kb_id: str, query: str, steps: int) -> list[str]:
        if not query.startswith("(:"):
            query = f"(: $prf {query} $tv)"
        result = self._request(
            "POST",
            f"/v1/knowledge-bases/{kb_id}/reason/backward",
            json={"query": query, "steps": steps},
        )
        return result["results"]
