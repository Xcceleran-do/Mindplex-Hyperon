from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests


PREDICATE_RE = re.compile(r"\(([A-Za-z_][A-Za-z0-9_-]{0,127})(?=[\s)])")
RESERVED_PREDICATES = {"Implication", "Premises", "Conclusions", "STV"}


class NL2PLNError(RuntimeError):
    pass


class NL2PLNClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        namespace: str = "mindplex",
        timeout_seconds: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("NL2PLN_BASE_URL must be an absolute HTTP(S) URL")
        if not api_key:
            raise ValueError("NL2PLN_API_KEY is required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", namespace):
            raise ValueError("NL2PLN_NAMESPACE is invalid")
        if timeout_seconds <= 0:
            raise ValueError("NL2PLN_TIMEOUT_SECONDS must be positive")
        self.base_url = base_url.rstrip("/")
        self.namespace = namespace
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )

    def translate_query(self, question: str, statements: list[str]) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        context_statements = list(dict.fromkeys(item.strip() for item in statements if item.strip()))[:500]
        predicates = sorted(
            {
                predicate
                for statement in context_statements
                for predicate in PREDICATE_RE.findall(statement)
                if predicate not in RESERVED_PREDICATES
            }
        )
        if not predicates:
            raise NL2PLNError("the active knowledge base has no queryable predicates")

        payload = {
            "namespace": self.namespace,
            "queries": [question],
            "context": {
                "statements": context_statements,
                "predicates": predicates,
            },
        }
        try:
            response = self.session.post(
                f"{self.base_url}/v1/translate",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise NL2PLNError("NL2PLN request timed out") from exc
        except requests.RequestException as exc:
            raise NL2PLNError("NL2PLN is unavailable") from exc

        if not response.ok:
            try:
                code = response.json().get("error", {}).get("code")
            except (ValueError, AttributeError):
                code = None
            suffix = f": {code}" if code else ""
            raise NL2PLNError(f"NL2PLN rejected the query (HTTP {response.status_code}){suffix}")

        try:
            body: Any = response.json()
            queries = body["queries"]
            translated = next(item for item in queries if item.get("source_query_index") == 0)
            source = translated["source"].strip()
        except (ValueError, TypeError, KeyError, StopIteration, AttributeError) as exc:
            raise NL2PLNError("NL2PLN returned an invalid translation response") from exc
        if not source.startswith("(:"):
            raise NL2PLNError("NL2PLN returned an invalid PeTTaChainer query")
        return source
