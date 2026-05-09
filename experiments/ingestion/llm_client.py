import json
import os
import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import requests


ASI_BASE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-mini"


class LLMClient:
    """Small JSON-first LLM wrapper used by ingestion planning and extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 45,
        temperature: float = 0.2,
        transport: Optional[Callable[..., Any]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url or ASI_BASE_URL
        self.model = model or ASI_MODEL
        self.timeout = timeout
        self.temperature = temperature
        self.transport = transport or requests.post

    @classmethod
    def from_env(cls, api_key: Optional[str] = None) -> "LLMClient":
        return cls(
            api_key=api_key or os.getenv("ASI_API_KEY"),
            base_url=os.getenv("ASI_BASE_URL", ASI_BASE_URL),
            model=os.getenv("ASI_MODEL", ASI_MODEL),
            timeout=int(os.getenv("INGESTION_LLM_TIMEOUT", "45")),
            temperature=float(os.getenv("INGESTION_LLM_TEMPERATURE", "0.2")),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: Iterable[Mapping[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError("ASI_API_KEY is required for LLM-backed ingestion.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        response = self.transport(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM response did not contain message content: {data}") from exc

    def complete_json(self, messages: Iterable[Mapping[str, str]]) -> Dict[str, Any]:
        text = self.complete(messages)
        return extract_json_object(text)


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object.")

    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM JSON response must be an object.")
    return value
