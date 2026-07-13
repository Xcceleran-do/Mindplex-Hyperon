from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from experiments.api.config import ASI_API_KEY, ASI_BASE_URL, ASI_MODEL, ASI_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

if not ASI_API_KEY:
    logger.warning("ASI_API_KEY environment variable is not set. AI features will fail.")


def call_asi_api(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Calls the ASI API with the given messages and tools."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ASI_API_KEY}"
    }
    payload = {
        "model": ASI_MODEL,
        "messages": messages,
        "temperature": 0.7
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        response = requests.post(ASI_BASE_URL, headers=headers, json=payload, timeout=ASI_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        if hasattr(exc, "response") and exc.response is not None:
            logger.warning("ASI API request failed: %s | response=%s", exc, exc.response.text)
        else:
            logger.warning("ASI API request failed: %s", exc)
        return {"error": "The language model service is unavailable."}
