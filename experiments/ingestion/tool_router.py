"""Deterministic tool router used by ingestion agents."""

from __future__ import annotations

import csv
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from .constants import (
    ANALYTICAL_HINTS,
    GENERIC_BUCKET_LABELS,
    INSTRUCTIONAL_HINTS,
    NEGATIVE_TERMS,
    OPINION_HINTS,
    POSITIVE_TERMS,
    STOPWORDS,
    SUPPORTED_FILE_EXTENSIONS,
)
from .models import Fact, SourceDocument
from .multimedia_ingester import MultimediaIngester


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML to text extractor."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


class ToolRouter:
    """Tool belt used by specialized agents."""

    def __init__(self, multimedia_ingester: Optional[MultimediaIngester] = None) -> None:
        self.multimedia_ingester = multimedia_ingester or MultimediaIngester()

    def load_source(self, source: str, source_reliability: float) -> Tuple[Optional[SourceDocument], Optional[str]]:
        try:
            if source.startswith("http://") or source.startswith("https://"):
                return self._load_from_url(source, source_reliability), None

            path = Path(source)
            if path.is_file():
                return self._load_from_file(path, source_reliability), None

            extension = path.suffix.lower()
            if extension in {".png", ".jpg", ".jpeg", ".gif", ".mp3", ".wav", ".mp4"}:
                media_result = self.multimedia_ingester.ingest(source)
                return None, media_result.message

            return None, "unsupported source"
        except Exception as exc:  # defensive guard
            return None, str(exc)

    def expand_directory(self, directory: Path) -> List[Path]:
        results: List[Path] = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                results.append(file_path)
        return results

    def extract_records_from_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            if payload and all(isinstance(x, dict) for x in payload):
                return list(payload)
            return [{"content": " ".join(str(x) for x in payload)}]

        if isinstance(payload, dict):
            return [payload]

        text = str(payload).strip()
        if not text:
            return []

        words = re.findall(r"\w+", text)
        sentence_count = max(1, len(re.findall(r"[.!?]", text)))
        word_count = len(words)
        reading_time_min = max(1, int(math.ceil(word_count / 220)))
        title = text.splitlines()[0][:120] if text.splitlines() else text[:120]

        return [
            {
                "title": title,
                "content": text,
                "word_count": word_count,
                "reading_time": reading_time_min,
                "sentence_count": sentence_count,
                "content_type": "UnstructuredText",
            }
        ]

    def profile_schema(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        profile: Dict[str, Dict[str, Any]] = {}
        total = max(1, len(records))

        for record in records:
            for key, value in record.items():
                stats = profile.setdefault(
                    key,
                    {
                        "count": 0,
                        "types": set(),
                        "unique": set(),
                        "numeric_values": [],
                    },
                )
                if value is None or value == "":
                    continue
                normalized_value = value
                if isinstance(value, dict) and "value" in value:
                    normalized_value = value.get("value")

                stats["count"] += 1
                stats["types"].add(type(normalized_value).__name__)
                stats["unique"].add(str(normalized_value))
                if isinstance(normalized_value, (int, float)):
                    stats["numeric_values"].append(float(normalized_value))

        finalized: Dict[str, Dict[str, Any]] = {}
        for key, stats in profile.items():
            finalized[key] = {
                "coverage": stats["count"] / total,
                "type_count": len(stats["types"]),
                "types": sorted(stats["types"]),
                "cardinality": len(stats["unique"]),
                "numeric_values": list(stats["numeric_values"]),
            }
        return finalized

    def discretize_numeric(self, value: float, values: Sequence[float]) -> str:
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return "Medium"

        thresholds: List[float] = []
        for fraction in (0.2, 0.4, 0.6, 0.8):
            idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * fraction))))
            thresholds.append(sorted_values[idx])

        for idx, threshold in enumerate(thresholds):
            if value <= threshold:
                return GENERIC_BUCKET_LABELS[idx]
        return GENERIC_BUCKET_LABELS[-1]

    def compute_stv(
        self,
        record: Dict[str, Any],
        raw_property: str,
        value: Any,
        raw_value: Any,
        source_reliability: float,
    ) -> Tuple[float, float]:
        explicit_confidence = self._extract_explicit_score(record, raw_property, raw_value, "confidence")
        explicit_strength = self._extract_explicit_score(record, raw_property, raw_value, "strength")

        if explicit_confidence is not None or explicit_strength is not None:
            confidence = explicit_confidence if explicit_confidence is not None else 0.85
            strength = explicit_strength if explicit_strength is not None else source_reliability
            return round(self._clamp01(confidence), 3), round(self._clamp01(strength), 3)

        confidence = 0.9 if isinstance(value, str) else 0.85
        strength = min(1.0, max(0.5, source_reliability))

        if isinstance(value, (int, float)):
            normalized = min(1.0, max(0.0, float(value)))
            confidence = 0.55 + (normalized * 0.4)
            strength = min(1.0, strength + 0.05)

        return round(confidence, 3), round(strength, 3)

    def normalize_subject(self, raw_subject: Any, prefix: str) -> str:
        text = str(raw_subject or "")
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", text)
        if sanitized.startswith(prefix + "_"):
            return sanitized
        if sanitized.startswith("A_"):
            return sanitized
        if not sanitized:
            return f"{prefix}_00000"
        return f"{prefix}_{sanitized}"

    def normalize_predicate(self, raw_property: str) -> str:
        normalized = raw_property.lower().strip()
        normalized = re.sub(r"\s+", "-", normalized)
        normalized = normalized.replace("_", "-")
        normalized = re.sub(r"[^a-z0-9\-]", "-", normalized)
        normalized = re.sub(r"-+", "-", normalized).strip("-")
        return normalized or "unknown-property"

    def normalize_object(self, value: Any) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return f"{value:.4f}".rstrip("0").rstrip(".")

        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text[:280]

    def write_metta(self, output_path: str, facts: Sequence[Fact]) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [self.format_fact_line(fact) for fact in facts]
        output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return str(output)

    def format_fact_line(self, fact: Fact) -> str:
        escaped_object = fact.object_value.replace('"', '\\"')
        return f'(({fact.predicate} {fact.subject} "{escaped_object}") (STV {fact.confidence} {fact.strength}))'

    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        tokens = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
        if not tokens:
            return {"value": "Neutral", "confidence": 0.6, "strength": 0.7}

        pos = sum(1 for token in tokens if token in POSITIVE_TERMS)
        neg = sum(1 for token in tokens if token in NEGATIVE_TERMS)
        signal = abs(pos - neg)
        coverage = min(1.0, signal / max(1, len(tokens) * 0.08))
        confidence = round(0.6 + (0.35 * coverage), 3)

        if pos > neg:
            value = "Positive"
        elif neg > pos:
            value = "Negative"
        elif pos + neg > 0:
            value = "Mixed"
        else:
            value = "Neutral"

        strength = round(0.65 + (0.25 * coverage), 3)
        return {"value": value, "confidence": confidence, "strength": strength}

    def classify_content(self, title: str, text: str) -> Dict[str, Any]:
        combined = f"{title} {text}".lower()
        tokens = set(re.findall(r"[A-Za-z0-9']+", combined))

        if tokens.intersection(INSTRUCTIONAL_HINTS):
            value = "Instructional"
            confidence = 0.85
        elif tokens.intersection(ANALYTICAL_HINTS):
            value = "Analytical"
            confidence = 0.82
        elif tokens.intersection(OPINION_HINTS):
            value = "Opinionated"
            confidence = 0.8
        else:
            value = "Informational"
            confidence = 0.72

        return {"value": value, "confidence": confidence, "strength": 0.82}

    def semantic_parse(self, title: str, text: str, max_keywords: int = 5) -> Dict[str, Any]:
        tokens = re.findall(r"[A-Za-z0-9']+", f"{title} {text}".lower())
        filtered = [token for token in tokens if len(token) > 2 and token not in STOPWORDS]
        frequencies: Dict[str, int] = {}
        for token in filtered:
            frequencies[token] = frequencies.get(token, 0) + 1

        ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        keywords = [token for token, _ in ranked[:max_keywords]]
        return {
            "keywords": keywords,
            "confidence": 0.78 if keywords else 0.6,
            "strength": 0.8 if keywords else 0.65,
        }

    def derive_recommendation_signals(self, record: Dict[str, Any]) -> Dict[str, Any]:
        text = str(record.get("content", ""))
        title = str(record.get("title", ""))
        combined_len = len(re.findall(r"\w+", f"{title} {text}"))
        reading_time = record.get("reading_time")
        if not isinstance(reading_time, (int, float)):
            reading_time = max(1, int(math.ceil(combined_len / 220)))

        engagement_raw = record.get("engagement")
        engagement_signal = 0.5
        if isinstance(engagement_raw, (int, float)):
            engagement_signal = min(1.0, max(0.0, float(engagement_raw)))

        complexity = min(1.0, max(0.0, combined_len / 1200.0))
        novelty = min(1.0, 0.4 + (0.6 * min(1.0, len(set(re.findall(r"\w+", text.lower()))) / 180.0)))
        utility = min(1.0, 0.35 + (0.3 * min(1.0, float(reading_time) / 12.0)) + (0.35 * engagement_signal))

        return {
            "recommendation_novelty": round(novelty, 3),
            "recommendation_utility": round(utility, 3),
            "complexity_signal": round(complexity, 3),
        }

    def _load_from_url(self, source: str, source_reliability: float) -> SourceDocument:
        response = requests.get(source, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            payload = response.json()
            source_type = "json"
        elif "text/html" in content_type:
            extractor = _HTMLTextExtractor()
            extractor.feed(response.text)
            payload = extractor.text()
            source_type = "text"
        else:
            payload = response.text
            source_type = "text"

        return SourceDocument(
            source=source,
            source_type=source_type,
            payload=payload,
            source_reliability=source_reliability,
        )

    def _load_from_file(self, path: Path, source_reliability: float) -> SourceDocument:
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".json":
            payload: Any = json.loads(text)
            source_type = "json"
        elif suffix in {".jsonl", ".ndjson"}:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
            source_type = "json"
        elif suffix == ".csv":
            payload = list(csv.DictReader(text.splitlines()))
            source_type = "json"
        elif suffix in {".html", ".htm"}:
            extractor = _HTMLTextExtractor()
            extractor.feed(text)
            payload = extractor.text()
            source_type = "text"
        else:
            payload = text
            source_type = "text"

        return SourceDocument(
            source=str(path),
            source_type=source_type,
            payload=payload,
            source_reliability=source_reliability,
        )

    def _extract_explicit_score(
        self,
        record: Dict[str, Any],
        raw_property: str,
        raw_value: Any,
        score_name: str,
    ) -> Optional[float]:
        key_variants = [
            f"{raw_property}_{score_name}",
            f"{raw_property}-{score_name}",
            f"{raw_property}{score_name.capitalize()}",
        ]
        for key in key_variants:
            if key in record:
                try:
                    return float(record[key])
                except (TypeError, ValueError):
                    pass

        if isinstance(raw_value, dict) and score_name in raw_value:
            try:
                return float(raw_value[score_name])
            except (TypeError, ValueError):
                return None

        return None

    def _clamp01(self, value: float) -> float:
        return min(1.0, max(0.0, value))
