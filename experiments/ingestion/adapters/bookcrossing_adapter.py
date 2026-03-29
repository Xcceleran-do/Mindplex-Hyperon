"""Adapter for Kaggle Book Recommendation (Book-Crossing) dataset.

Expected files from dataset archive:
- BX-Books.csv
- BX-Users.csv
- BX-Book-Ratings.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _read_semicolon_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="latin-1", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)

        delimiter = ";"
        quotechar = '"'
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t,")
            delimiter = dialect.delimiter
            quotechar = dialect.quotechar or '"'
        except csv.Error:
            pass

        reader = csv.DictReader(handle, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            normalized: Dict[str, str] = {}
            for key, value in row.items():
                if key is None:
                    continue
                safe_key = str(key).strip().strip('"')
                safe_value = ""
                if isinstance(value, list):
                    safe_value = " ".join(str(v) for v in value if v is not None)
                elif value is not None:
                    safe_value = str(value)
                normalized[safe_key] = safe_value.strip().strip('"')
            if normalized:
                yield normalized


def _bucket_year(year_text: str) -> str:
    try:
        year = int(year_text)
    except ValueError:
        return "Unknown"

    if year < 1980:
        return "Classic"
    if year < 2000:
        return "Modern"
    if year < 2015:
        return "Recent"
    return "Contemporary"


def _bucket_age(age_text: str) -> str:
    try:
        age = int(float(age_text))
    except ValueError:
        return "Unknown"

    if age < 18:
        return "Teen"
    if age < 30:
        return "Young_Adult"
    if age < 50:
        return "Adult"
    return "Senior"


def _rating_bucket(value: float) -> str:
    if value <= 3:
        return "Low"
    if value <= 7:
        return "Medium"
    return "High"


def build_bookcrossing_records(dataset_dir: str, limit_books: Optional[int] = 10000) -> List[Dict[str, object]]:
    root = Path(dataset_dir)
    books_path = _pick_existing(root, ["BX-Books.csv", "Books.csv"])
    users_path = _pick_existing(root, ["BX-Users.csv", "Users.csv"])
    ratings_path = _pick_existing(root, ["BX-Book-Ratings.csv", "Ratings.csv"])

    if books_path is None or users_path is None or ratings_path is None:
        raise FileNotFoundError(
            "Dataset directory must contain either BX-Books/BX-Users/BX-Book-Ratings or Books/Users/Ratings CSV files"
        )

    users: Dict[str, Dict[str, str]] = {}
    for row in _read_semicolon_csv(users_path):
        user_id = row.get("User-ID", "")
        if user_id:
            users[user_id] = row

    rating_sums = defaultdict(float)
    rating_counts = defaultdict(int)
    rating_zero_counts = defaultdict(int)
    unique_raters = defaultdict(set)

    for row in _read_semicolon_csv(ratings_path):
        isbn = row.get("ISBN", "")
        user_id = row.get("User-ID", "")
        raw_rating = row.get("Book-Rating", "")
        if not isbn:
            continue

        try:
            rating = float(raw_rating)
        except ValueError:
            continue

        if rating <= 0:
            rating_zero_counts[isbn] += 1
            continue

        rating_sums[isbn] += rating
        rating_counts[isbn] += 1
        if user_id:
            unique_raters[isbn].add(user_id)

    records: List[Dict[str, object]] = []
    for row in _read_semicolon_csv(books_path):
        isbn = row.get("ISBN", "")
        if not isbn:
            continue

        ratings_n = rating_counts.get(isbn, 0)
        avg_rating = (rating_sums[isbn] / ratings_n) if ratings_n else 0.0

        author = row.get("Book-Author", "Unknown") or "Unknown"
        title = row.get("Book-Title", "Untitled") or "Untitled"
        publisher = row.get("Publisher", "Unknown") or "Unknown"

        sample_user_age_bucket = "Unknown"
        raters = list(unique_raters.get(isbn, set()))
        if raters:
            sample_user = users.get(raters[0], {})
            sample_user_age_bucket = _bucket_age(sample_user.get("Age", ""))

        record: Dict[str, object] = {
            "id": isbn,
            "title": title,
            "author": author,
            "publisher": publisher,
            "publish_year_bucket": _bucket_year(row.get("Year-Of-Publication", "")),
            "avg_rating": round(avg_rating, 3),
            "rating_count": ratings_n,
            "implicit_interest_count": rating_zero_counts.get(isbn, 0),
            "engagement": min(1.0, ratings_n / 50.0),
            "audience_age_bucket": sample_user_age_bucket,
            "content_type": "Book",
        }
        records.append(record)

        if isinstance(limit_books, int) and limit_books > 0 and len(records) >= limit_books:
            break

    return records


def _pick_existing(root: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def write_records_jsonl(records: List[Dict[str, object]], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return str(path)
