import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]

POSITIVE_TERMS = {
    "good",
    "great",
    "best",
    "win",
    "wins",
    "improve",
    "growth",
    "benefit",
    "success",
}

NEGATIVE_TERMS = {
    "bad",
    "worst",
    "loss",
    "losses",
    "risk",
    "crisis",
    "decline",
    "fail",
    "fails",
}


def _safe_quote(value: str) -> str:
    return str(value).replace('"', '\\"').replace("\n", " ").strip()


def _normalize_category(category: str) -> str:
    category = (category or "unknown").strip().lower()
    category = re.sub(r"\s+", "-", category)
    return re.sub(r"[^a-z0-9\-_]", "", category) or "unknown"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _length_bucket(word_count: int) -> str:
    if word_count < 18:
        return "Short"
    if word_count < 45:
        return "Medium"
    return "Long"


def _reading_time_bucket(word_count: int, wpm: int = 220) -> str:
    minutes = max(1, math.ceil(word_count / max(1, wpm)))
    if minutes <= 1:
        return "Short"
    if minutes <= 3:
        return "Medium"
    return "Long"


def _tone_bucket(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    if "?" in (title or "") or any(k in text for k in ["how to", "guide", "tips", "why "]):
        return "Instructional"
    if any(k in text for k in ["exclusive", "shocking", "amazing", "must", "you won't believe"]):
        return "Casual"
    return "Formal"


def _expertise_bucket(title: str, abstract: str) -> str:
    tokens = _tokenize(f"{title} {abstract}")
    if not tokens:
        return "Beginner"
    long_ratio = sum(1 for token in tokens if len(token) >= 8) / len(tokens)
    if long_ratio < 0.18:
        return "Beginner"
    if long_ratio < 0.30:
        return "Intermediate"
    return "Advanced"


def _primary_goal_bucket(category: str) -> str:
    category = (category or "").lower()
    if category in {"sports", "video", "tv", "entertainment", "lifestyle"}:
        return "Entertain"
    if category in {"opinion"}:
        return "Persuade"
    return "Inform"


def _content_type_bucket(category: str) -> str:
    return "Opinion" if (category or "").lower() == "opinion" else "News"


def _sentiment_bucket(title: str, abstract: str) -> str:
    tokens = set(_tokenize(f"{title} {abstract}"))
    pos = len(tokens.intersection(POSITIVE_TERMS))
    neg = len(tokens.intersection(NEGATIVE_TERMS))
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Mixed"


def _engagement_bucket(ctr: float) -> str:
    if ctr < 0.05:
        return "Low"
    if ctr <= 0.15:
        return "Medium"
    return "High"


def parse_impression_token(token: str) -> Tuple[Optional[str], Optional[int]]:
    if not token:
        return None, None
    parts = token.rsplit("-", 1)
    if len(parts) != 2:
        return parts[0], None
    news_id, label = parts
    try:
        return news_id, int(label)
    except ValueError:
        return news_id, None


def _iter_dataset_files(mind_dir: str) -> Iterable[Tuple[str, str]]:
    split_dirs = ["train", "valid", "test"]
    found_split = False
    for split in split_dirs:
        split_path = os.path.join(mind_dir, split)
        news_path = os.path.join(split_path, "news.tsv")
        behaviors_path = os.path.join(split_path, "behaviors.tsv")
        if os.path.exists(news_path):
            found_split = True
            yield news_path, behaviors_path if os.path.exists(behaviors_path) else ""
    if not found_split:
        news_path = os.path.join(mind_dir, "news.tsv")
        behaviors_path = os.path.join(mind_dir, "behaviors.tsv")
        if not os.path.exists(news_path):
            raise FileNotFoundError(f"Cannot find news.tsv under: {mind_dir}")
        yield news_path, behaviors_path if os.path.exists(behaviors_path) else ""


def _read_news(news_path: str) -> Dict[str, Dict[str, str]]:
    news_map: Dict[str, Dict[str, str]] = {}
    with open(news_path, "r", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj, delimiter="\t")
        for row in reader:
            if not row:
                continue
            padded = row + [""] * max(0, len(NEWS_COLUMNS) - len(row))
            record = dict(zip(NEWS_COLUMNS, padded))
            news_id = record.get("news_id", "").strip()
            if news_id:
                news_map[news_id] = record
    return news_map


def _update_behavior_stats(
    behaviors_path: str,
    impressions_by_news: Dict[str, int],
    clicks_by_news: Dict[str, int],
    timestamps: List[datetime],
) -> None:
    if not behaviors_path or not os.path.exists(behaviors_path):
        return
    with open(behaviors_path, "r", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj, delimiter="\t")
        for row in reader:
            if not row:
                continue
            padded = row + [""] * max(0, len(BEHAVIOR_COLUMNS) - len(row))
            record = dict(zip(BEHAVIOR_COLUMNS, padded))

            timestamp_raw = (record.get("time") or "").strip()
            if timestamp_raw:
                for dt_format in ["%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        timestamps.append(datetime.strptime(timestamp_raw, dt_format).replace(tzinfo=timezone.utc))
                        break
                    except ValueError:
                        continue

            for token in (record.get("impressions") or "").split():
                news_id, label = parse_impression_token(token)
                if not news_id:
                    continue
                impressions_by_news[news_id] += 1
                if label == 1:
                    clicks_by_news[news_id] += 1


def _date_period_bucket(reference_date: Optional[datetime]) -> str:
    if not reference_date:
        return "Archived"
    age_days = (datetime.now(timezone.utc) - reference_date).days
    if age_days <= 90:
        return "Recent"
    if age_days <= 365:
        return "Last_Year"
    return "Archived"


def _midpoint_datetime(values: List[datetime]) -> Optional[datetime]:
    if not values:
        return None
    sorted_values = sorted(values)
    return sorted_values[len(sorted_values) // 2]


def _emit_fact(lines: List[str], prop: str, document_id: str, value: str) -> None:
    safe_value = _safe_quote(value)
    if safe_value:
        lines.append(f'({prop} {document_id} "{safe_value}")')


def convert_mind_to_metta(
    mind_dir: str,
    output_metta_path: str,
    report_dir: str,
    min_documents: int = 1000,
    max_documents: Optional[int] = None,
) -> Dict[str, object]:
    all_news: Dict[str, Dict[str, str]] = {}
    impressions_by_news: Dict[str, int] = defaultdict(int)
    clicks_by_news: Dict[str, int] = defaultdict(int)
    timestamps: List[datetime] = []
    loaded_files: List[Dict[str, object]] = []

    for news_path, behaviors_path in _iter_dataset_files(mind_dir):
        split_news = _read_news(news_path)
        all_news.update(split_news)
        loaded_files.append(
            {
                "news_path": news_path,
                "behaviors_path": behaviors_path,
                "news_records": len(split_news),
                "has_behaviors": bool(behaviors_path),
            }
        )
        _update_behavior_stats(behaviors_path, impressions_by_news, clicks_by_news, timestamps)

    if not all_news:
        raise RuntimeError("No news records were loaded from MIND.")

    if len(all_news) < min_documents:
        file_details = "; ".join(
            [
                f"{entry['news_path']} (news={entry['news_records']}, behaviors={entry['has_behaviors']})"
                for entry in loaded_files
            ]
        )
        raise RuntimeError(
            "Loaded too few documents for a meaningful benchmark "
            f"({len(all_news)} < min_documents={min_documents}). "
            "This usually means --mind-dir points to a sample folder. "
            f"Loaded files: {file_details}"
        )

    impression_counts = [impressions_by_news.get(news_id, 0) for news_id in all_news]
    sorted_impressions = sorted(impression_counts)
    top_10_cutoff = sorted_impressions[max(0, int(len(sorted_impressions) * 0.9) - 1)] if sorted_impressions else 0
    reference_date = _midpoint_datetime(timestamps)

    selected_news_ids = list(all_news.keys())
    if isinstance(max_documents, int) and max_documents > 0 and len(selected_news_ids) > max_documents:
        selected_news_ids = sorted(
            selected_news_ids,
            key=lambda news_id: (-impressions_by_news.get(news_id, 0), news_id),
        )[:max_documents]

    metta_lines: List[str] = []
    distribution = {
        "engagement": Counter(),
        "popularity": Counter(),
        "category": Counter(),
        "tone": Counter(),
    }

    ctr_values: List[float] = []

    for news_id in selected_news_ids:
        record = all_news[news_id]
        document_id = f"A_{news_id}"
        category = _normalize_category(record.get("category", "unknown"))
        title = record.get("title", "")
        abstract = record.get("abstract", "")
        words = _tokenize(f"{title} {abstract}")
        word_count = len(words)

        impressions = impressions_by_news.get(news_id, 0)
        clicks = clicks_by_news.get(news_id, 0)
        ctr = (clicks / impressions) if impressions > 0 else 0.0
        ctr_values.append(ctr)

        length = _length_bucket(word_count)
        reading_time = _reading_time_bucket(word_count)
        tone = _tone_bucket(title, abstract)
        audience_expertise = _expertise_bucket(title, abstract)
        content_type = _content_type_bucket(category)
        date_period = _date_period_bucket(reference_date)
        primary_goal = _primary_goal_bucket(category)
        popularity = "Top_10" if impressions >= top_10_cutoff and top_10_cutoff > 0 else "Other"
        engagement = _engagement_bucket(ctr)
        audience_sentiment = _sentiment_bucket(title, abstract)

        _emit_fact(metta_lines, "length", document_id, length)
        _emit_fact(metta_lines, "reading-time", document_id, reading_time)
        _emit_fact(metta_lines, "tone", document_id, tone)
        _emit_fact(metta_lines, "audience-expertise", document_id, audience_expertise)
        _emit_fact(metta_lines, "content-type", document_id, content_type)
        _emit_fact(metta_lines, "date-period", document_id, date_period)
        _emit_fact(metta_lines, "primary-goal", document_id, primary_goal)
        _emit_fact(metta_lines, "category", document_id, category)
        _emit_fact(metta_lines, "popularity", document_id, popularity)
        _emit_fact(metta_lines, "engagement", document_id, engagement)
        _emit_fact(metta_lines, "audience-sentiment", document_id, audience_sentiment)
        _emit_fact(metta_lines, "authored-by", document_id, "MIND")
        _emit_fact(metta_lines, "title", document_id, title)

        distribution["engagement"][engagement] += 1
        distribution["popularity"][popularity] += 1
        distribution["category"][category] += 1
        distribution["tone"][tone] += 1

    os.makedirs(os.path.dirname(output_metta_path), exist_ok=True)
    with open(output_metta_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(metta_lines))

    os.makedirs(report_dir, exist_ok=True)
    report_json_path = os.path.join(report_dir, "mind_preliminary_results.json")
    report_md_path = os.path.join(report_dir, "mind_preliminary_results.md")

    stats = {
        "dataset": "MIND",
        "document_count": len(selected_news_ids),
        "source_document_count": len(all_news),
        "with_impressions": sum(1 for news_id in selected_news_ids if impressions_by_news.get(news_id, 0) > 0),
        "total_impressions": sum(impressions_by_news.get(news_id, 0) for news_id in selected_news_ids),
        "total_clicks": int(sum(clicks_by_news.values())),
        "avg_ctr": (sum(ctr_values) / len(ctr_values)) if ctr_values else 0.0,
        "top_10_threshold_impressions": int(top_10_cutoff),
        "distribution": {
            key: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
            for key, counter in distribution.items()
        },
        "output_metta_path": output_metta_path,
        "loaded_files": loaded_files,
    }

    with open(report_json_path, "w", encoding="utf-8") as file_obj:
        json.dump(stats, file_obj, indent=2)

    top_categories = list(stats["distribution"]["category"].items())[:8]
    top_tones = list(stats["distribution"]["tone"].items())[:5]
    md_lines = [
        "# Preliminary Results on MIND",
        "",
        "## Executive Snapshot",
        f"- Documents processed: **{stats['document_count']}**",
        f"- Source documents available: **{stats['source_document_count']}**",
        f"- Documents with impressions: **{stats['with_impressions']}**",
        f"- Total impressions: **{stats['total_impressions']}**",
        f"- Total clicks: **{stats['total_clicks']}**",
        f"- Average CTR: **{stats['avg_ctr']:.4f}**",
        f"- Top-10 popularity cutoff (impressions): **{stats['top_10_threshold_impressions']}**",
        "",
        "## Label Distribution",
        "### Engagement",
    ]

    for key, value in stats["distribution"]["engagement"].items():
        md_lines.append(f"- {key}: **{value}**")

    md_lines.extend(["", "### Popularity"])
    for key, value in stats["distribution"]["popularity"].items():
        md_lines.append(f"- {key}: **{value}**")

    md_lines.extend(["", "## Top Categories"])
    for key, value in top_categories:
        md_lines.append(f"- {key}: **{value}**")

    md_lines.extend(["", "## Top Tone Buckets"])
    for key, value in top_tones:
        md_lines.append(f"- {key}: **{value}**")

    md_lines.extend(
        [
            "",
            "## Artifacts",
            f"- MeTTa facts: `{output_metta_path}`",
            f"- JSON stats: `{report_json_path}`",
            "",
            "## Caveats",
            "- Labels are proxy labels from click logs (CTR/impression based), not editorial ground-truth.",
            "- Heuristic fields (tone/sentiment/expertise) are rule-based and intended for preliminary benchmarking.",
        ]
    )

    with open(report_md_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(md_lines))

    stats["report_json_path"] = report_json_path
    stats["report_md_path"] = report_md_path
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MIND dataset to Mindplex-compatible MeTTa facts.")
    parser.add_argument("--mind-dir", required=True, help="Path to MIND root folder or split folder containing TSV files.")
    parser.add_argument(
        "--output-metta",
        default="experiments/atomspace_visualizer/public/data.metta",
        help="Output path for generated MeTTa facts.",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/reports",
        help="Directory where summary report files are written.",
    )
    parser.add_argument(
        "--min-documents",
        type=int,
        default=1000,
        help="Fail fast if fewer than this many documents are loaded (guards against wrong/small folders).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on number of documents exported (top by impressions).",
    )
    args = parser.parse_args()

    stats = convert_mind_to_metta(
        mind_dir=args.mind_dir,
        output_metta_path=args.output_metta,
        report_dir=args.report_dir,
        min_documents=args.min_documents,
        max_documents=args.max_documents,
    )

    print("MIND conversion finished.")
    print(f"- Documents processed: {stats['document_count']}")
    print(f"- MeTTa output: {stats['output_metta_path']}")
    print(f"- Report (markdown): {stats['report_md_path']}")


if __name__ == "__main__":
    main()
