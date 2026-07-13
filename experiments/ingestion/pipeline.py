# experiments/ingestion/pipeline.py
import os
import sys
from dotenv import load_dotenv

# Add workspace root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.ingestion.fetcher import MindplexFetcher, build_fetcher
from experiments.ingestion.analyzer import ArticleAnalyzer
from experiments.ingestion.converter import JsonToMetta
from experiments.ingestion.config import DEFAULT_USERNAME
from experiments.ingestion.utils import excluded_predicates

DEFAULT_OUTPUT_PATH = "experiments/atomspace_visualizer/public/data.metta"
DEFAULT_LIMIT = 50


def resolve_output_path(output_path=None):
    return output_path or os.getenv("METTA_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)


def run_ingestion(username=None, source_name="mindplex", limit=DEFAULT_LIMIT, output_path=None, source_config=None):
    load_dotenv()
    output_path = resolve_output_path(output_path)
    if source_name == "mindplex" and not username:
        username = os.getenv("MINDPLEX_USERNAME", DEFAULT_USERNAME)
        
    print(f"Starting ingestion pipeline for source: {source_name}")

    print("1. Fetching source records...")
    if source_name == "mindplex":
        fetcher = MindplexFetcher(username=username)
    else:
        fetcher = build_fetcher(source_name=source_name, source_config=source_config)
    records = fetcher.fetch_all(limit=limit)
    print(f"   Fetched {len(records)} records.")

    if not records:
        print("No articles found. Exiting.")
        return {"status": "error", "message": "No articles found"}

    print("2. Normalizing source metrics...")
    has_views_metric = any("views" in record for record in records)
    if has_views_metric:
        for record in records:
            try:
                record["views"] = int(record.get("views", 0))
            except (TypeError, ValueError):
                record["views"] = 0
        records.sort(key=lambda item: item.get("views", 0), reverse=True)

    rank_map = {
        record.get("id", record.get("uuid", index)): index + 1
        for index, record in enumerate(records)
    }

    print("3. Planning extraction and enriching records...")
    api_key = os.getenv("ASI_API_KEY")
    analyzer = ArticleAnalyzer(api_key=api_key)
    analyzer.prepare_corpus(records, source_name=source_name)

    enriched_articles = []
    for i, art in enumerate(records):
        title = art.get('post_title') or art.get('title') or art.get('name') or 'Untitled'
        print(f"   Processing [{i+1}/{len(records)}]: {str(title)[:30]}...")
        enriched = analyzer.process(art, rank_stats=rank_map)
        enriched_articles.append(enriched)

    print("4. Converting to MeTTa...")
    converter = JsonToMetta(include_author_alias=False, excluded=excluded_predicates())
    metta_output = converter.convert(enriched_articles)
    fact_count = len([line for line in metta_output.splitlines() if line.strip()])

    print(f"5. Saving to {output_path}...")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(metta_output)

    print("Done!")
    plan = getattr(analyzer, "plan", None)
    return {
        "status": "success",
        "message": f"Ingested {len(records)} records from source {source_name}",
        "source": source_name,
        "records": len(records),
        "facts": fact_count,
        "output_path": output_path,
        "planner": getattr(plan, "planner", None) if plan else None,
        "property_count": len(getattr(plan, "properties", [])) if plan else 0,
        "properties": [spec.name for spec in getattr(plan, "properties", [])] if plan else [],
    }

def main():
    run_ingestion()

if __name__ == "__main__":
    main()
