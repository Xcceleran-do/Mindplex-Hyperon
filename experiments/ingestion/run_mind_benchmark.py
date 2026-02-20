import argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.ingestion.mind_adapter import convert_mind_to_metta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a one-shot MIND benchmark export for AtomSpace visualizer and boss-ready report."
    )
    parser.add_argument("--mind-dir", required=True, help="Path to MIND dataset directory.")
    parser.add_argument(
        "--output-metta",
        default="experiments/atomspace_visualizer/public/data.metta",
        help="Output MeTTa facts file used by the visualizer/miner.",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/reports",
        help="Directory for benchmark summary artifacts.",
    )
    parser.add_argument(
        "--min-articles",
        type=int,
        default=1000,
        help="Fail if fewer than this many articles are loaded (helps catch wrong dataset path).",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=10000,
        help="Cap exported articles for faster visualization (top by impressions).",
    )
    args = parser.parse_args()

    stats = convert_mind_to_metta(
        mind_dir=args.mind_dir,
        output_metta_path=args.output_metta,
        report_dir=args.report_dir,
        min_articles=args.min_articles,
        max_articles=args.max_articles,
    )

    print("\n=== Ready to show your boss ===")
    print(f"Articles processed: {stats['article_count']}")
    print(f"Total impressions: {stats['total_impressions']}")
    print(f"Average CTR: {stats['avg_ctr']:.4f}")
    print(f"Data file updated: {stats['output_metta_path']}")
    print(f"Shareable report: {stats['report_md_path']}")
    print("Loaded splits/files:")
    for entry in stats.get("loaded_files", []):
        print(
            f"- {entry['news_path']} | news={entry['news_records']} | behaviors={entry['has_behaviors']}"
        )


if __name__ == "__main__":
    main()
