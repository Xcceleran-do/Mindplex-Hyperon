import argparse
import os
import zipfile

from experiments.ingestion.mind_adapter import convert_mind_to_metta


def _extract(zip_path: str, target_dir: str) -> None:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MIND train/dev zips and generate MeTTa + report in one shot."
    )
    parser.add_argument("--train-zip", required=True, help="Path to MINDsmall_train.zip")
    parser.add_argument("--dev-zip", required=True, help="Path to MINDsmall_dev.zip")
    parser.add_argument(
        "--mind-root",
        default="datasets/MIND",
        help="Extraction root; train/ and valid/ will be created here.",
    )
    parser.add_argument(
        "--output-metta",
        default="experiments/atomspace_visualizer/public/data.metta",
        help="Output MeTTa file path.",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/reports",
        help="Directory for JSON/Markdown report artifacts.",
    )
    parser.add_argument(
        "--min-articles",
        type=int,
        default=1000,
        help="Fail if loaded article count is below this threshold.",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.mind_root, "train")
    valid_dir = os.path.join(args.mind_root, "valid")

    _extract(args.train_zip, train_dir)
    _extract(args.dev_zip, valid_dir)

    stats = convert_mind_to_metta(
        mind_dir=args.mind_root,
        output_metta_path=args.output_metta,
        report_dir=args.report_dir,
        min_articles=args.min_articles,
    )

    print("\n=== MIND setup complete ===")
    print(f"Extracted train to: {train_dir}")
    print(f"Extracted valid to: {valid_dir}")
    print(f"Articles processed: {stats['article_count']}")
    print(f"Data file updated: {stats['output_metta_path']}")
    print(f"Shareable report: {stats['report_md_path']}")


if __name__ == "__main__":
    main()
