import argparse
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "PeTTa" / "run.sh"
DEFAULT_MAIN = REPO_ROOT / "experiments" / "chainer" / "main.metta"
DEFAULT_RULES = REPO_ROOT / "experiments" / "chainer" / "rules.metta"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "chainer" / "rules" / "main_chainer_latest.txt"


def _import_target(path: Path) -> str:
    if path.suffix == ".metta":
        return path.with_suffix("").as_posix()
    return path.as_posix()


def run_query(runner: Path, main_file: Path, rules_file: Path, query: str, depth: int) -> str:
    main_import = _import_target(main_file)
    rules_import = _import_target(rules_file)

    metta_code = (
        f"!(import! &self {main_import})\n"
        f"!(import! &rules {rules_import})\n"
        f"!(backward-chain &rules (fromNumber {depth}) (: $prf {query}))\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".metta", delete=False, encoding="utf-8") as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(metta_code)

    try:
        proc = subprocess.run(
            ["bash", str(runner), str(temp_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "PeTTa run failed")

    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backward chaining through main.metta using exported rules.metta")
    parser.add_argument("--query", required=True, help='MeTTa query, e.g. "(engagement A_N123 \"Low\")"')
    parser.add_argument("--depth", type=int, default=5, help="Backward chaining depth")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="Path to PeTTa run.sh")
    parser.add_argument("--main-file", default=str(DEFAULT_MAIN), help="Path to main.metta")
    parser.add_argument("--rules-file", default=str(DEFAULT_RULES), help="Path to rules.metta")
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT), help="Where to save raw output")
    args = parser.parse_args()

    output = run_query(
        runner=Path(args.runner),
        main_file=Path(args.main_file),
        rules_file=Path(args.rules_file),
        query=args.query,
        depth=args.depth,
    )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output, encoding="utf-8")

    print("main.metta backward chaining completed.")
    print(f"- output: {output_file}")
    print("- preview:")
    preview = output.strip().splitlines()
    for line in preview[-10:]:
        print(line)


if __name__ == "__main__":
    main()
