from __future__ import annotations

import os

from dotenv import load_dotenv

from experiments.ingestion.pipeline import resolve_output_path

load_dotenv()

ASI_API_KEY = os.getenv("ASI_API_KEY")
ASI_BASE_URL = os.getenv("ASI_BASE_URL", "https://api.asi1.ai/v1/chat/completions")
ASI_MODEL = os.getenv("ASI_MODEL", "asi1-mini")
ASI_TIMEOUT_SECONDS = float(os.getenv("ASI_TIMEOUT_SECONDS", "45"))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CONJUNCTION_COUNT = int(os.getenv("MINING_DEFAULT_CONJUNCTION_COUNT", "2"))
DEFAULT_MIN_SUPPORT = int(os.getenv("MINING_DEFAULT_MIN_SUPPORT", "3"))
DEFAULT_CHAIN_DEPTH = int(os.getenv("PETTA_CHAIN_DEPTH", "3"))
PETTA_MINING_TIMEOUT_SECONDS = int(os.getenv("PETTA_MINING_TIMEOUT_SECONDS", "90"))
PETTA_MINING_MAX_OUTPUT_BYTES = int(os.getenv("PETTA_MINING_MAX_OUTPUT_BYTES", str(8 * 1024 * 1024)))
PETTA_CHAIN_TIMEOUT_SECONDS = int(os.getenv("PETTA_CHAIN_TIMEOUT_SECONDS", "90"))
PETTACHAINER_BASE_URL = os.getenv("PETTACHAINER_BASE_URL", "http://127.0.0.1:8000")
PETTACHAINER_API_KEY = os.getenv("PETTACHAINER_API_KEY", "")
PETTACHAINER_KB_PREFIX = os.getenv("PETTACHAINER_KB_PREFIX", "mindplex")


def dataset_file_path() -> str:
    output_path = resolve_output_path()
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    return os.path.abspath(output_path)


CHAINER_METTA_SETUP = f"""
!(import! &self {PROJECT_ROOT}/experiments/utils/common-utils)
!(import! &self {PROJECT_ROOT}/experiments/frequent-pattern-miner/etv-utils)
"""

MINING_METTA_SETUP = f"""
!(import! &self {PROJECT_ROOT}/PeTTa/lib/lib_import.metta)
!(import! &self {PROJECT_ROOT}/PeTTa/lib/lib_spaces)
!(import_prolog_functions_from_file "{PROJECT_ROOT}/experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char promote_engagement_conj))
{CHAINER_METTA_SETUP}
!(import! &self {PROJECT_ROOT}/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self {PROJECT_ROOT}/experiments/pattern-miner/pattern-miner)
"""
