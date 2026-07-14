from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from experiments.api.chat.support import parse_pattern as parse_pattern_impl
from experiments.api.config import (
    DEFAULT_CONJUNCTION_COUNT,
    DEFAULT_MIN_SUPPORT,
    MINING_METTA_SETUP,
    PETTA_MINING_MAX_OUTPUT_BYTES,
    PETTA_MINING_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    dataset_file_path,
)
from experiments.api.support import (
    extract_support_of_expressions,
    patterns_to_chainer_rules,
    parse_pattern_string,
    parse_petta_output,
)
from experiments.api.runtime import (
    record_chainer_rules,
    reload_petta_dataset_if_ready,
)

logger = logging.getLogger(__name__)


@dataclass
class MiningJob:
    job_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: float = 0
    end_time: Optional[float] = None
    conjunction_count: int = 0
    min_support: int = DEFAULT_MIN_SUPPORT


mining_jobs: Dict[str, MiningJob] = {}


def run_metta_with_petta(metta_code: str) -> str:
    """
    Run mining MeTTa in a fresh worker process so heavy mining imports do not
    poison the persistent chainer runtime.
    """
    worker_code = f"""
import sys
from experiments.services.petta_service import PeTTaService

service = PeTTaService.create_required(
    project_root={PROJECT_ROOT!r},
    setup_metta={MINING_METTA_SETUP!r},
    verbose=False,
    load_chainer=False,
)
service.reload_dataset({dataset_file_path()!r})
result = service.process_metta_string({metta_code!r})
if isinstance(result, (list, tuple)):
    sys.stdout.write("\\n".join(str(item) for item in result))
else:
    sys.stdout.write(str(result))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    stdout_path = ""
    stderr_path = ""
    process: Optional[subprocess.Popen] = None
    try:
        with tempfile.NamedTemporaryFile(prefix="petta-mine-out-", delete=False) as stdout_file:
            stdout_path = stdout_file.name
        with tempfile.NamedTemporaryFile(prefix="petta-mine-err-", delete=False) as stderr_file:
            stderr_path = stderr_file.name

        with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-c", worker_code],
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                start_new_session=True,
            )

            started_at = time.monotonic()
            killed_reason: Optional[str] = None
            while process.poll() is None:
                elapsed = time.monotonic() - started_at
                output_size = os.path.getsize(stdout_path)
                error_size = os.path.getsize(stderr_path)
                if elapsed > PETTA_MINING_TIMEOUT_SECONDS:
                    killed_reason = f"exceeded {PETTA_MINING_TIMEOUT_SECONDS}s timeout"
                    break
                if output_size > PETTA_MINING_MAX_OUTPUT_BYTES:
                    killed_reason = (
                        f"produced more than {PETTA_MINING_MAX_OUTPUT_BYTES} bytes of output"
                    )
                    break
                if error_size > PETTA_MINING_MAX_OUTPUT_BYTES:
                    killed_reason = (
                        f"produced more than {PETTA_MINING_MAX_OUTPUT_BYTES} bytes of stderr"
                    )
                    break

                time.sleep(0.25)

            if killed_reason:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                raise TimeoutError(f"Mining worker {killed_reason}.")

            process.wait()

        with open(stdout_path, "rb") as handle:
            stdout = handle.read().decode("utf-8", errors="replace")
        with open(stderr_path, "rb") as handle:
            stderr = handle.read().decode("utf-8", errors="replace")

        if process.returncode != 0:
            stderr = stderr.strip() or "unknown mining worker failure"
            raise RuntimeError(f"Mining worker failed: {stderr}")
        return stdout
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for path in (stdout_path, stderr_path):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


def mine_pattern(numberOfConjunction: int, min_support: int = DEFAULT_MIN_SUPPORT) -> dict:
    """Mine frequent patterns with PeTTa and parse the returned support records."""
    try:
        dataset_reload = reload_petta_dataset_if_ready(force=False)
        numberOfConjunction = int(numberOfConjunction)
        min_support = int(min_support)
        if numberOfConjunction < 1:
            return {
                "status": "error",
                "message": "numberOfConjunction must be a positive integer",
            }
        if min_support < 1:
            return {
                "status": "error",
                "message": "min_support must be a positive integer",
            }

        query = f"!(pattern-miner &purifiedDbSpace {min_support} {numberOfConjunction})"
        petta_output = run_metta_with_petta(query)
        normalized_query = query.strip().lstrip("!").strip()
        if petta_output.strip() == normalized_query:
            logger.error("PeTTa returned an unevaluated mining query: %s", petta_output)
            return {
                "status": "error",
                "message": "The reasoning engine did not evaluate the mining query.",
            }
        result_lines = parse_petta_output(petta_output)

        patterns = []
        full_answer_str = " ".join(result_lines)
        support_matches = extract_support_of_expressions(full_answer_str)

        for match in support_matches:
            parsed = parse_pattern_string(match)
            if parsed:
                patterns.append(parsed)

        if not patterns:
            return {"status": "no_results", "patterns": [], "dataset": dataset_reload}

        return {
            "answer": full_answer_str,
            "status": "success",
            "conjunction_count": numberOfConjunction,
            "min_support": min_support,
            "dataset": dataset_reload,
            "patterns": patterns,
            "total_count": len(patterns)
        }

    except Exception:
        logger.exception("mine_pattern failed")
        return {
            "status": "error",
            "message": "Pattern mining failed inside the reasoning engine.",
        }


def insert_mined_rules_into_chainer(mining_result: dict) -> dict:
    """Compile mined patterns into PeTTa rules before returning mining results."""
    if not isinstance(mining_result, dict):
        return mining_result
    if mining_result.get("status") != "success":
        return mining_result

    patterns = mining_result.get("patterns", [])
    if not patterns:
        mining_result["rule_insertion"] = {
            "status": "no_rules",
            "insertedRuleCount": 0,
            "rules": [],
        }
        mining_result["rules"] = []
        mining_result["inserted_rule_count"] = 0
        return mining_result

    rules = patterns_to_chainer_rules(patterns)
    insertion_result = {
        "status": "success",
        "insertedRuleCount": len(rules),
        "rules": rules,
    }
    mining_result["rule_insertion"] = insertion_result
    mining_result["rules"] = insertion_result.get("rules", []) if isinstance(insertion_result, dict) else []
    mining_result["inserted_rule_count"] = (
        insertion_result.get("insertedRuleCount", 0)
        if isinstance(insertion_result, dict)
        else 0
    )
    if isinstance(insertion_result, dict) and insertion_result.get("status") == "success":
        record_chainer_rules(insertion_result.get("rules", []))
    return mining_result


def run_mining_task_inprocess(conjunction_count: int, min_support: int = DEFAULT_MIN_SUPPORT) -> dict:
    result = mine_pattern(conjunction_count, min_support)
    return insert_mined_rules_into_chainer(result)


def run_mining_task(job_id: str, conjunction_count: int, min_support: int = DEFAULT_MIN_SUPPORT):
    job = mining_jobs[job_id]
    job.start_time = time.time()
    job.min_support = min_support

    result_path = ""
    stdout_path = ""
    stderr_path = ""
    process: Optional[subprocess.Popen] = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"petta-job-{job_id}-", suffix=".json", delete=False) as result_file:
            result_path = result_file.name
        with tempfile.NamedTemporaryFile(prefix=f"petta-job-{job_id}-out-", delete=False) as stdout_file:
            stdout_path = stdout_file.name
        with tempfile.NamedTemporaryFile(prefix=f"petta-job-{job_id}-err-", delete=False) as stderr_file:
            stderr_path = stderr_file.name

        with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.api.workers.mining_job",
                    "--conjunction-count",
                    str(conjunction_count),
                    "--min-support",
                    str(min_support),
                    "--result-path",
                    result_path,
                ],
                cwd=PROJECT_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                env={
                    **os.environ,
                    "PYTHONPATH": PROJECT_ROOT
                    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
                },
                start_new_session=True,
            )

            started_at = time.monotonic()
            while process.poll() is None:
                elapsed = time.monotonic() - started_at
                stdout_size = os.path.getsize(stdout_path)
                stderr_size = os.path.getsize(stderr_path)
                result_size = os.path.getsize(result_path)

                if elapsed > PETTA_MINING_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Mining worker exceeded {PETTA_MINING_TIMEOUT_SECONDS}s timeout.")
                if max(stdout_size, stderr_size, result_size) > PETTA_MINING_MAX_OUTPUT_BYTES:
                    raise RuntimeError(
                        f"Mining worker output exceeded {PETTA_MINING_MAX_OUTPUT_BYTES} bytes."
                    )
                time.sleep(0.25)

            if process.returncode != 0:
                with open(stderr_path, "rb") as handle:
                    stderr = handle.read(4096).decode("utf-8", errors="replace").strip()
                raise RuntimeError(stderr or f"Mining worker exited with code {process.returncode}.")

        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)

        job.result = result
        if isinstance(result, dict) and result.get("status") == "error":
            logger.error("Mining worker returned an error for job %s: %r", job_id, result)
            job.status = 'error'
            job.error = "Mining failed inside the reasoning engine."
            job.result = None
        else:
            job.status = 'completed'
            if isinstance(result, dict) and result.get("status") == "success":
                record_chainer_rules(result.get("rules", []))
        job.end_time = time.time()
        return {
            'jobId': job_id,
            'status': job.status,
            'result': job.result,
            'min_support': job.min_support,
            'start_time': job.start_time,
            'end_time': job.end_time
        }
    except Exception:
        logger.exception("Mining job %s failed", job_id)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        job.status = 'error'
        job.error = "Mining failed inside the reasoning engine."
        job.end_time = time.time()
        return {
            'jobId': job_id,
            'status': job.status,
            'error': job.error,
            'min_support': job.min_support,
            'start_time': job.start_time,
            'end_time': job.end_time
        }
    finally:
        for path in (result_path, stdout_path, stderr_path):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


def start_mining_job(conjunction_count: int = DEFAULT_CONJUNCTION_COUNT, min_support: int = DEFAULT_MIN_SUPPORT):
    try:
        if not isinstance(conjunction_count, int):
            conjunction_count = int(conjunction_count)
    except Exception:
        return {'error': 'conjunction_count must be an integer'}
    try:
        if not isinstance(min_support, int):
            min_support = int(min_support)
    except Exception:
        return {'error': 'min_support must be an integer'}
    if conjunction_count < 1:
        return {'error': 'conjunction_count must be a positive integer'}
    if min_support < 1:
        return {'error': 'min_support must be a positive integer'}

    job_id = str(uuid.uuid4())
    job = MiningJob(
        job_id=job_id,
        status='running',
        conjunction_count=conjunction_count,
        min_support=min_support,
    )
    mining_jobs[job_id] = job

    run_mining_task(job_id, conjunction_count, min_support)

    return {
        'jobId': job_id,
        'status': mining_jobs[job_id].status,
        'conjunction_count': conjunction_count,
        'min_support': min_support,
        'result': mining_jobs[job_id].result
    }


def formatter(mined_patterns):
    patterns = mined_patterns.get("patterns", []) if isinstance(mined_patterns, dict) else []
    rules = patterns_to_chainer_rules(patterns)
    record_chainer_rules(rules)
    return {"status": "success", "insertedRuleCount": len(rules), "rules": rules}


def get_mining_results() -> dict:
    jobs = list(mining_jobs.values())
    if not jobs:
        return {"status": "no_results", "message": "No mining jobs have been run yet."}

    latest_job = max(jobs, key=lambda j: j.start_time)
    if latest_job.status != 'completed':
        return {"status": "not_ready", "message": f"Latest job is {latest_job.status}"}

    patterns_data = []
    if latest_job.result and isinstance(latest_job.result, dict):
        patterns = latest_job.result.get('patterns', [])
        for idx, item in enumerate(patterns, 1):
            pattern = item.get('pattern', '')
            support = item.get('support', '0')
            properties = parse_pattern_impl(pattern)
            patterns_data.append({
                "index": idx,
                "pattern": pattern,
                "support": support,
                "properties": properties
            })

    return {
        "status": "success",
        "patterns": patterns_data,
        "total_count": len(patterns_data),
        "conjunction_size": latest_job.conjunction_count,
        "min_support": latest_job.min_support,
        "rules": latest_job.result.get("rules", []) if isinstance(latest_job.result, dict) else [],
        "rule_insertion": latest_job.result.get("rule_insertion") if isinstance(latest_job.result, dict) else None,
    }


def get_pattern_statistics() -> dict:
    jobs = [j for j in mining_jobs.values() if j.status == 'completed']
    if not jobs:
        return {"status": "no_data", "message": "No completed mining jobs"}

    total_patterns = sum(
        (len(j.result.get('patterns', [])) if isinstance(j.result, dict) else 0)
        for j in jobs
    )

    return {
        "total_jobs": len(jobs),
        "total_patterns": total_patterns,
        "average_patterns_per_job": total_patterns / len(jobs) if jobs else 0
    }
