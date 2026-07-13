from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from typing import Any, Callable

from flask import jsonify, request

from experiments.api.errors import error_payload, public_error, unexpected_error
from experiments.ingestion.config import DEFAULT_USERNAME
from experiments.ingestion.fetcher import MindplexFetcher

active_mining_lock = threading.Lock()


def register_core_routes(
    app,
    *,
    logger: Any,
    run_ingestion: Callable[..., dict],
    reload_petta_dataset_if_ready: Callable[..., dict],
    dataset_file_path: Callable[[], str],
    get_chainer_service: Callable[[], Any],
    petta_startup_error_type: type[Exception],
    default_conjunction_count: int,
    default_min_support: int,
    default_chain_depth: int,
    mining_jobs: dict[str, Any],
    mining_job_type: type,
    run_mining_task: Callable[..., dict],
    simulate_engagement: Callable[[dict], dict],
    run_chainer_query: Callable[[str, int], list[str]],
    make_json_safe: Callable[[Any], Any],
) -> None:
    @app.route('/api/ingest', methods=['POST'])
    def ingest_data():
        try:
            data = request.get_json(silent=True) or {}
            username = data.get('username') if data else None
            source_name = data.get('source', "mindplex") if data else None
            limit = data.get('limit') if data else None
            source_config = data.get('source_config') if data else None
            output_path = data.get('output_path') if data else None

            if source_name == "mindplex" and not username:
                return public_error("username_required", "Enter a Mindplex username.", 400)

            logger.info("Received ingestion request for source=%s", source_name)
            result = run_ingestion(
                username=username,
                source_name=source_name,
                limit=int(limit or 50),
                output_path=output_path,
                source_config=source_config,
            )

            if result.get("status") == "error":
                logger.error("Ingestion pipeline failed: %r", result)
                return public_error(
                    "ingestion_failed",
                    "Articles could not be loaded. Check the username and Mindplex session, then try again.",
                    502,
                )

            try:
                result["runtime_dataset_reload"] = reload_petta_dataset_if_ready(force=True)
            except Exception:
                return unexpected_error(
                    logger,
                    "Dataset reload failed after ingestion",
                    "Articles were loaded, but the reasoning engine could not reload the dataset.",
                    code="dataset_reload_failed",
                )

            return jsonify(result)

        except Exception:
            return unexpected_error(
                logger,
                "Ingestion endpoint failed",
                "The ingestion request could not be completed.",
                code="ingestion_failed",
            )

    @app.route('/api/data.metta', methods=['GET'])
    def get_metta_dataset():
        try:
            path = dataset_file_path()
            if not os.path.exists(path):
                return public_error("dataset_not_found", "No dataset is available. Run ingestion first.", 404)

            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()

            response = app.response_class(content, mimetype="text/plain")
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        except Exception:
            return unexpected_error(
                logger,
                "Failed to read dataset",
                "The dataset could not be read.",
                code="dataset_read_failed",
            )

    @app.route('/api/health', methods=['GET'])
    def health_check():
        try:
            chainer = get_chainer_service()
            chainer_health = chainer.health()
            return jsonify({
                'status': 'healthy',
                'service': 'mining-api',
                'petta': {
                    'mining': {'mode': 'subprocess'},
                    'chainer': {
                        'status': chainer_health.get('status', 'unknown'),
                        'added_atoms': chainer_health.get('added_atoms', 0),
                    },
                },
            })
        except petta_startup_error_type:
            logger.exception("PeTTa health check failed")
            return jsonify({
                'status': 'unhealthy',
                'service': 'mining-api',
                'message': 'The reasoning engine is unavailable.',
                'error': {'code': 'reasoning_engine_unavailable', 'message': 'The reasoning engine is unavailable.'},
            }), 503

    @app.route('/api/mindplex/auth', methods=['GET', 'POST'])
    def mindplex_auth_status():
        try:
            username = request.args.get('username') or os.getenv("MINDPLEX_USERNAME") or DEFAULT_USERNAME
            fetcher = MindplexFetcher(username=username)

            if request.method == 'POST':
                authenticated = fetcher.ensure_authenticated()
            else:
                authenticated = bool(fetcher.token)

            return jsonify({
                'status': 'ok' if authenticated else 'not_authenticated',
                'authenticated': authenticated,
                'mindplex': fetcher.auth_status(),
            }), 200 if authenticated or request.method == 'GET' else 503
        except Exception:
            return unexpected_error(
                logger,
                "Mindplex authentication check failed",
                "Mindplex authentication could not be verified. Check the service account configuration.",
                code="mindplex_auth_failed",
                status=503,
            )

    @app.route('/api/mine', methods=['POST'])
    def start_mining():
        data = request.get_json(silent=True) or {}
        conjunction_count = data.get('conjunction_count', default_conjunction_count)
        min_support = data.get('min_support', default_min_support)

        if not isinstance(conjunction_count, int) or conjunction_count < 1:
            return public_error("invalid_conjunction_count", "Conjunction count must be a positive integer.", 400)
        if not isinstance(min_support, int) or min_support < 1:
            return public_error("invalid_min_support", "Minimum support must be a positive integer.", 400)

        job_id = str(uuid.uuid4())
        job = mining_job_type(
            job_id=job_id,
            status='queued' if active_mining_lock.locked() else 'running',
            conjunction_count=conjunction_count,
            min_support=min_support,
        )
        mining_jobs[job_id] = job

        def run_job() -> None:
            job_for_thread = mining_jobs[job_id]
            try:
                with active_mining_lock:
                    logger.info(
                        "Mining job %s started: conjunction_count=%s min_support=%s",
                        job_id,
                        conjunction_count,
                        min_support,
                    )
                    job_for_thread.status = 'running'
                    run_mining_task(job_id, conjunction_count, min_support)
                    logger.info("Mining job %s finished with status=%s", job_id, job_for_thread.status)
            except Exception:
                logger.exception("Mining background job %s failed", job_id)
                job_for_thread.status = 'error'
                job_for_thread.error = "Mining failed inside the reasoning engine."
                job_for_thread.end_time = time.time()

        thread = threading.Thread(target=run_job, name=f"mine-{job_id}", daemon=True)
        thread.start()

        return jsonify({
            'jobId': job_id,
            'status': job.status,
            'conjunction_count': conjunction_count,
            'min_support': min_support,
            'message': 'Mining job started',
        }), 202

    @app.route('/api/mine/<job_id>', methods=['GET'])
    def get_mining_status(job_id: str):
        if job_id not in mining_jobs:
            return public_error("mining_job_not_found", "Mining job not found.", 404)

        job = mining_jobs[job_id]
        response = {
            'jobId': job_id,
            'status': job.status,
            'conjunction_count': job.conjunction_count,
            'min_support': job.min_support,
            'startTime': job.start_time,
        }

        if job.status == 'completed' and job.result is not None:
            result = make_json_safe(job.result)
            response['result'] = result
            if isinstance(result, dict):
                mining_status = result.get('status')
                if mining_status == 'success':
                    response['message'] = 'Mining job finished successfully'
                elif mining_status == 'no_results':
                    response['message'] = result.get(
                        'message',
                        'No patterns found for the selected parameters.',
                    )
        elif job.status == 'error' and job.error:
            response['error'] = job.error

        return jsonify(response)

    @app.route('/api/mine', methods=['GET'])
    def list_mining_jobs():
        jobs = []
        for job_id, job in mining_jobs.items():
            job_info = {
                'jobId': job_id,
                'status': job.status,
                'conjunctionCount': job.conjunction_count,
                'minSupport': job.min_support,
                'startTime': job.start_time,
            }
            jobs.append(job_info)

        return jsonify({'jobs': jobs})

    @app.route('/api/simulate', methods=['POST'])
    def simulate_endpoint():
        try:
            payload = request.get_json() or {}
            result = simulate_engagement(payload)
            if result.get("status") == "error":
                logger.warning("Simulation rejected: %r", result)
                return public_error(
                    "simulation_failed",
                    "Simulation could not be completed. Mine rules first and verify the selected attributes.",
                    400,
                )
            return jsonify(result)
        except ValueError:
            logger.warning("Invalid simulation request", exc_info=True)
            return public_error(
                "invalid_simulation_request",
                "Check the selected attributes and confidence values, then try again.",
                400,
            )
        except Exception:
            return unexpected_error(
                logger,
                "Simulation endpoint failed",
                "The simulation could not be completed.",
                code="simulation_failed",
            )

    @app.route('/api/chainer', methods=['POST'])
    def query_chainer():
        payload = request.get_json(silent=True) or {}
        query = payload.get('query')
        depth = payload.get('depth', default_chain_depth)

        if not isinstance(query, str) or not query.strip():
            return public_error(
                "query_required",
                "Provide a non-empty MeTTa query.",
                400,
            )
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            return public_error(
                "invalid_chain_depth",
                "Chain depth must be a positive integer.",
                400,
            )

        query = query.strip()
        try:
            proofs = run_chainer_query(query, depth)
        except subprocess.TimeoutExpired:
            logger.warning("Chainer endpoint timed out: depth=%s query=%s", depth, query)
            return public_error(
                "chainer_timeout",
                "The reasoning engine did not complete the query in time.",
                504,
            )
        except Exception:
            return unexpected_error(
                logger,
                "Chainer endpoint failed",
                "The reasoning engine could not complete the query.",
                code="chainer_failed",
            )

        return jsonify({
            'status': 'success' if proofs else 'no_proof',
            'query': query,
            'depth_used': depth,
            'proof_count': len(proofs),
            'proofs': proofs,
        })

    @app.errorhandler(404)
    def not_found(error):
        return public_error("endpoint_not_found", "Endpoint not found.", 404)

    @app.errorhandler(500)
    def internal_error(error):
        error_id = uuid.uuid4().hex[:12]
        logger.error("Unhandled API error [error_id=%s]", error_id, exc_info=error)
        return jsonify(error_payload(
            "internal_error",
            "The server could not complete this request.",
            error_id=error_id,
        )), 500
