from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable

from flask import jsonify, request

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
    max_conjunction_count: int,
    mining_jobs: dict[str, Any],
    mining_job_type: type,
    run_mining_task: Callable[..., dict],
    simulate_engagement: Callable[[dict], dict],
    make_json_safe: Callable[[Any], Any],
) -> None:
    @app.route('/api/ingest', methods=['POST'])
    def ingest_data():
        try:
            data = request.get_json()
            username = data.get('username') if data else None
            source_name = data.get('source') if data else None
            source_name = source_name or "mindplex"
            limit = data.get('limit') if data else None
            source_config = data.get('source_config') if data else None

            if source_name == "mindplex" and not username:
                return jsonify({"status": "error", "message": "Username is required"}), 400

            logger.info("Received ingestion request for source=%s", source_name)
            result = run_ingestion(
                username=username,
                source_name=source_name,
                limit=int(limit or 50),
                source_config=source_config,
            )

            if result.get("status") == "error":
                return jsonify(result), 500

            try:
                result["runtime_dataset_reload"] = reload_petta_dataset_if_ready(force=True)
            except Exception as reload_error:
                logger.exception("Dataset reload failed after ingestion")
                return jsonify({
                    "status": "error",
                    "message": "Ingestion wrote data.metta, but PeTTa failed to reload it.",
                    "ingestion": result,
                    "reload_error": str(reload_error),
                }), 500

            return jsonify(result)

        except Exception as exc:
            logger.exception("Ingestion endpoint failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route('/api/data.metta', methods=['GET'])
    def get_metta_dataset():
        try:
            path = dataset_file_path()
            if not os.path.exists(path):
                return jsonify({
                    "status": "error",
                    "message": f"MeTTa dataset not found at {path}",
                }), 404

            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()

            response = app.response_class(content, mimetype="text/plain")
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        except Exception as exc:
            logger.exception("Failed to read dataset")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        try:
            chainer = get_chainer_service()
            return jsonify({
                'status': 'healthy',
                'service': 'mining-api',
                'petta': {
                    'mining': {
                        'mode': 'subprocess',
                        'dataset_path': dataset_file_path(),
                    },
                    'chainer': chainer.health(),
                },
            })
        except petta_startup_error_type as exc:
            return jsonify({
                'status': 'unhealthy',
                'service': 'mining-api',
                'error': str(exc),
            }), 503

    @app.route('/api/mine', methods=['POST'])
    def start_mining():
        data = request.get_json() or {}
        conjunction_count = data.get('conjunction_count', default_conjunction_count)
        min_support = data.get('min_support', default_min_support)

        if not isinstance(conjunction_count, int) or conjunction_count < 1:
            return jsonify({'message': 'conjunction_count must be a positive integer'}), 400
        if conjunction_count > max_conjunction_count:
            return jsonify({
                'message': (
                    f'conjunction_count must be <= {max_conjunction_count}. '
                    'Use PETTA_MAX_CONJUNCTION_COUNT to raise this limit after validating the dataset.'
                )
            }), 400
        if not isinstance(min_support, int) or min_support < 1:
            return jsonify({'message': 'min_support must be a positive integer'}), 400

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
            except Exception as exc:
                logger.exception("Mining background job %s failed", job_id)
                job_for_thread.status = 'error'
                job_for_thread.error = str(exc)
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
            return jsonify({'error': 'Job not found'}), 404

        job = mining_jobs[job_id]
        response = {
            'jobId': job_id,
            'status': job.status,
            'conjunction_count': job.conjunction_count,
            'min_support': job.min_support,
            'startTime': job.start_time,
        }

        if job.end_time:
            response['endTime'] = job.end_time
            response['duration'] = job.end_time - job.start_time

        if job.status == 'completed' and job.result is not None:
            result = make_json_safe(job.result)
            response['result'] = result
            if isinstance(result, dict):
                mining_status = result.get('status')
                if mining_status == 'success':
                    response['patterns'] = result.get('patterns', [])
                    response['inserted_rules'] = result.get('rules', [])
                    response['rule_insertion'] = result.get('rule_insertion')
                    response['message'] = 'Mining job finished successfully'
                elif mining_status == 'no_results':
                    response['patterns'] = []
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
            if job.end_time:
                job_info['endTime'] = job.end_time
                job_info['duration'] = job.end_time - job.start_time
            jobs.append(job_info)

        return jsonify({'jobs': jobs})

    @app.route('/api/simulate', methods=['POST'])
    def simulate_endpoint():
        try:
            payload = request.get_json() or {}
            result = simulate_engagement(payload)
            if result.get("status") == "error":
                return jsonify(result), 400
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            logger.exception("simulate_endpoint failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
