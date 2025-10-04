#!/usr/bin/env python3
"""
Mining API Server
A Flask-based API server that exposes the pattern mining functionality
"""

import os
import sys
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional
from hyperon import MeTTa

metta4Miner = MeTTa()

metta4Miner.run("""
    ! (register-module! ../experiments)

    ! (import! &self experiments:pattern-miner:pattern-miner)
    ! (import! &self experiments:utils:common-utils)
    ! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
    ! (import! &tempo experiments:data:small-ugly)

    !(bind! &db (new-space)) ;; create the database
                
    !(add-reduct &db (get-atoms &tempo)) ;; add the data to the database
                
    !(bind! &dbb (new-space)) ;; create the database
                
    !(bind! &res1 (new-space)) ;; space to hold the miner result
""")

def mine_pattern(numberOfConjunction):
    """this function will mine patterns with the given number of conjunction"""
    answer = metta4Miner.run(f" !(pattern-miner &res1 &db 3 {numberOfConjunction})")
    return answer

# def mine_pattern_demo(numberOfConjunction):
#     """this function will mine patterns with the given number of conjunction"""
#     answer = "((supportOf (some pattern $x) 4) (supportOf (some pattern $y) 3))"

#     return answer

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

@dataclass
class MiningJob:
    job_id: str
    status: str  # 'running', 'completed', 'error'
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: float = 0
    end_time: Optional[float] = None
    conjunction_count: int = 0
class rule:
    pattern: str
    support: int
# In-memory storage for mining jobs
mining_jobs: Dict[str, MiningJob] = {}

def run_mining_task(job_id: str, conjunction_count: int):
    """Run the mining task in a background thread"""
    job = mining_jobs[job_id]
    job.start_time = time.time()
    
    try:
        print(f"Starting mining job {job_id} with {conjunction_count} conjunctions")
        result = mine_pattern(conjunction_count)
        
        job.status = 'completed'
        job.result = result[0][0]
        job.end_time = time.time()
        print(f"Mining job {job_id} completed successfully")
        
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        job.end_time = time.time()
        print(f"Mining job {job_id} failed: {e}")
        traceback.print_exc()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'mining-api'})

@app.route('/api/mine', methods=['POST'])
def start_mining():
    """Start a new mining job"""
    try:
        data = request.get_json() or {}
        conjunction_count = data.get('conjunction_count', 2)
        
        # Validate conjunction count
        if not isinstance(conjunction_count, int) or conjunction_count < 1:
            return jsonify({'error': 'conjunctionCount must be a positive integer'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create new job
        job = MiningJob(
            job_id=job_id,
            status='running',
            conjunction_count=conjunction_count
        )
        mining_jobs[job_id] = job
        
        # Start mining in background thread
        # thread = threading.Thread(
        #     target=run_mining_task,
        #     args=(job_id, conjunction_count),
        #     daemon=True
        # )
        # thread.start()
        run_mining_task(job_id, conjunction_count)
        listOfRowPatterns = mining_jobs[job_id].result.get_children()
        rules = []
        for pattern in listOfRowPatterns:
            pattern = pattern.get_children()
            rules.append({
                "pattern": f'{pattern[1]}',
                "support": f'{pattern[2]}',
            })
        print(f"Mining job {job_id} finished with result: {rules}")
        return jsonify({
            'jobId': job_id,
            'status': 'finished',
            'conjunction count': conjunction_count,
            'message': 'Mining job finished successfully',
            'result': rules
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start mining: {str(e)}'}), 500

@app.route('/api/mine/<job_id>', methods=['GET'])
def get_mining_status(job_id: str):
    """Get the status of a mining job"""
    if job_id not in mining_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = mining_jobs[job_id]
    
    response = {
        'jobId': job_id,
        'status': job.status,
        'conjunction count': job.conjunction_count,
        'startTime': job.start_time
    }
    
    if job.end_time:
        response['endTime'] = job.end_time
        response['duration'] = job.end_time - job.start_time
    
    if job.status == 'completed' and job.result is not None:
        # Ensure the result is JSON serializable (Hyperon / MeTTa atoms are not
        # directly serializable by Flask's JSON encoder). Convert common
        # containers recursively and fall back to string for unknown objects.
        def make_json_serializable(o):
            # primitive types
            if o is None or isinstance(o, (str, int, float, bool)):
                return o
            # dict-like
            if isinstance(o, dict):
                return {k: make_json_serializable(v) for k, v in o.items()}
            # list/tuple/set
            if isinstance(o, (list, tuple, set)):
                return [make_json_serializable(x) for x in o]
            # dataclasses and objects: try to extract __dict__
            if hasattr(o, '__dict__'):
                try:
                    return {k: make_json_serializable(v) for k, v in o.__dict__.items()}
                except Exception:
                    pass
            # Fallback: convert to string (works for Hyperon atoms)
            try:
                return str(o)
            except Exception:
                return repr(o)

        response['result'] = make_json_serializable(job.result)
    elif job.status == 'error' and job.error:
        response['error'] = job.error
    
    return jsonify(response)

@app.route('/api/mine', methods=['GET'])
def list_mining_jobs():
    """List all mining jobs"""
    jobs = []
    for job_id, job in mining_jobs.items():
        job_info = {
            'jobId': job_id,
            'status': job.status,
            'conjunctionCount': job.conjunction_count,
            'startTime': job.start_time
        }
        if job.end_time:
            job_info['endTime'] = job.end_time
            job_info['duration'] = job.end_time - job.start_time
        jobs.append(job_info)
    
    return jsonify({'jobs': jobs})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Mining API Server...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/mine - Start mining job")
    print("  GET  /api/mine/<job_id> - Get job status")
    print("  GET  /api/mine - List all jobs")
    print()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True,
        threaded=True
    )