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

# Add the current directory to Python path to import talk_with_metta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from talk_with_metta import mine_pattern
except ImportError as e:
    print(f"Error importing mine_pattern: {e}")
    print("Make sure talk_with_metta.py is in the same directory")
    sys.exit(1)

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
        job.result = result
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
        conjunction_count = data.get('conjunctionCount', 2)
        
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
        thread = threading.Thread(
            target=run_mining_task,
            args=(job_id, conjunction_count),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'jobId': job_id,
            'status': 'running',
            'conjunctionCount': conjunction_count,
            'message': 'Mining job started successfully'
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
        'conjunctionCount': job.conjunction_count,
        'startTime': job.start_time
    }
    
    if job.end_time:
        response['endTime'] = job.end_time
        response['duration'] = job.end_time - job.start_time
    
    if job.status == 'completed' and job.result is not None:
        response['result'] = job.result
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
        port=5000,
        debug=True,
        threaded=True
    )