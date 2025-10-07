#!/usr/bin/env python3
"""
Unified API Server
A Flask-based API server that exposes pattern mining and AI chat functionality
"""

import os
import sys
import time
import traceback
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional
from hyperon import MeTTa
import google.generativeai as genai

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyChGxk4M-RrG4q7_Oi-sPQgGIRBx8snHcs"
genai.configure(api_key=GOOGLE_API_KEY)

metta4Miner = MeTTa()

metta4Miner.run("""
    ! (register-module! experiments)

    ! (import! &self experiments:pattern-miner:pattern-miner)
    ! (import! &self experiments:utils:common-utils)
    ! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
    ! (import! &tempo experiments:data:small-ugly)
                
    !(bind! &res1 (new-space)) ;; space to hold the miner result
                
    ! (bind! purifiedDbSpace (new-space)) ; space to hold the database atoms
    ! (add-reduct purifiedDbSpace (get-atoms &tempo))
""")

def mine_pattern(numberOfConjunction):
    """this function will mine patterns with the given number of conjunction"""
    answer = metta4Miner.run(f" !(pattern-miner purifiedDbSpace 3 {numberOfConjunction})")
    return answer


app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Define available functions for the AI
def get_mining_results():
    """Get the latest mining results"""
    jobs = list(mining_jobs.values())
    if not jobs:
        return {"status": "no_results", "message": "No mining jobs have been run yet."}
    
    latest_job = max(jobs, key=lambda j: j.start_time)
    if latest_job.status != 'completed':
        return {"status": "not_ready", "message": f"Latest job is {latest_job.status}"}
    
    return {
        "status": "success",
        "patterns": latest_job.result if latest_job.result else []
    }

def analyze_specific_pattern(pattern: str):
    """Analyze a specific pattern in detail"""
    properties = parse_pattern(pattern)
    return {
        "pattern": pattern,
        "properties": properties,
        "property_count": len(properties),
        "description": f"Pattern with {len(properties)} properties: {', '.join(properties.keys())}"
    }

def get_pattern_statistics():
    """Get statistics about all mining results"""
    jobs = [j for j in mining_jobs.values() if j.status == 'completed']
    if not jobs:
        return {"status": "no_data", "message": "No completed mining jobs"}
    
    total_patterns = sum(len(j.result) if j.result else 0 for j in jobs)
    return {
        "total_jobs": len(jobs),
        "total_patterns": total_patterns,
        "average_patterns_per_job": total_patterns / len(jobs) if jobs else 0
    }

def visualize_pattern_request(pattern: str):
    """Request to visualize a specific pattern"""
    return {
        "action": "visualize",
        "pattern": pattern,
        "message": "Pattern visualization requested. The frontend will display this pattern."
    }

# Define function declarations for Gemini
function_declarations = [
    {
        "name": "get_mining_results",
        "description": "Retrieves the latest pattern mining results from the system",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "analyze_specific_pattern",
        "description": "Analyzes a specific pattern in detail, extracting properties and values",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The pattern string to analyze, e.g., '((length $x \"low\") (engagement_level $x \"high\"))'"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "get_pattern_statistics",
        "description": "Gets statistics about all mining results including total jobs and patterns",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "visualize_pattern_request",
        "description": "Requests visualization of a specific pattern on the graph canvas",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The pattern string to visualize"
                }
            },
            "required": ["pattern"]
        }
    }
]

# Function name to actual function mapping
available_functions = {
    "get_mining_results": get_mining_results,
    "analyze_specific_pattern": analyze_specific_pattern,
    "get_pattern_statistics": get_pattern_statistics,
    "visualize_pattern_request": visualize_pattern_request
}

# Initialize Gemini model (basic version without function calling for now)
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
)

# System instruction for the AI
SYSTEM_INSTRUCTION = """You are an AI assistant specialized in analyzing data mining patterns and knowledge graphs. 
You help users understand pattern mining results, explain conjunctions, and provide insights about relationships in data.

You have access to the following functions:
- get_mining_results(): Get the latest mining results
- analyze_specific_pattern(pattern): Analyze a pattern in detail
- get_pattern_statistics(): Get statistics about all mining jobs
- visualize_pattern_request(pattern): Request visualization of a pattern

ALWAYS use these functions when users ask about:
- "What patterns were found?" → Use get_mining_results()
- "Show me the patterns" → Use get_mining_results()
- "Analyze this pattern" → Use analyze_specific_pattern()
- "Statistics" or "how many patterns" → Use get_pattern_statistics()
- "Visualize" or "show me" a pattern → Use visualize_pattern_request()

When analyzing a pattern/conjunct:
1. Explain what the pattern represents in simple terms
2. Interpret the variables (like $x) as placeholders for entities
3. Describe what kind of entities would match this pattern
4. Explain the significance of the support value
5. Provide practical examples if possible

Be concise but informative. Use emojis sparingly to make responses friendly.
Format your responses with markdown-like syntax: **bold**, *italic*, `code`.
"""

# Store conversation history
conversations = {}

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
        print(mining_jobs[job_id].result)
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

# ============= CHAT API ENDPOINTS =============

def parse_pattern(pattern: str) -> dict:
    """Parse a pattern string to extract properties and values"""
    properties = {}
    pattern = pattern.strip()
    if pattern.startswith('(') and pattern.endswith(')'):
        pattern = pattern[1:-1]
    
    matches = re.findall(r'\((\w+)\s+\$\w+\s+"([^"]+)"\)', pattern)
    for prop, value in matches:
        properties[prop] = value
    
    return properties

def analyze_pattern(pattern: str, support: str) -> str:
    """Analyze a pattern and generate a summary"""
    properties = parse_pattern(pattern)
    
    if not properties:
        return f"📊 **Pattern Analysis**\n\nPattern: `{pattern}`\nSupport: **{support}**\n\nThis pattern appears {support} times in the dataset."
    
    property_descriptions = []
    for prop, value in properties.items():
        property_descriptions.append(f"**{prop}** = `{value}`")
    
    description = " AND ".join(property_descriptions)
    
    summary = f"""📊 **Pattern Analysis**

**Support:** {support} occurrences

This pattern identifies topics that have:
{chr(10).join(f"• {prop}: **{value}**" for prop, value in properties.items())}

**Interpretation:**
Topics matching this pattern combine {description}. 
The support value of {support} indicates this specific combination appears {support} times in your dataset.

**Example Use Case:**
This pattern can help identify content that has this specific combination of characteristics, useful for content recommendation, categorization, or trend analysis.
"""
    
    return summary

@app.route('/api/chat/health', methods=['GET'])
def chat_health_check():
    """Chat health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'chat-api'})

@app.route('/api/chat/analyze', methods=['POST'])
def analyze_conjunct():
    """Analyze a pattern/conjunct and return a summary"""
    try:
        data = request.get_json()
        pattern = data.get('pattern', '')
        support = data.get('support', '0')
        
        summary = analyze_pattern(pattern, support)
        
        return jsonify({
            'summary': summary,
            'pattern': pattern,
            'support': support
        })
        
    except Exception as e:
        print(f"Error in analyze_conjunct: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with Gemini AI and automatic function calling"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        history = data.get('history', [])
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Build conversation history for Gemini
        gemini_history = []
        for msg in history[-10:]:
            if msg['role'] == 'user':
                gemini_history.append({
                    'role': 'user',
                    'parts': [msg['content']]
                })
            elif msg['role'] == 'assistant':
                gemini_history.append({
                    'role': 'model',
                    'parts': [msg['content']]
                })
        
        # Start chat with history
        chat_session = model.start_chat(history=gemini_history)
        
        # Send the user message
        response = chat_session.send_message(SYSTEM_INSTRUCTION + "\n\n" + message)
        
        # Handle automatic function calling
        max_iterations = 5
        iteration = 0
        function_results = []
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if the model wants to call a function
            if response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                
                # Check for function call
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    function_args = dict(function_call.args)
                    
                    print(f"🔧 Function call: {function_name}({function_args})")
                    
                    # Execute the function
                    if function_name in available_functions:
                        try:
                            function_result = available_functions[function_name](**function_args)
                            function_results.append({
                                'name': function_name,
                                'args': function_args,
                                'result': function_result
                            })
                            
                            print(f"✓ Function result: {function_result}")
                            
                            # Send function result back to the model
                            response = chat_session.send_message(
                                genai.types.content_types.to_content({
                                    'role': 'function',
                                    'parts': [{
                                        'function_response': {
                                            'name': function_name,
                                            'response': function_result
                                        }
                                    }]
                                })
                            )
                            
                        except Exception as func_error:
                            print(f"✗ Function error: {func_error}")
                            error_result = {'error': str(func_error)}
                            
                            response = chat_session.send_message(
                                genai.types.content_types.to_content({
                                    'role': 'function',
                                    'parts': [{
                                        'function_response': {
                                            'name': function_name,
                                            'response': error_result
                                        }
                                    }]
                                })
                            )
                    else:
                        print(f"✗ Unknown function: {function_name}")
                        break
                else:
                    # No more function calls, we have the final response
                    break
            else:
                break
        
        # Extract final text response
        response_text = ""
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    response_text += part.text
        
        if not response_text:
            response_text = "I apologize, but I couldn't generate a proper response. Please try again."
        
        # Store in conversation history
        conversations[session_id].append({
            'role': 'user',
            'content': message
        })
        conversations[session_id].append({
            'role': 'assistant',
            'content': response_text
        })
        
        return jsonify({
            'response': response_text,
            'functionCalls': function_results,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if session_id in conversations:
            del conversations[session_id]
        
        return jsonify({'status': 'cleared'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Unified API Server (Mining + Chat)...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/mine - Start mining job")
    print("  GET  /api/mine/<job_id> - Get job status")
    print("  GET  /api/mine - List all jobs")
    print("  GET  /api/chat/health - Chat health check")
    print("  POST /api/chat/analyze - Analyze a pattern")
    print("  POST /api/chat - Chat with AI assistant")
    print("  POST /api/chat/clear - Clear conversation history")
    print()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )