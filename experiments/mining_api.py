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
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY4"))

metta4Miner = MeTTa()

metta4Miner.run("""
    ! (register-module! experiments)

    ! (import! &self experiments:pattern-miner:pattern-miner)
    ! (import! &self experiments:utils:common-utils)
    ! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
    ! (import! &tempo experiments:atomspace_visualizer:public:small-ugly)
    ! (import! &self experiments:chainer:main)
                            
    !(bind! &res1 (new-space)) ;; space to hold the formatted miner result
    !(add-reduct &res1 (let $fact (get-atoms &tempo) (: (fact:- $fact) $fact)))
                
    ! (bind! purifiedDbSpace (new-space)) ; space to hold the database atoms
    ! (add-reduct purifiedDbSpace (get-atoms &tempo))
""")

def mine_pattern(numberOfConjunction: int) -> dict:
    """
    Mines patterns with a specified number of conjunctions.

    Args:
        numberOfConjunction: The number of conjunctions to use in pattern mining.

    Returns:
        A dictionary containing the mining results with parsed patterns.
    """
    answer = metta4Miner.run(f"!(pattern-miner purifiedDbSpace 3 {int(numberOfConjunction)})")
    
    # Parse the result into JSON-serializable format
    if not answer or len(answer) == 0:
        return {"status": "no_results", "patterns": []}
    
    try:
        # Extract the atom result
        result_atom = answer[0][0]
        list_of_patterns = result_atom.get_children()
        
        # Parse each pattern
        patterns = []
        for pattern_atom in list_of_patterns:
            pattern_parts = pattern_atom.get_children()
            if len(pattern_parts) >= 3:  # (supportOf <pattern> <count>)
                pattern_str = str(pattern_parts[1])
                support_str = str(pattern_parts[2])
                patterns.append({
                    "pattern": pattern_str,
                    "support": support_str
                })
        
        return {
            "answer": f"{result_atom}",
            "status": "success",
            "conjunction_count": numberOfConjunction,
            "patterns": patterns,
            "total_count": len(patterns)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse mining result: {str(e)}",
            "raw_result": str(answer)
        }


app = Flask(__name__)
# Enable CORS for all domains on all routes with all methods
CORS(app, resources={r"/api/*": {
    "origins": "*",  # Allow all origins
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": False,
    "max_age": 3600
}})

def getAllFactsAndRules():
    """" """"
    facts = metta4Miner.run("!(collapse (get-atoms &res1))")
    return facts

# Define available functions for the AI with proper docstrings for automatic function calling
def get_mining_results() -> dict:
    """Retrieves the latest pattern mining results from the system.
    
    Use this when the user asks about mining results, patterns found, or says "Mine rules with X patterns".
    
    Returns:
        A dictionary containing all patterns with their indices, support values, and properties.
    """
    jobs = list(mining_jobs.values())
    if not jobs:
        return {"status": "no_results", "message": "No mining jobs have been run yet."}
    
    latest_job = max(jobs, key=lambda j: j.start_time)
    if latest_job.status != 'completed':
        return {"status": "not_ready", "message": f"Latest job is {latest_job.status}"}
    
    # Parse all patterns to extract detailed information
    patterns_data = []
    if latest_job.result and isinstance(latest_job.result, dict):
        # Get patterns from the dict returned by mine_pattern()
        patterns = latest_job.result.get('patterns', [])
        for idx, item in enumerate(patterns, 1):
            pattern = item.get('pattern', '')
            support = item.get('support', '0')
            properties = parse_pattern(pattern)
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
        "conjunction_size": latest_job.conjunction_count
    }

def analyze_specific_pattern(pattern: str) -> dict:
    """Analyzes a specific pattern in detail, extracting properties and values.
    
    Args:
        pattern: The pattern string to analyze, e.g., '((length $x "low") (engagement_level $x "high"))'
        
    Returns:
        A dictionary with pattern analysis including properties and their values.
    """
    properties = parse_pattern(pattern)
    return {
        "pattern": pattern,
        "properties": properties,
        "property_count": len(properties),
        "description": f"Pattern with {len(properties)} properties: {', '.join(properties.keys())}"
    }

def get_pattern_statistics() -> dict:
    """Gets statistics about all mining results including total jobs and patterns.
    
    Returns:
        A dictionary with statistics about all completed mining jobs.
    """
    jobs = [j for j in mining_jobs.values() if j.status == 'completed']
    if not jobs:
        return {"status": "no_data", "message": "No completed mining jobs"}
    
    # Each job.result is expected to be the dict returned by mine_pattern().
    # Count how many patterns are stored under the 'patterns' key for each job.
    total_patterns = sum(
        (len(j.result.get('patterns', [])) if isinstance(j.result, dict) else 0)
        for j in jobs
    )

    return {
        "total_jobs": len(jobs),
        "total_patterns": total_patterns,
        "average_patterns_per_job": total_patterns / len(jobs) if jobs else 0
    }

def visualize_pattern_request(pattern: str) -> dict:
    """Requests visualization of a specific pattern on the graph canvas.
    
    Args:
        pattern: The pattern string to visualize
        
    Returns:
        A confirmation message that the pattern will be visualized.
    """
    return {
        "action": "visualize",
        "pattern": pattern,
        "message": "Pattern visualization requested. The frontend will display this pattern."
    }

def run_mining_task(job_id: str, conjunction_count: int):
    """
    Run the mining task for a given job.
    Args:
        job_id (str): Unique identifier for the mining job.
        conjunction_count (int): Number of conjunctions to use in the mining process.
    Returns:
        dict: A dictionary containing the job status, result, error (if any), and timestamps.
    """
    job = mining_jobs[job_id]
    job.start_time = time.time()
    try:
        result = mine_pattern(conjunction_count)
        job.status = 'completed'
        job.result = result  # Store the dict directly, not result[0][0]
        job.end_time = time.time()
        return {
            'jobId': job_id,
            'status': job.status,
            'result': job.result,
            'start_time': job.start_time,
            'end_time': job.end_time
        }
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        job.end_time = time.time()
        return {
            'jobId': job_id,
            'status': job.status,
            'error': job.error,
            'start_time': job.start_time,
            'end_time': job.end_time
        }

def start_mining_job(conjunction_count: int):
    """
    Wrapper that creates a MiningJob and runs the mining task synchronously
    so that function-calls from the LLM go through the same code path as the
    HTTP `/api/mine` endpoint.
    Returns a dict similar to the `/api/mine` response (jobId, status, result).
    """
    try:
        if not isinstance(conjunction_count, int):
            conjunction_count = int(conjunction_count)
    except Exception:
        return { 'error': 'conjunction_count must be an integer' }

    job_id = str(uuid.uuid4())
    job = MiningJob(job_id=job_id, status='running', conjunction_count=conjunction_count)
    mining_jobs[job_id] = job

    # Run synchronously (this will call mine_pattern internally)
    result = run_mining_task(job_id, conjunction_count)

    # Kick off formatter thread as the HTTP endpoint does, but only if we have an answer
    try:
        if isinstance(mining_jobs[job_id].result, dict) and mining_jobs[job_id].result.get('answer'):
            thread = threading.Thread(
                target=formatter,
                args=(f"{mining_jobs[job_id].result['answer']}",),
                daemon=True
            )
            thread.start()
    except Exception as e:
        print('start_mining_job: formatter thread failed to start', e)

    # Return a normalized result
    return {
        'jobId': job_id,
        'status': mining_jobs[job_id].status,
        'conjunction_count': conjunction_count,
        'result': mining_jobs[job_id].result
    }

def formatter(mined_patterns):
    print("formatter started :--:")
    mined_patterns = metta4Miner.parse_single(mined_patterns)
    metta4Miner.run(f""" !(add-reduct &res1 (main {mined_patterns})) """)
    print("formatter ended :-_-:")

def backWardChainer(whatToCheck, depth=5):
    whatToCheck = metta4Miner.parse_single(whatToCheck)
    answer = metta4Miner.run(f""" !(backward-chain &res1 (fromNumber {2}) (: $prf {whatToCheck})) """)
    return answer

def getChainerResult(whatToCheck, depth=5):
    """ Get the result of backward chaining for a specific query. 
    Args:
        whatToCheck (str): The query to check, e.g., '(reputation 0 high)'
        depth (int): The depth limit for backward chaining. (default 5)
    Returns:
        The justification of the backward chaining operation.
    """
    chainAnswer = backWardChainer(whatToCheck, depth)
    
    # If no proofs found, return early
    if not chainAnswer or len(chainAnswer) == 0:
        return {
            "query": whatToCheck,
            "status": "no_proof",
            "justification": f"No logical proof could be found for the query '{whatToCheck}' within depth {depth}. This means the query cannot be deduced from the available rules and facts in the knowledge base."
        }
    
    # Simple prompt that relies on system instruction for formatting guidance
    prompt = f"""
        Analyze this backward chaining result and provide a clear justification:

        **Query:** {whatToCheck}
        **Backward Chaining Results:** {chainAnswer}

        **Backward Chaining Example:**
        When user asks "why is article 1 did get high engagement?", format query as "(engagement_level 1 high)" and call getChainerResult. 
        
        If backward chaining returns: [(: ((rule:- (, (engagement_level 1 high) (topic 1 AI))) (fact:- (topic 1 AI))) (engagement_level 1 high)), (: ((rule:- (, (engagement_level 1 high) (length 1 low))) (fact:- (length 1 low))) (engagement_level 1 high))]
        
        Analyze as: "I found 2 proofs for why article 1 has high engagement:
        
        **Proof 1:** Based on the rule that states 'if an article is about AI, then it has high engagement', and since we have the fact that 'article 1 is about AI', we can conclude that article 1 has high engagement.
        
        **Proof 2:** Based on the rule that states 'if an article is short (low length), then it has high engagement', and since we have the fact that 'article 1 has low length', we can also conclude that article 1 has high engagement.
        
        **Overall Justification:** Article 1's high engagement is well-supported by two independent logical proofs - both its AI topic and its concise length contribute to high engagement according to the rules in our knowledge base."

        The backward chaining system tried to prove the query "{whatToCheck}" and found the above results. Please analyze these results and explain the logical reasoning behind the proof(s).
        """

    try:
        # Use Gemini to analyze the results
        response = model.generate_content(prompt)
        justification = response.text if response.text else "Unable to generate justification analysis."
        
        return {
            "query": whatToCheck,
            "status": "success",
            "raw_proofs": str(chainAnswer),
            "proof_count": len(chainAnswer),
            "justification": justification,
            "depth_used": depth
        }
        
    except Exception as e:
        # Fallback to basic analysis if LLM fails
        proof_count = len(chainAnswer)
        basic_justification = f"""
        **Query Analysis:** {whatToCheck}

        **Result:** Found {proof_count} logical proof(s) supporting this query.

        **Raw Evidence:** {chainAnswer}

        **Basic Interpretation:** The backward chaining system discovered {proof_count} different logical path(s) that support the query "{whatToCheck}". Each proof represents a combination of rules and facts from the knowledge base that logically leads to this conclusion.

        **Note:** Advanced analysis unavailable due to processing error: {str(e)}
        """
        
        return {
            "query": whatToCheck,
            "status": "partial_success",
            "raw_proofs": str(chainAnswer),
            "proof_count": proof_count,
            "justification": basic_justification,
            "depth_used": depth,
            "error": str(e)
        }


def summarize_patterns(patterns: list) -> str:
    """Use the Gemini model to create a single comprehensive summary of the
    supplied mined patterns. The summary will reference patterns as [Pattern N]
    so the frontend can make them clickable for visualization.
    """
    if not patterns:
        return "No patterns to summarize."

    # Build a compact prompt that lists the patterns and asks for a concise
    # analytic summary that includes references like [Pattern 1], [Pattern 2]
    prompt_parts = ["Please analyze the following mined patterns and produce a single concise summary that references specific patterns using [Pattern N] notation. Explain trends and actionable insights."]
    for i, p in enumerate(patterns, 1):
        patt = p.get('pattern') if isinstance(p, dict) else str(p)
        supp = p.get('support', '') if isinstance(p, dict) else ''
        prompt_parts.append(f"[Pattern {i}] Pattern: {patt} -- Support: {supp}")

    prompt = "\n\n".join(prompt_parts)

    try:
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, 'text') and resp.text else None
        if not text:
            # Try extracting from candidates-like structure
            try:
                parts = resp.candidates[0].content.parts
                acc = ""
                for part in parts:
                    if hasattr(part, 'text'):
                        acc += part.text
                text = acc
            except Exception:
                text = None

        return text or "The model could not produce a summary."
    except Exception as e:
        print('summarize_patterns error:', e)
        return f"Summary generation failed: {e}"

# Function name to actual function mapping (for execution)
available_functions = {
    "mine_pattern": start_mining_job,
    # Aliases so the model can call either the wrapper or the original-style name
    "start_mining_job": start_mining_job,
    "startMiningJob": start_mining_job,
    "minePattern": start_mining_job,
    "get_mining_results": get_mining_results,
    "analyze_specific_pattern": analyze_specific_pattern,
    "get_pattern_statistics": get_pattern_statistics,
    "visualize_pattern_request": visualize_pattern_request,
    "getChainerResult": getChainerResult
}

# Initialize Gemini model with automatic function calling
model = genai.GenerativeModel(
    "gemini-2.0-flash-exp",
    tools=[start_mining_job, mine_pattern, analyze_specific_pattern, get_pattern_statistics, visualize_pattern_request, getChainerResult],
    system_instruction="""You are a friendly and knowledgeable AI assistant with expertise in data mining patterns, knowledge graphs, and pattern analysis. 

        **Your Primary Specialty:**
        You excel at analyzing pattern mining results, explaining conjunctions, and providing insights about relationships in data.

        **When to Use Functions:**
        - User says "Mine rules with X patterns" | "What patterns were found?" | "Show me the patterns" |or something like this → ALWAYS call mine_pattern(job_id: str , with the given conjunct number or default 3) first
        - "Analyze this pattern" / "Explain this pattern" → Use analyze_specific_pattern()
        - "Statistics" / "how many patterns" → Use get_pattern_statistics()
        - "Visualize" / "show me" a pattern → Use visualize_pattern_request()
        - "Why is..." / "Explain why..." / "Prove that..." questions → Use getChainerResult() with the query formatted as a MeTTa expression

        **CRITICAL: When User Says "Mine rules with X patterns":**
        1. ALWAYS call mine_pattern() immediately to get all patterns
        2. Analyze ALL patterns together to find common themes
        3. Create ONE comprehensive summary (not individual summaries)
        4. In your summary, reference specific patterns using [Pattern N] notation where N is the pattern index
        5. Format: "Based on the mining results, most of high engagement level is correlated to... [Pattern 1] ... the longer the article is ... [Pattern 3]"
        6. Focus on insights and trends across ALL patterns

        **Pattern Reference Format:**
        - Use [Pattern 1], [Pattern 2], etc. to reference patterns in your summary
        - These will become clickable for visualization
        - Only reference patterns that support your statements
        - You don't need to list the patterns separately, just reference them in context

        **When Analyzing Patterns:**
        1. Explain what the pattern represents in simple terms
        2. Interpret variables (like $x) as placeholders for entities (articles/topics)
        3. Describe what kind of entities would match this pattern
        4. For visualization: ALL conditions must be met (AND logic, not OR)
        5. Provide practical examples when possible

        **General Conversations:**
        You can engage in friendly, helpful conversations on any topic. If someone asks about something outside of pattern mining:
        - Answer naturally and helpfully based on your general knowledge
        - Be conversational and engaging
        - If appropriate, you can relate the topic back to data analysis, patterns, or insights
        - Never say "that's outside my scope" - just answer the question to the best of your ability

        **Backward Chaining Analysis (for getChainerResult function):**
        When analyzing backward chaining results, you are an expert in logical reasoning and knowledge graph analysis. Provide clear, human-readable justifications that explain:

        1. **Main Conclusion:** What was proven and with how many different proof paths
        2. **Proof Analysis:** For each proof path, explain:
           - What rule was used
           - What facts were needed
           - How they combine to prove the query
        3. **Logical Reasoning:** Explain the logical flow in simple terms
        4. **Confidence:** Based on the number of proofs and their strength

        **Backward Chaining Response Format:**
        "Based on the backward chaining analysis, we have found [X] different logical proofs for why [query explanation].

        **Proof 1:** The rule states that [rule explanation], and since we have the fact that [fact explanation], we can conclude that [conclusion].

        **Proof 2:** Another supporting rule indicates that [rule explanation], combined with the established fact [fact explanation], also leads to [conclusion].

        **Overall Justification:** [Summary of why this conclusion is well-supported]"

        **Backward Chaining Style Guidelines:**
        - do not call the function getChainerResult() more than once, just call once.
        - Use clear, conversational language
        - Avoid technical jargon
        - Focus on the logical reasoning
        - Be concise but thorough
        - Use bullet points or numbered lists for clarity

        **Communication Style:**
        - Be friendly, concise, and informative
        - Use emojis occasionally to keep things engaging (but not excessively)
        - Format responses with markdown: **bold**, *italic*, `code`
        - Adapt your tone to match the user's style

        Remember: While your expertise is in pattern mining, you're a helpful general-purpose assistant who can discuss any topic!"""
)
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'mining-api'})

@app.route('/api/mine', methods=['POST'])
def start_mining():
    """Start a new mining job"""
    print("🔍 DEBUG: Received mining request")

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
    run_mining_task(job_id, conjunction_count)
    print(f"🔍 DEBUG: Starting mining job {job_id} with conjunction count {conjunction_count}")
    result = mining_jobs[job_id].result
    # Start formatting in background thread
    print("🔍 DEBUG: Starting formatting thread")
    print("🔍 DEBUG: Result before formatting =", result)
    thread = threading.Thread(
        target=formatter,
        args=(f"{result['answer']}",),
        daemon=True
    )
    thread.start()
    
    print(f"🔍 DEBUG: result type = {type(result)}")
    print(f"🔍 DEBUG: result = {result}")
    
    # Check if mining was successful
    if isinstance(result, dict) and result.get('status') == 'success':
        rules = result.get('patterns', [])
        print(f"✅ Mining job {job_id} finished with {len(rules)} patterns")
        return jsonify({
            'jobId': job_id,
            'status': 'finished',
            'conjunction_count': conjunction_count,
            'message': 'Mining job finished successfully',
            'result': rules
        })
        
    else:
        # Handle error case
        error_msg = result.get('message', 'Unknown error') if isinstance(result, dict) else str(result)
        print(f"❌ Mining error: {error_msg}")
        return jsonify({
            'jobId': job_id,
            'status': 'error',
            'message': error_msg
        }), 500
    

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

@app.route('/api/chat/analyze', methods=['POST', 'OPTIONS'])
def analyze_conjunct():
    """Analyze a pattern/conjunct and return a summary"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response, 200
    
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


@app.route('/api/chat/summarize', methods=['POST', 'OPTIONS'])
def summarize_patterns_endpoint():
    """Summarize a list of patterns into one comprehensive analysis string."""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response, 200

    try:
        data = request.get_json() or {}
        patterns = data.get('patterns', [])
        summary = summarize_patterns(patterns)
        return jsonify({'summary': summary})
    except Exception as e:
        print('Error in summarize_patterns_endpoint:', e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Main chat endpoint with Gemini AI and automatic function calling"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response, 200
    
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
        response = chat_session.send_message(message)
        
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
                            # Normalize args: map common alias names and convert numeric strings to ints
                            norm_args = {}
                            for k, v in function_args.items():
                                key = k
                                if k in ('conjunction_count', 'conjunctions', 'conjunctionCount', 'numberOfConjunction', 'n'):
                                    key = 'conjunction_count'

                                # Normalize numeric-looking strings and floats
                                if isinstance(v, str) and re.fullmatch(r"\d+", v):
                                    norm_args[key] = int(v)
                                elif isinstance(v, float) and v.is_integer():
                                    norm_args[key] = int(v)
                                else:
                                    norm_args[key] = v

                            function_result = available_functions[function_name](**norm_args)
                            function_results.append({
                                'name': function_name,
                                'args': norm_args,
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

        # If the model produced no textual response but it executed a mining
        # function, synthesize a summary from the mining result so the chat
        # endpoint returns a useful assistant message instead of an apology.
        if not response_text:
            # Look for a mining function call result in function_results
            mining_fr = None
            for fr in function_results:
                if fr.get('name') == 'mine_pattern' and isinstance(fr.get('result'), dict):
                    mining_fr = fr
                    break

            if mining_fr:
                # mining_fr['result'] is the dict returned by start_mining_job
                mining_payload = mining_fr['result']
                patterns = []
                try:
                    # mining_payload['result'] should be the mine_pattern dict
                    candidate = mining_payload.get('result') if isinstance(mining_payload, dict) else None
                    if isinstance(candidate, dict):
                        patterns = candidate.get('patterns', [])
                except Exception:
                    patterns = []

                if patterns:
                    try:
                        response_text = summarize_patterns(patterns)
                    except Exception as e:
                        print('Error generating summary after function call:', e)

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

@app.route('/api/chat/clear', methods=['POST', 'OPTIONS'])
def clear_chat():
    """Clear conversation history"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response, 200
    
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