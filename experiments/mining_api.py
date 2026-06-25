#!/usr/bin/env python3
"""
Unified API Server
A Flask-based API server that exposes pattern mining and AI chat functionality
"""

import os
import sys
import json
import time
import traceback
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import logging  

logging.basicConfig(  
    format="%(asctime)s [%(levelname)s] %(message)s",  
    level=logging.INFO  
)  
logger = logging.getLogger(__name__)  



# Add workspace root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from experiments.ingestion.pipeline import resolve_output_path, run_ingestion
from experiments.services.petta_service import (
    PeTTaService,
    PeTTaStartupError,
    format_proofs_for_prompt,
    unique_preserve_order,
)
load_dotenv()

# Configure ASI API
ASI_API_KEY = os.getenv("ASI_API_KEY")
if not ASI_API_KEY:
    print("WARNING: ASI_API_KEY environment variable is not set. AI features will fail.")
ASI_BASE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-mini"
ASI_TIMEOUT_SECONDS = float(os.getenv("ASI_TIMEOUT_SECONDS", "45"))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT_METTA = os.path.abspath(PROJECT_ROOT).replace('\\', '/')

# Enable extra PeTTa debug probes for backward chaining when set to "1".
PETTA_DEBUG = os.getenv("PETTA_DEBUG", "0") == "1"
DEFAULT_CONJUNCTION_COUNT = 2
DEFAULT_MIN_SUPPORT = 3
DEFAULT_CHAIN_DEPTH = int(os.getenv("PETTA_CHAIN_DEPTH", "3"))


def dataset_file_path() -> str:
    output_path = resolve_output_path()
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    return os.path.abspath(output_path)


def dataset_module_path() -> str:
    path = dataset_file_path()
    if path.endswith(".metta"):
        path = path[:-6]
    return path.replace("\\", "/")
METTA_SETUP = f"""
!(import! &self {PROJECT_ROOT_METTA}/PeTTa/lib/lib_import.metta)
!(import! &self {PROJECT_ROOT_METTA}/PeTTa/lib/lib_spaces)
!(import_prolog_functions_from_file "{PROJECT_ROOT_METTA}/experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char promote_engagement_conj))
!(import! &self {PROJECT_ROOT_METTA}/experiments/utils/common-utils)
!(import! &self {PROJECT_ROOT_METTA}/experiments/frequent-pattern-miner/etv-utils)
!(import! &self {PROJECT_ROOT_METTA}/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self {PROJECT_ROOT_METTA}/experiments/pattern-miner/pattern-miner)
!(import! &self {PROJECT_ROOT_METTA}/experiments/chainer/main)
!(import! &tempo {PROJECT_ROOT_METTA}/experiments/atomspace_visualizer/public/data.metta)
!(import! &stv-formulas {PROJECT_ROOT_METTA}/experiments/PLN/Formulas)

!(let $atom (match &tempo ($fact $stv) (: (fact:- $fact) $fact $stv)) (add-atom &res1 $atom))
!(let $atom (match &tempo ($fact $stv) $fact) (add-atom &purifiedDbSpace $atom))
"""

petta_service: Optional[PeTTaService] = None
runtime_lock = threading.Lock()

def bootstrap_runtime() -> PeTTaService:
    """Initialize the mandatory PeTTa runtime exactly once."""
    global petta_service
    if petta_service is not None:
        return petta_service

    with runtime_lock:
        if petta_service is None:
            logger.info("Initializing mandatory PeTTa runtime")
            petta_service = PeTTaService.create_required(
                project_root=PROJECT_ROOT,
                setup_metta=METTA_SETUP,
                verbose=False,
            )
            petta_service.reload_dataset(dataset_module_path(), dataset_file_path())
            logger.info("PeTTa runtime initialized: %s", petta_service.health())
    return petta_service

def get_petta_service() -> PeTTaService:
    if petta_service is None:
        raise PeTTaStartupError(
            "PeTTa runtime has not been initialized. Start the app through create_app() "
            "or call bootstrap_runtime() before serving requests."
        )
    return petta_service


def reload_petta_dataset_if_ready(force: bool = False) -> dict:
    """Keep PeTTa mining/chainer spaces aligned with the generated data file."""
    service = get_petta_service()
    module_path = dataset_module_path()
    file_path = dataset_file_path()
    if force:
        return service.reload_dataset(module_path, file_path)
    return service.reload_dataset_if_changed(module_path, file_path)

# Define tools for ASI API
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "mine_pattern",
            "description": "Runs the PeTTa pattern miner with a specified number of conjunctions and optional minimum support.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numberOfConjunction": {
                        "type": "integer",
                        "description": "The number of conjunctions to use in pattern mining."
                    },
                    "min_support": {
                        "type": "integer",
                        "description": "Minimum support threshold for returned patterns. Defaults to 3."
                    }
                },
                "required": ["numberOfConjunction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_mining_job",
            "description": "Starts a PeTTa pattern mining job with a specified number of conjunctions and optional minimum support.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conjunction_count": {
                        "type": "integer",
                        "description": "The number of conjunctions to use in pattern mining."
                    },
                    "min_support": {
                        "type": "integer",
                        "description": "Minimum support threshold for returned patterns. Defaults to 3."
                    }
                },
                "required": ["conjunction_count"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_mining_results",
            "description": "Retrieves the latest pattern mining results from the system.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_specific_pattern",
            "description": "Analyzes a specific pattern in detail, extracting properties and values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The pattern string to analyze."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pattern_statistics",
            "description": "Gets statistics about all mining results including total jobs and patterns.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_pattern_request",
            "description": "Requests visualization of a specific pattern on the graph canvas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The pattern string to visualize."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getChainerResult",
            "description": "Get the result of backward chaining for a specific query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "whatToCheck": {
                        "type": "string",
                        "description": "The query to check, e.g., '(reputation 0 \"\\High\")'"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "The depth limit for backward chaining.",
                        "default": DEFAULT_CHAIN_DEPTH
                    }
                },
                "required": ["whatToCheck"]
            }
        }
    }
]

def call_asi_api(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Calls the ASI API with the given messages and tools."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ASI_API_KEY}"
    }
    payload = {
        "model": ASI_MODEL,
        "messages": messages,
        "temperature": 0.7
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        response = requests.post(ASI_BASE_URL, headers=headers, json=payload, timeout=ASI_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ASI API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return {"error": str(e)}

def run_metta_with_petta(metta_code: str) -> str:
    """
    Run MeTTa code using the mandatory in-process PeTTa service.
    There is intentionally no CLI or Hyperon fallback in production mode.
    """
    return get_petta_service().run_metta_string(metta_code)

def parse_petta_output(output: str):
    """
    Parses the output from petta to extract the final result.
    Removes ANSI escape codes and debug lines.
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    lines = clean_output.splitlines()
    result_lines = []
    capture = False
    
    # Simple heuristic to find the result line(s)
    # PeTTa output usually has debug info then the actual result
    # In my test it looked like:
    # "Hello from PeTTa"
    # true
    # We want the lines that aren't debug info.
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("-->") or line.startswith("prolog goal") or line.startswith("metta runnable"):
            continue
        if line.startswith("^^^^^"):
            continue
        result_lines.append(line)
    
    return result_lines

def parse_pattern_string(p_str: str):
    """
    Parses a pattern string like (supportOf ((length $x "low") (engagement $x "high")) 3)
    into a dictionary with 'pattern' and 'support' keys.
    """
    # Capture the last numeric token as support, everything before it as pattern.
    match = re.match(r'^\(supportOf\s+(.+)\s+(\d+)\)$', p_str.strip(), re.DOTALL)
    if match:
        return {
            "pattern": match.group(1).strip(),
            "support": match.group(2).strip()
        }
    return None

def extract_support_of_expressions(text: str) -> list[str]:
    """Extract all (supportOf ...) expressions from text with balanced parentheses."""
    results = []
    start = 0
    while True:
        idx = text.find("(supportOf", start)
        if idx == -1:
            break
        depth = 0
        end = None
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            break
        results.append(text[idx:end])
        start = end
    return results

STV_EXPR_RE = re.compile(
    r"\(STV\s+"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"\)"
)


def _balanced_expression_at(text: str, idx: int) -> Optional[str]:
    """Return the balanced expression starting at idx, ignoring parens in strings."""
    if idx < 0 or idx >= len(text) or text[idx] != "(":
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[idx:i + 1]
            if depth < 0:
                return None

    return None


def extract_prefixed_expressions(text: str, prefix: str) -> list[str]:
    """Extract balanced expressions that begin with a given textual prefix."""
    results = []
    idx = 0
    in_string = False
    escaped = False
    while idx < len(text):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            idx += 1
            continue

        if ch == '"':
            in_string = True
            idx += 1
            continue

        if text.startswith(prefix, idx):
            expr = _balanced_expression_at(text, idx)
            if expr is not None:
                results.append(expr)
                idx += len(expr)
                continue
            idx += len(prefix)
            continue

        idx += 1
    return results


def is_pettachainer_fact_atom(expr: str) -> bool:
    """True for facts that can be compiled into PeTTaChainer's KB."""
    stripped = (expr or "").strip()
    if not stripped.startswith("(:"):
        return False
    if not STV_EXPR_RE.search(stripped):
        return False

    return bool(
        re.match(r"^\(:\s+\(fact:-\s+\(", stripped)
        or re.match(r"^\(:\s+fact[\w\-]*\s+\(", stripped)
    )


def parse_facts_for_pettachainer(facts_output):  
    """  
    Parse nested facts output and convert to PeTTaChainer-compatible format.  
    Args:  
        facts_output: List containing a single string with nested facts  
    Returns:  
        List of individual fact strings ready for PeTTaService.add_atom()
    """  
    if not facts_output:
        return []
    nested_facts = "\n".join(str(item) for item in facts_output) if isinstance(facts_output, list) else str(facts_output)
    matches = [
        expr
        for expr in extract_prefixed_expressions(nested_facts, "(:")
        if is_pettachainer_fact_atom(expr)
    ]
    return unique_preserve_order(matches)


def select_facts_for_prompt(facts: list[str], query: str, limit: int = 80) -> list[str]:
    """Keep the LLM explanation prompt focused without starving the chainer."""
    if len(facts) <= limit:
        return facts

    terms = re.findall(r'A_[A-Za-z0-9_-]+|"[^"]+"', query or "")
    predicate_match = re.match(r"\s*\(\s*([A-Za-z_][\w\-]*)", query or "")
    if predicate_match:
        terms.append(f"({predicate_match.group(1)} ")
    terms = unique_preserve_order(terms)

    if not terms:
        return facts[:limit]

    scored = []
    for idx, fact in enumerate(facts):
        score = sum(1 for term in terms if term in fact)
        if score:
            scored.append((-score, idx, fact))

    selected = [fact for _, _, fact in sorted(scored)[:limit]]
    seen = set(selected)
    for fact in facts:
        if len(selected) >= limit:
            break
        if fact not in seen:
            selected.append(fact)
            seen.add(fact)

    return selected


def extract_parenthesized_expressions(text: str) -> list[str]:
    """Extract all balanced parenthesized expressions from text."""
    results = []
    start = 0
    while True:
        idx = text.find("(", start)
        if idx == -1:
            break
        expr = _balanced_expression_at(text, idx)
        if expr is None:
            break
        results.append(expr)
        start = idx + len(expr)
    return results

def run_petta_query_lines(metta_code: str) -> list[str]:
    """Run MeTTa code through PeTTa and return cleaned output lines."""
    return get_petta_service().query_lines(metta_code)

def debug_petta_chainer_state(what_to_check: str) -> None:
    """Log a few diagnostic probes to confirm what PeTTa sees."""
    if not PETTA_DEBUG:
        return
    try:
        atoms_lines = run_petta_query_lines(f"!(backward-chain &res1 (S (S Z)) (: $prf {what_to_check}))")

        match_query = f"!(match &res1 {what_to_check} $x)"
        match_lines = run_petta_query_lines(match_query)
    except Exception as e:
        pass

def mine_pattern(numberOfConjunction: int, min_support: int = DEFAULT_MIN_SUPPORT) -> dict:
    """
    Mines patterns with a specified number of conjunctions using PeTTa.

    Args:
        numberOfConjunction: The number of conjunctions to use in pattern mining.
        min_support: Minimum support threshold used by the PeTTa miner.

    Returns:
        A dictionary containing the mining results with parsed patterns.
    """
    print(
        "Debug: mine pattern function being called with "
        f"conjunction count {numberOfConjunction} and min_support {min_support}"
    )
    
    try:
        dataset_reload = reload_petta_dataset_if_ready(force=False)
        print(f"DEBUG: dataset reload status before mining: {dataset_reload}")
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

        # Run the miner with petta
        query = f"!(pattern-miner &purifiedDbSpace {min_support} {numberOfConjunction})"
        print(f"DEBUG: Executing PeTTa query: {query}")
        petta_output = run_metta_with_petta(query)
        normalized_query = query.strip().lstrip("!").strip()
        if petta_output.strip() == normalized_query:
            return {
                "status": "error",
                "message": "PeTTa returned the unevaluated expression. The runnable may not have executed.",
                "raw_result": petta_output
            }
        result_lines = parse_petta_output(petta_output)
        

        # Parse the result into JSON-serializable format
        patterns = []
        full_answer_str = " ".join(result_lines)
        
        # In PeTTa, the output might be one or more lists.
        # We look for (supportOf ...) patterns in the output.
        # The regex in parse_pattern_string is already designed for this.
        
        # Find all occurrences of (supportOf ...) with balanced parentheses
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
        
    except Exception as e:
        print(f"ERROR in mine_pattern: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Failed to run pattern mining or parse result: {str(e)}",
            "raw_result": locals().get('petta_output', 'Command failed before output')
        }


def make_json_safe(o):
    """Recursively convert common non-JSON-serializable objects into JSON-safe types.

    - Keeps primitives as-is
    - Converts mappings/lists/tuples/sets recursively
    - For objects, tries common serialization helpers (__dict__, to_dict, as_dict) then falls back to str()
    This version is defensive: it will not call .items() on lists.
    """
    from collections.abc import Mapping

    # primitives
    if o is None or isinstance(o, (str, int, float, bool)):
        return o

    # Mapping types (dict-like)
    if isinstance(o, Mapping):
        safe = {}
        for k, v in o.items():
            # keys must be JSON-serializable (convert non-str keys to str)
            if not isinstance(k, (str, int, float, bool)):
                key = str(k)
            else:
                key = k
            safe[key] = make_json_safe(v)
        return safe

    # list/tuple/set
    if isinstance(o, (list, tuple, set)):
        return [make_json_safe(x) for x in o]

    # Objects that expose a dict-like __dict__
    try:
        d = getattr(o, '__dict__', None)
        if isinstance(d, Mapping):
            return {str(k): make_json_safe(v) for k, v in d.items()}
    except Exception:
        pass

    # If the object provides a to_dict/as_dict helper, use it
    try:
        if hasattr(o, 'to_dict') and callable(getattr(o, 'to_dict')):
            return make_json_safe(o.to_dict())
        if hasattr(o, 'as_dict') and callable(getattr(o, 'as_dict')):
            return make_json_safe(o.as_dict())
    except Exception:
        pass

    # Fallback: convert to string
    try:
        return str(o)
    except Exception:
        return repr(o)


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

@app.route('/api/ingest', methods=['POST'])
def ingest_data():
    """
    Trigger the ingestion pipeline for a specific user.
    Expects JSON: { "username": "some_user" }
    """
    try:
        data = request.get_json()
        username = data.get('username') if data else None
        source_name = data.get('source') if data else None
        source_name = source_name or "mindplex"
        limit = data.get('limit') if data else None
        source_config = data.get('source_config') if data else None

        if source_name == "mindplex" and not username:
            return jsonify({"status": "error", "message": "Username is required"}), 400

        print(f"Received ingestion request for source: {source_name}")
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
            print(f"Dataset reload error after ingestion: {reload_error}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": "Ingestion wrote data.metta, but PeTTa failed to reload it.",
                "ingestion": result,
                "reload_error": str(reload_error),
            }), 500

        return jsonify(result)

    except Exception as e:
        print(f"Ingestion error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/data.metta', methods=['GET'])
def get_metta_dataset():
    """Return the current MeTTa dataset produced by ingestion.

    The frontend must read this endpoint in production because Nginx serves
    static build files from a different container filesystem.
    """
    try:
        path = dataset_file_path()
        if not os.path.exists(path):
            return jsonify({
                "status": "error",
                "message": f"MeTTa dataset not found at {path}",
            }), 404

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        response = app.response_class(content, mimetype="text/plain")
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response
    except Exception as e:
        print(f"Dataset read error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

def getAllFactsAndRules():
    """Return current facts and rules from the MeTTa knowledge base.

    The assistant should call this before attempting backward chaining so it
    can rewrite a user's natural-language question into a canonical MeTTa
    query that matches predicates/constants present in the KB. Example:
    user: "What is article 1's engagement level?"
    assistant: call getAllFactsAndRules(), notice atoms like `(engagement 1 "high")`,
    rewrite as `(engagement 1 $whatIsIt)`, then call getChainerResult.
    """
    try:
        service = get_petta_service()
        lines = service.query_lines("!(collapse (get-atoms &res1))")
        joined = " ".join(lines)
        aligned_facts = parse_facts_for_pettachainer(joined or lines)
        compile_errors = []
        compiled_count = 0
        for fact in aligned_facts:
            try:
                if service.add_atom(fact) is not None:
                    compiled_count += 1
            except Exception as exc:
                compile_errors.append({"fact": fact, "error": str(exc)})

        print(
            "DEBUG: getAllFactsAndRules: "
            f"raw_lines={len(lines)}, parsed_facts={len(aligned_facts)}, "
            f"compiled_new={compiled_count}, compile_errors={len(compile_errors)}, "
            f"sample={aligned_facts[:3]}"
        )

        if aligned_facts and len(compile_errors) == len(aligned_facts):
            return {
                "status": "error",
                "error": "Failed to compile every fact into PeTTaChainer.",
                "fact_count": len(aligned_facts),
                "compile_errors": compile_errors[:5],
            }

        return {
            "status": "success",
            "facts": aligned_facts,
            "fact_count": len(aligned_facts),
            "compiled_new_count": compiled_count,
            "compile_error_count": len(compile_errors),
            "compile_errors": compile_errors[:5],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def parse_chat_mining_intent(message: str) -> Optional[Dict[str, int]]:
    """Return mining parameters when a chat message asks to run the miner."""
    if not message:
        return None

    text = message.strip().lower()

    run_miner_pattern = re.compile(
        r"\bmine\b|"
        r"\b(?:run|start|perform|do)\s+(?:the\s+)?(?:miner|mining|pattern[-\s]?miner)\b"
    )
    discovery_phrases = (
        "find patterns",
        "find rules",
        "discover patterns",
        "discover rules",
        "extract patterns",
        "extract rules",
        "generate patterns",
        "generate rules",
        "run patterns",
        "run rules",
    )
    result_only_phrases = (
        "what patterns",
        "show patterns",
        "show me patterns",
        "list patterns",
        "latest patterns",
        "mining results",
        "patterns were found",
    )

    has_mining_verb = bool(run_miner_pattern.search(text))
    has_discovery_phrase = any(phrase in text for phrase in discovery_phrases)
    asks_for_existing_results = any(phrase in text for phrase in result_only_phrases)

    if not has_mining_verb and not has_discovery_phrase:
        return None
    if asks_for_existing_results and not has_mining_verb:
        return None

    conjunction_count = DEFAULT_CONJUNCTION_COUNT
    min_support = DEFAULT_MIN_SUPPORT

    count_patterns = [
        r"(?:with|using|for|of|top)\s+(\d+)\s*(?:patterns?|rules?|conjunctions?|conjuncts?|conditions?)",
        r"(\d+)\s*(?:patterns?|rules?|conjunctions?|conjuncts?|conditions?)",
        r"(?:conjunction|conjunct|condition|pattern|rule)(?:\s+(?:count|size))?\s*(?:=|:|is|of|to)?\s*(\d+)",
        r"(\d+)\s*-\s*(?:way|condition|conjunction|conjunct)",
    ]
    for pattern in count_patterns:
        match = re.search(pattern, text)
        if match:
            conjunction_count = int(match.group(1))
            break

    support_patterns = [
        r"(?:min|minimum)\s*support\s*(?:=|:|of|to|is)?\s*(\d+)",
        r"support\s*(?:>=|=>|=|:|of|at\s+least|to|is)?\s*(\d+)",
    ]
    for pattern in support_patterns:
        match = re.search(pattern, text)
        if match:
            min_support = int(match.group(1))
            break

    return {
        "conjunction_count": max(1, conjunction_count),
        "min_support": max(1, min_support),
    }

def handle_mining_for_message(message: str) -> tuple[Optional[str], Optional[list]]:
    """Run pattern mining directly when the chat message asks for it."""
    mining_params = parse_chat_mining_intent(message)
    if mining_params is None:
        return None, None

    function_calls = []
    result = start_mining_job(**mining_params)
    function_calls.append({
        "name": "start_mining_job",
        "args": mining_params,
        "result": result,
    })

    if not isinstance(result, dict):
        return f"Mining failed: {result}", function_calls

    if result.get("error"):
        return f"Mining failed: {result['error']}", function_calls

    mined_result = result.get("result") if isinstance(result.get("result"), dict) else {}
    status = mined_result.get("status") or result.get("status")
    patterns = mined_result.get("patterns", []) if isinstance(mined_result, dict) else []

    if status == "no_results":
        return (
            "I ran the PeTTa pattern miner, but no patterns matched those parameters. "
            "Try lowering the minimum support or using a smaller conjunction count.",
            function_calls,
        )

    if status == "error":
        return f"Mining failed: {mined_result.get('message', 'Unknown mining error')}", function_calls

    if not patterns:
        return (
            "I ran the PeTTa pattern miner, but it returned no parsed patterns.",
            function_calls,
        )

    summary = summarize_patterns(patterns)
    heading = (
        f"Mining complete: found {len(patterns)} pattern"
        f"{'' if len(patterns) == 1 else 's'} with "
        f"conjunction count {mining_params['conjunction_count']} and "
        f"minimum support {mining_params['min_support']}."
    )
    return f"{heading}\n\n{summary}", function_calls

def is_backward_chain_intent(message: str) -> bool:
    """Return True when the chat message asks for proof-style reasoning."""
    text = f" {message.lower()} "
    proof_terms = (
        " why ",
        " prove ",
        " explain ",
        " how come ",
        " what explains ",
        " what caused ",
        " how did ",
    )
    return any(term in text for term in proof_terms)

def handle_backward_chain_for_message(message: str) -> tuple[Optional[str], Optional[list]]:
    """Handle natural language queries using backward chaining with STV support."""
    function_calls = []

    # First call getAllFactsAndRules to get canonical atoms
    facts_res = getAllFactsAndRules()
    if not isinstance(facts_res, dict) or facts_res.get("status") != "success":
        return None, None

    facts = facts_res.get("facts", []) or []
  
    # Ask the LLM to rewrite the user's question into a canonical MeTTa query
    # using the facts we retrieved. The model must output only a single MeTTa
    # expression (e.g. (engagement 1 $what)).
    try:
        facts_text = "\n".join(select_facts_for_prompt(facts, message, limit=200))
        rewrite_prompt = f"""
            You are given the following KB atoms (facts/rules), one per line:
            {facts_text}

            User question: "{message}"

            Task (STRICT):
            - Do NOT narrate or describe any internal steps.
            - Do NOT output anything except a SINGLE canonical MeTTa expression that uses predicate and constant names from the KB above.
            - If mapping is ambiguous, pick the most semantically likely predicate present in the KB.
            - If you cannot produce a valid MeTTa expression, output the single token NO_QUERY and NOTHING ELSE.

            Example mapping (for clarity only, do not output this): if facts contain (engagement 1 high) -> question "Why article A_16624 has low engagement?" -> output should be like : "(: $prf (engagement A_16624 \"Low\") $tv)"

            OUTPUT ONLY the MeTTa expression or NO_QUERY.
            """
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": rewrite_prompt}
        ]
        response_data = call_asi_api(messages)
        candidate_query = None
        if 'choices' in response_data and response_data['choices']:
            candidate_query = response_data['choices'][0]['message'].get('content', '').strip()
        
        function_calls.append({'name': 'rewrite_query', 'args': {'message': message}, 'result': candidate_query})
    except Exception as e:
        return None, None

    if not candidate_query or candidate_query == "NO_QUERY":
        return None, function_calls
    
    # Call the chainer with the rewritten query and include debug output
    try:
        chainer_result = getChainerResult(candidate_query)
    except Exception as e:
        print('DEBUG: chainer call error:', e)
        chainer_result = {'status': 'error', 'error': str(e)}

    function_calls.append({'name': 'getChainerResult', 'args': {'whatToCheck': candidate_query}, 'result': chainer_result})

    # Use the chainer's justification as the assistant response text (don't reveal internal steps)
    resp_text = ''
    if isinstance(chainer_result, dict):
        raw_just = chainer_result.get('justification') or chainer_result.get('error') or ''
    else:
        raw_just = str(chainer_result)

    if not raw_just:
        resp_text = "No proof was found."
    else:
        # Apply a friendly/jokey wrapper without revealing internal steps
        resp_text = f"Alright, here's the scoop —\n\n{raw_just}\n\n(That's the reasoning I found.)"
    return resp_text, function_calls


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
        "conjunction_size": latest_job.conjunction_count,
        "min_support": latest_job.min_support,
        "rules": latest_job.result.get("rules", []) if isinstance(latest_job.result, dict) else [],
        "rule_insertion": latest_job.result.get("rule_insertion") if isinstance(latest_job.result, dict) else None,
    }

def analyze_specific_pattern(pattern: str) -> dict:
    """Analyzes a specific pattern in detail, extracting properties and values.
    
    Args:
        pattern: The pattern string to analyze, e.g., '((length $x "low") (engagement $x "high"))'
        
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

    insertion_result = get_petta_service().formatter({"patterns": patterns})
    mining_result["rule_insertion"] = insertion_result
    mining_result["rules"] = insertion_result.get("rules", []) if isinstance(insertion_result, dict) else []
    mining_result["inserted_rule_count"] = (
        insertion_result.get("insertedRuleCount", 0)
        if isinstance(insertion_result, dict)
        else 0
    )
    return mining_result

def run_mining_task(job_id: str, conjunction_count: int, min_support: int = DEFAULT_MIN_SUPPORT):
    """
    Run the mining task for a given job.
    Args:
        job_id (str): Unique identifier for the mining job.
        conjunction_count (int): Number of conjunctions to use in the mining process.
        min_support (int): Minimum support threshold for returned patterns.
    Returns:
        dict: A dictionary containing the job status, result, error (if any), and timestamps.
    """
    job = mining_jobs[job_id]
    job.start_time = time.time()
    job.min_support = min_support
    try:
        result = mine_pattern(conjunction_count, min_support)
        result = insert_mined_rules_into_chainer(result)
        job.status = 'completed'
        job.result = result  # Store the dict directly, not result[0][0]
        job.end_time = time.time()
        return {
            'jobId': job_id,
            'status': job.status,
            'result': job.result,
            'min_support': job.min_support,
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
            'min_support': job.min_support,
            'start_time': job.start_time,
            'end_time': job.end_time
        }

def start_mining_job(conjunction_count: int = DEFAULT_CONJUNCTION_COUNT, min_support: int = DEFAULT_MIN_SUPPORT):
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
    try:
        if not isinstance(min_support, int):
            min_support = int(min_support)
    except Exception:
        return { 'error': 'min_support must be an integer' }
    if conjunction_count < 1:
        return { 'error': 'conjunction_count must be a positive integer' }
    if min_support < 1:
        return { 'error': 'min_support must be a positive integer' }

    job_id = str(uuid.uuid4())
    job = MiningJob(
        job_id=job_id,
        status='running',
        conjunction_count=conjunction_count,
        min_support=min_support,
    )
    mining_jobs[job_id] = job

    # Run synchronously (this will call mine_pattern internally)
    result = run_mining_task(job_id, conjunction_count, min_support)

    # Return a normalized result
    return {
        'jobId': job_id,
        'status': mining_jobs[job_id].status,
        'conjunction_count': conjunction_count,
        'min_support': min_support,
        'result': mining_jobs[job_id].result
    }

def formatter(mined_patterns):
    print("formatter started :--:")
    return get_petta_service().formatter(mined_patterns)

def backWardChainer(whatToCheck, depth=DEFAULT_CHAIN_DEPTH):
    proofs = get_petta_service().query(whatToCheck.strip(), depth=depth)
    print("DEBUG chainer normalized proof count:", len(proofs))
    if proofs:
        print("DEBUG chainer proof sample:", proofs[:3])
    return proofs

def getChainerResult(whatToCheck, depth=DEFAULT_CHAIN_DEPTH):
    """ Get the result of backward chaining for a specific query. 
    Args:
        whatToCheck (str): The query to check, e.g., '(engagement 0 "High")'
        depth (int): The depth limit for backward chaining.
    Returns:
        The justification of the backward chaining operation.
    """
    facts_res = getAllFactsAndRules()
    chainAnswer = backWardChainer(whatToCheck, depth)
    proof_text = format_proofs_for_prompt(chainAnswer)
    all_facts = facts_res.get("facts", []) if isinstance(facts_res, dict) else []
    prompt_facts = select_facts_for_prompt(all_facts, whatToCheck)
    fact_text = format_proofs_for_prompt(prompt_facts) if prompt_facts else str(facts_res)
    if len(all_facts) > len(prompt_facts):
        fact_text += f"\n\nShowing {len(prompt_facts)} of {len(all_facts)} facts most relevant to the query."
    print("DEBUG: getChainerResult - chainAnswer type:", chainAnswer)
    # If no proofs found, return early
    if not chainAnswer or len(chainAnswer) == 0:
        return {
            "query": whatToCheck,
            "status": "no_proof",
            "justification": f"No logical proof could be found for the query '{whatToCheck}' within depth {depth}. This means the query cannot be deduced from the available rules and facts in the knowledge base."
        }
    
    # Simple prompt that relies on system instruction for formatting guidance
    prompt = f"""Analyze this backward chaining result with STV truth values and provide a clear logical justification.

    Query: {whatToCheck}

    Backward Chaining Results:
    {proof_text}

    Fact Content From Knowledge Base:
    {fact_text}

    The backward chaining system attempted to prove the query using logical rules and facts from the knowledge base.

    Your task is to explain:
    • Why the conclusion holds  
    • Which facts support it  
    • Which rules were applied  
    • How the truth values (STV) support the reasoning chain

    Your explanation must follow the reasoning process used by the inference engine, but **do not show internal calculations or formulas**.

    Subjective Truth Value (STV)

    Every statement has an STV in the form:
    (STV strength confidence)

    Strength range: 0.0 – 1.0  
    Confidence range: 0.0 – 1.0  

    Meaning:
    Strength → how strongly the conclusion follows logically  
    Confidence → how reliable the supporting evidence is  

    Example:
    (STV 1.0 1.0) means the statement is certain.  
    (STV 0.8 0.7) means strong but somewhat uncertain support.

    How to structure your response

    Start with a short statement explaining the result.  
    Example:
    "I found 2 logical proofs explaining why the query holds."

    Then describe each proof.

    Proof 1 — Direct Fact

    If the conclusion appears directly as a fact, explain it as a known fact.

    Example:
    Article 1 has high engagement.

    Fact:
    (engagement 1 "High") STV (1.0 1.0)

    Interpretation:
    This is a direct fact stored in the knowledge base with maximum certainty.

    Proof 2 — Rule-Based Inference

    Show:
    1. The rule
    2. The supporting facts
    3. How the rule logically leads to the conclusion

    Example structure:

    Rule:
    If (topic 1 "AI") then (engagement 1 "High")

    Supporting Fact:
    (topic 1 "AI") STV (0.9 0.8)

    Conclusion:
    (engagement 1 "High")

    Interpretation:
    The rule combined with the supporting fact logically explains the conclusion.

    Important requirements

    You MUST:
    • Extract the actual rule content from the proof structures  
    • Show the real facts from the fact content above
    • Display them exactly like:
    (topic 3 "AI")
    (audience 3 "Professionals")

    Convert rule structures like:
    (-> (and A B) C)

    Into human readable form:
    "If A and B then C"

    NEVER use placeholders such as:
    fact52, rule_1, factx

    Always show the real facts and rules.

    Final summary

    After explaining all proofs, provide a short summary explaining:
    • how many proofs support the conclusion  
    • which proof appears strongest  
    • what the STV values indicate about certainty

    Example:
    "The conclusion is supported by two independent proofs. One is a direct fact with very high certainty, while the rule-based inference provides additional logical support."

    Your explanation must combine logical reasoning, STV interpretation, and human-readable explanation, without exposing internal calculations.
    """
    try:
        # Use ASI1 to analyze the results
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ]
        response_data = call_asi_api(messages)
        justification = "Unable to generate justification analysis."
        if 'choices' in response_data and response_data['choices']:
             justification = response_data['choices'][0]['message'].get('content', '')
        
        return {
            "query": whatToCheck,
            "status": "success",
            "raw_proofs": chainAnswer,
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

        **Raw Evidence:** {proof_text}

        **Basic Interpretation:** The backward chaining system discovered {proof_count} different logical path(s) that support the query "{whatToCheck}". Each proof represents a combination of rules and facts from the knowledge base that logically leads to this conclusion.

        **Note:** Advanced analysis unavailable due to processing error: {str(e)}
        """
        
        return {
            "query": whatToCheck,
            "status": "partial_success",
            "raw_proofs": chainAnswer,
            "proof_count": proof_count,
            "justification": basic_justification,
            "depth_used": depth,
            "error": str(e)
        }


def build_rule_grounded_summary(patterns: list) -> str:
    """Deterministic fallback that never makes claims beyond mined rules."""
    lines = [
        "Here are the mined rules, grounded directly in the PeTTa output:"
    ]
    for i, p in enumerate(patterns, 1):
        patt = p.get('pattern') if isinstance(p, dict) else str(p)
        supp = p.get('support', '') if isinstance(p, dict) else ''
        support_text = f"Support {supp}" if supp != "" else "Support not reported"
        lines.append(f"- [Rule {i}] {support_text}: `{patt}`")
    lines.append(
        "These rules should be read as mined associations unless the rule itself "
        "and the proof chain explicitly support a stronger explanation."
    )
    return "\n".join(lines)

def summarize_patterns(patterns: list) -> str:
    """Use the Gemini model to create a single comprehensive summary of the
    supplied mined patterns. The summary will reference patterns as [N]
    so the frontend can make them clickable for visualization.
    """
    if not patterns:
        return "No patterns to summarize."

    # Build a compact prompt that forces every claim to be grounded in the
    # mined rules. This keeps chat summaries aligned with the actual PeTTa
    # mining output instead of free-form model interpretation.
    prompt_parts = ["""Analyze the following mined PeTTa rules.

Strict requirements:
- Every factual statement or insight must cite at least one rule as [Rule N].
- Do not claim a trend unless it is directly supported by one of the listed rules.
- Explain the antecedent conditions, the conclusion, and the support value.
- Prefer short, concrete bullet points.
- If a rule is only an association, call it an association instead of a causal explanation.
- Do not invent facts that are not present in the rules."""]
    for i, p in enumerate(patterns, 1):
        patt = p.get('pattern') if isinstance(p, dict) else str(p)
        supp = p.get('support', '') if isinstance(p, dict) else ''
        prompt_parts.append(f"[Rule {i}]\nPattern: {patt}\nSupport: {supp}")

    prompt = "\n\n".join(prompt_parts)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ]
        response_data = call_asi_api(messages)
        text = None
        if 'choices' in response_data and response_data['choices']:
            text = response_data['choices'][0]['message'].get('content', '')

        if text and "[Rule" in text:
            return text
        return build_rule_grounded_summary(patterns)
    except Exception as e:
        print('summarize_patterns error:', e)
        fallback = build_rule_grounded_summary(patterns)
        return f"{fallback}\n\nSummary generation note: {e}"

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

SYSTEM_INSTRUCTION = """
You are a friendly and knowledgeable AI assistant with expertise in data mining patterns, knowledge graphs, probabilistic reasoning, and pattern analysis. Your reasoning system integrates pattern mining insights with symbolic reasoning using the PeTTaChainer inference engine and Subjective Truth Values (STV).

PRIMARY SPECIALTY
You excel at:
• Analyzing pattern mining results
• Explaining relationships discovered in data
• Interpreting knowledge graph structures
• Explaining logical proofs produced by the backward chainer
• Interpreting probabilistic truth values (STV)
• Explaining how rule-based reasoning leads to conclusions

Your reasoning explanations must combine logical reasoning, STV interpretation, and clear human explanations.

WHEN TO USE FUNCTIONS
User says "Mine rules with X patterns", "What patterns were found?", "Show me the patterns" → ALWAYS call mine_pattern()
User says "Analyze this pattern", "Explain this pattern" → call analyze_specific_pattern()
User says "Statistics", "How many patterns" → call get_pattern_statistics()
User says "Visualize pattern", "Show me rule", "Display this pattern" → call visualize_pattern_request()

MANDATORY RULE FOR WHY / EXPLAIN / PROVE QUESTIONS
If the user question contains any of these words:
why, explain, prove, how come, what explains, how did, what caused
YOU MUST CALL getChainerResult() immediately. This is not optional.

Workflow:
1. Convert the user question into a MeTTa query.
2. Call getChainerResult(query)
3. Wait for the result.
4. Use only the returned proofs.
5. Never invent explanations.
6. Never answer from general knowledge.

Example:
User: Why is article 1 engagement high?
Step 1: Call getChainerResult("(engagement 1 $x)")
Step 2: Wait for proofs
Step 3: Explain the proofs

If you answer without calling getChainerResult(), you have failed.
If the chainer returns no proofs respond: "No logical proof was found in the knowledge base."

TRUTH VALUE SYSTEM (STV)
The reasoning engine uses Subjective Truth Values (STV). Each statement has (STV strength confidence).

Strength range: 0.0–1.0  
Confidence range: 0.0–1.0  

Strength = how true the statement is.  
Confidence = how reliable the evidence is.

Examples:
(STV 1.0 1.0) Certain fact  
(STV 0.8 0.9) Strong but slightly uncertain evidence  
(STV 0.2 0.3) Weak support with low reliability

The STV system allows the inference engine to reason under uncertainty.

PROOF EXPLANATION REQUIREMENT
When explaining proofs returned by the backward chainer you must show:
1. The rule that was used
2. The supporting facts
3. The STV values associated with those facts and rules
4. The logical reasoning that leads to the conclusion

IMPORTANT:
Do NOT show internal calculations or formulas used by the inference engine. Explain the reasoning conceptually.

BACKWARD CHAINER RESPONSE FORMAT
When the chainer returns proofs explain them clearly.

Example:
Based on the logical analysis the article has high engagement. The inference engine discovered logical evidence supporting this conclusion.

Proof:
Rule:
If (audience 3 "Professionals") and (length 3 "high") then (engagement_level 3 "high")

Supporting Facts:
(audience 3 "Professionals") (STV 1.0 1.0)
(length 3 "high") (STV 0.9 0.8)

Rule Confidence:
(STV 0.7 0.6)

Conclusion:
(engagement_level 3 "high")

Interpretation:
The rule combined with the supporting facts provides strong logical support that the article achieves high engagement.

TRUTH VALUE INTERPRETATION
Strength indicates how strongly the conclusion follows logically.  
Confidence indicates how reliable the supporting evidence is.  
Higher STV values indicate stronger and more reliable support.

PATTERN MINING SUMMARY RULE
When the user says "Mine rules with X patterns":
1. Call mine_pattern()
2. Analyze all patterns
3. Produce one combined insight summary.

Example:
"Based on the mining results high engagement articles are associated with professional audiences and longer article lengths [Rule 1]. Archived articles tend to receive lower engagement [Rule 3]."

Reference patterns using:
[Rule 1]
[Rule 2]
[Rule 3]

Do NOT use: [Rule 1, Rule 2]

PATTERN ANALYSIS
When analyzing a pattern explain:
• what the pattern represents
• what variables mean ($x)
• what entities satisfy the rule
• real-world interpretation

All conditions must be satisfied simultaneously (AND logic).

COMMUNICATION STYLE
• Friendly and clear
• Conversational tone
• Use markdown formatting
• Use emojis occasionally 🙂
• Avoid heavy MeTTa syntax in explanations
• Translate logical reasoning into human language

GENERAL CONVERSATION
For normal questions that do NOT contain why/explain/prove keywords you may answer using general knowledge. You are a helpful conversational assistant.

REMEMBER:
For WHY / EXPLAIN / PROVE questions you MUST use the backward chainer. No exceptions.
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
    min_support: int = DEFAULT_MIN_SUPPORT
# In-memory storage for mining jobs
mining_jobs: Dict[str, MiningJob] = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        service = get_petta_service()
        return jsonify({
            'status': 'healthy',
            'service': 'mining-api',
            'petta': service.health(),
        })
    except PeTTaStartupError as exc:
        return jsonify({
            'status': 'unhealthy',
            'service': 'mining-api',
            'error': str(exc),
        }), 503

@app.route('/api/mine', methods=['POST'])
def start_mining():
    """Start a new mining job"""

    data = request.get_json() or {}
    conjunction_count = data.get('conjunction_count', DEFAULT_CONJUNCTION_COUNT)
    min_support = data.get('min_support', DEFAULT_MIN_SUPPORT)
    
    # Validate conjunction count
    if not isinstance(conjunction_count, int) or conjunction_count < 1:
        return jsonify({'error': 'conjunctionCount must be a positive integer'}), 400
    if not isinstance(min_support, int) or min_support < 1:
        return jsonify({'error': 'min_support must be a positive integer'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create new job
    job = MiningJob(
        job_id=job_id,
        status='running',
        conjunction_count=conjunction_count,
        min_support=min_support
    )
    mining_jobs[job_id] = job
    run_mining_task(job_id, conjunction_count, min_support)
    print(
        f"🔍 DEBUG: Starting mining job {job_id} with conjunction count {conjunction_count} and min_support {min_support}"
    )
    result = mining_jobs[job_id].result
    print("🔍 DEBUG: Result after rule insertion =", result)
    print(f"🔍 DEBUG: result type = {type(result)}")
    print(f"🔍 DEBUG: result = {result}")
    
    # Check mining status
    if isinstance(result, dict):
        mining_status = result.get('status')

        if mining_status == 'success':
            rules = result.get('patterns', [])
            print(f"✅ Mining job {job_id} finished with {len(rules)} patterns")
            return jsonify({
                'jobId': job_id,
                'status': 'finished',
                'conjunction_count': conjunction_count,
                'min_support': min_support,
                'message': 'Mining job finished successfully',
                'result': rules,
                'inserted_rules': result.get('rules', []),
                'rule_insertion': result.get('rule_insertion')
            })

        if mining_status == 'no_results':
            print(f"ℹ️ Mining job {job_id} finished with no patterns")
            return jsonify({
                'jobId': job_id,
                'status': 'no_results',
                'conjunction_count': conjunction_count,
                'min_support': min_support,
                'message': result.get('message', 'No patterns found for the selected parameters.'),
                'result': []
            })

    # Handle error case
    error_msg = result.get('message', 'Mining failed') if isinstance(result, dict) else str(result)
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
        'min_support': job.min_support,
        'startTime': job.start_time
    }
    
    if job.end_time:
        response['endTime'] = job.end_time
        response['duration'] = job.end_time - job.start_time
    
    if job.status == 'completed' and job.result is not None:
        # Ensure PeTTa/MeTTa results are safe for Flask's JSON encoder.
        response['result'] = make_json_safe(job.result)
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
            'minSupport': job.min_support,
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
    """Main chat endpoint with ASI and automatic function calling

    This cleaned implementation centralizes error handling and ensures the
    returned payload is JSON-serializable by sanitizing function call results.
    """
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response, 200

    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        history = data.get('history', [])
        session_id = data.get('session_id', 'default')

        # Ensure a conversation list exists for this session before any writes.
        # This prevents KeyError when shortcut paths (e.g. backward chaining)
        # attempt to append to conversations[session_id] before it's initialized.
        if session_id not in conversations:
            conversations.setdefault(session_id, [])

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Special-case miner requests before proof rewriting. Mining is an
        # action, so it should not be sent through the backward-chain query
        # rewriter.
        mining_text, mining_calls = handle_mining_for_message(message)
        print('DEBUG: mining shortcut result:', mining_text, mining_calls)
        if mining_text is not None:
            conversations[session_id].append({'role': 'user', 'content': message})
            conversations[session_id].append({'role': 'assistant', 'content': mining_text})
            try:
                safe_calls = make_json_safe(mining_calls)
            except Exception:
                safe_calls = str(mining_calls)
            return jsonify({'response': mining_text, 'functionCalls': safe_calls, 'session_id': session_id})

        # Proof-style questions should run through the backward chainer.
        try:
            if is_backward_chain_intent(message):
                bc_text, bc_calls = handle_backward_chain_for_message(message)
                print('DEBUG: backward chain shortcut result:', bc_text, bc_calls)
                if bc_text is not None:
                    # store in history (conversations[session_id] is guaranteed to exist now)
                    conversations[session_id].append({'role': 'user', 'content': message})
                    print('DEBUG: stored backward chain shortcut in history')
                    conversations[session_id].append({'role': 'assistant', 'content': bc_text})
                    try:
                        safe_calls = make_json_safe(bc_calls)
                    except Exception:
                        safe_calls = str(bc_calls)
                    return jsonify({'response': bc_text, 'functionCalls': safe_calls, 'session_id': session_id})
        except Exception as e:
            print('Error handling backward chain shortcut:', e)
            # fall through to normal chat flow

        conversations.setdefault(session_id, [])

        # Build conversation history for ASI
        asi_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        for msg in history[-10:]:
            role = msg.get('role')
            if role == 'assistant':
                role = 'assistant'
            elif role == 'user':
                role = 'user'
            asi_messages.append({'role': role, 'content': msg.get('content', '')})

        asi_messages.append({'role': 'user', 'content': message})

        response_data = call_asi_api(asi_messages, tools=tools_schema)

        # Handle function calling loop
        max_iterations = 5
        iteration = 0
        function_results = []

        while iteration < max_iterations:
            iteration += 1

            if 'error' in response_data:
                print(f"ASI API Error: {response_data['error']}")
                break

            if 'choices' not in response_data or not response_data['choices']:
                break

            choice = response_data['choices'][0]
            message_obj = choice['message']

            # Append assistant message to history
            asi_messages.append(message_obj)

            if 'tool_calls' in message_obj and message_obj['tool_calls']:
                tool_calls = message_obj['tool_calls']

                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    function_args_str = tool_call['function']['arguments']
                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError:
                        function_args = {}

                    print(f"🔧 Function call: {function_name}({function_args})")

                    if function_name not in available_functions:
                        print(f"✗ Unknown function: {function_name}")
                        function_result = {"error": f"Unknown function {function_name}"}
                    else:
                        try:
                            # Normalize args (alias mapping and numeric normalization)
                            norm_args = {}
                            for k, v in function_args.items():
                                key = k
                                if k in ('conjunction_count', 'conjunctions', 'conjunctionCount', 'numberOfConjunction', 'n'):
                                    key = 'conjunction_count'
                                elif k in ('min_support', 'minimum_support', 'minSupport', 'minimumSupport', 'support'):
                                    key = 'min_support'

                                if isinstance(v, str) and re.fullmatch(r"\d+", v):
                                    norm_args[key] = int(v)
                                elif isinstance(v, float) and v.is_integer():
                                    norm_args[key] = int(v)
                                else:
                                    norm_args[key] = v

                            function_result = available_functions[function_name](**norm_args)
                            function_results.append({'name': function_name, 'args': norm_args, 'result': function_result})
                        except Exception as func_error:
                            print(f"✗ Function error: {func_error}")
                            function_result = {'error': str(func_error)}

                    # Append tool output to messages
                    asi_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": json.dumps(function_result)
                    })

                # Call API again with tool outputs
                response_data = call_asi_api(asi_messages, tools=tools_schema)
            else:
                # No more tool calls, we have the final response
                break

        # Extract final text response
        response_text = ''
        if 'choices' in response_data and response_data['choices']:
            response_text = response_data['choices'][0]['message'].get('content', '')

        # If the model didn't generate text but mining results exist, synthesize a summary
        if not response_text:
            mining_function_names = {'mine_pattern', 'start_mining_job', 'startMiningJob', 'minePattern'}
            mining_fr = next(
                (
                    fr for fr in function_results
                    if fr.get('name') in mining_function_names and isinstance(fr.get('result'), dict)
                ),
                None,
            )
            if mining_fr:
                mining_payload = mining_fr['result']
                patterns = []
                try:
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

        # Store conversation
        conversations[session_id].append({'role': 'user', 'content': message})
        conversations[session_id].append({'role': 'assistant', 'content': response_text})

        # Sanitize function results and return
        try:
            safe_function_results = make_json_safe(function_results)
        except Exception as e:
            print('Failed to sanitize function_results:', e)
            safe_function_results = str(function_results)

        return jsonify({'response': response_text, 'functionCalls': safe_function_results, 'session_id': session_id})

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

def create_app():
    """Application factory used by production WSGI servers.

    Startup is intentionally fail-fast: if Janus, SWI-Prolog, PeTTa, or the
    required MeTTa libraries cannot be loaded, this function raises and the
    server process does not start.
    """
    bootstrap_runtime()
    return app

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
    
    create_app()

    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        threaded=True
    )
