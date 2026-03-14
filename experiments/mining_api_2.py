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
import subprocess
import tempfile
import requests
import platform
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from hyperon import MeTTa
from dotenv import load_dotenv
import logging  
from petta import PeTTa  

logging.basicConfig(  
    format="%(asctime)s [%(levelname)s] %(message)s",  
    level=logging.INFO  
)  
logger = logging.getLogger(__name__)  


# Add workspace root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from experiments.ingestion.pipeline import run_ingestion
load_dotenv()

# Configure ASI API
ASI_API_KEY = os.getenv("ASI_API_KEY")
if not ASI_API_KEY:
    print("WARNING: ASI_API_KEY environment variable is not set. AI features will fail.")
ASI_BASE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-mini"

metta4Miner = MeTTa()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make PeTTa Python wrapper importable (persistent Prolog engine).
PETTA_PYTHON_PATH = os.path.join(PROJECT_ROOT, "PeTTa", "python")
if PETTA_PYTHON_PATH not in sys.path:
    sys.path.append(PETTA_PYTHON_PATH)

try:
    from petta import PeTTa
    PETTA_PYTHON_AVAILABLE = True
    PETTA_IMPORT_ERROR = None
except Exception as e:
    PeTTa = None
    PETTA_PYTHON_AVAILABLE = False
    PETTA_IMPORT_ERROR = str(e)

def to_wsl_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if re.match(r"^[A-Za-z]:\\", abs_path):
        drive = abs_path[0].lower()
        rest = abs_path[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"
    return abs_path.replace('\\', '/')

# For PeTTa/run.sh, always use forward slashes as it will be executed by bash
PROJECT_ROOT_WSL = to_wsl_path(PROJECT_ROOT)
PETTA_RUN_SH = to_wsl_path(os.path.join(PROJECT_ROOT, "PeTTa", "run.sh"))

# For in-process PeTTa (Janus), use native absolute paths with forward slashes.
PROJECT_ROOT_NATIVE = os.path.abspath(PROJECT_ROOT).replace('\\', '/')

def is_wsl_environment() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False

PROJECT_ROOT_PETTA = PROJECT_ROOT_WSL if is_wsl_environment() else PROJECT_ROOT_NATIVE

# Enable extra PeTTa debug probes for backward chaining when set to "1".
PETTA_DEBUG = os.getenv("PETTA_DEBUG", "0") == "1"

METTA_SETUP_WSL = f"""
!(import! &self {PROJECT_ROOT_WSL}/PeTTa/lib/lib_import.metta)
!(import_prolog_functions_from_file "{PROJECT_ROOT_WSL}/experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char promote_engagement_conj))

!(import! &self {PROJECT_ROOT_WSL}/experiments/utils/common-utils)
!(import! &stv-formulas {PROJECT_ROOT_WSL}/experiments/PLN/Formulas)
!(import! &self {PROJECT_ROOT_WSL}/experiments/frequent-pattern-miner/etv-utils)
!(import! &self {PROJECT_ROOT_WSL}/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self {PROJECT_ROOT_WSL}/experiments/pattern-miner/pattern-miner)
!(import! &self {PROJECT_ROOT_WSL}/experiments/chainer/main)
!(import! &tempo {PROJECT_ROOT_WSL}/experiments/atomspace_visualizer/public/data)

!(bind! &fact-count-petta (new-state 1))
(= (get-next-fact-id-petta)
   (let $n (get-state &fact-count-petta)
   (progn (change-state! &fact-count-petta (+ $n 1))
          (atom_concat fact $n))))

!(let $atom (match &tempo ($fact $stv) 
               (let $id (get-next-fact-id-petta)
                 (: $id $fact $stv)))
   (add-atom &res1 $atom))
!(let $atom (match &tempo ($fact $stv) $fact) (add-atom &purifiedDbSpace $atom))
"""

METTA_SETUP_PETTA = f"""
!(import! &self {PROJECT_ROOT_WSL}/PeTTa/lib/lib_import.metta)
!(import_prolog_functions_from_file "{PROJECT_ROOT_WSL}/experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char promote_engagement_conj))
!(import! &self {PROJECT_ROOT_WSL}/experiments/utils/common-utils)
!(import! &self {PROJECT_ROOT_WSL}/experiments/frequent-pattern-miner/etv-utils)
!(import! &self {PROJECT_ROOT_WSL}/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self {PROJECT_ROOT_WSL}/experiments/pattern-miner/pattern-miner)
!(import! &self {PROJECT_ROOT_WSL}/experiments/chainer/main)
!(import! &tempo {PROJECT_ROOT_WSL}/experiments/atomspace_visualizer/public/data)
!(import! &stv-formulas {PROJECT_ROOT_WSL}/experiments/PLN/Formulas)

!(bind! &fact-count (new-state 1))
(= (get-next-fact-id)
   (let $n (get-state &fact-count)
   (progn (change-state! &fact-count (+ $n 1))
          (atom_concat fact $n))))

!(let $atom (match &tempo ($fact $stv) 
               (let $id (get-next-fact-id)
                 (: $id $fact $stv)))
   (add-atom &res1 $atom))
!(let $atom (match &tempo ($fact $stv) $fact) (add-atom &purifiedDbSpace $atom))
"""

# Keep the original WSL-based setup for the existing MeTTa (hyperon) flow.
METTA_SETUP = METTA_SETUP_WSL

metta4Miner.run(METTA_SETUP)

# Persistent PeTTa runtime state (Janus-based).
petta_engine = None
petta_setup_loaded = False
petta_lock = threading.Lock()

def init_petta_engine() -> bool:
    """Initialize the in-process PeTTa engine and load METTA_SETUP once."""
    global petta_engine, petta_setup_loaded
    if not PETTA_PYTHON_AVAILABLE:
        return False

    if petta_engine is None:
        petta_engine = PeTTa(verbose=False, petta_path=os.path.join(PROJECT_ROOT, "PeTTa"))

    if not petta_setup_loaded:
        # Use load_metta_file so working_dir is set by the helper without changing PeTTa files.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.metta', delete=False) as tf:
            tf.write(METTA_SETUP_PETTA)
            setup_file_path = tf.name
        try:
            petta_engine.load_metta_file(setup_file_path)
        finally:
            if os.path.exists(setup_file_path):
                os.remove(setup_file_path)
        petta_setup_loaded = True

    return True


LOADEDLIB = False  
LOADED_LOCK = threading.Lock()  
class PeTTaChainer:  
    def __init__(self):  
        global LOADEDLIB  
        self.handler = PeTTa()  
          
        self.kb = "kb" + uuid.uuid4().hex  
        self._base_dir = os.path.dirname(__file__) 
        self.atomRe = re.compile(r'\([A-Za-z_][\w\-]*\s+\$[_\w\d]+\s+"[^"]*"\)')
        self.stvRe = re.compile(r'\(STV\s+([0-9eE\.\-]+)\s+([0-9eE\.\-]+)\)')
  
        if not LOADEDLIB:
            with LOADED_LOCK:
                if not LOADEDLIB:
                    metta_path = os.path.join(self._base_dir, "chainer", "petta_chainer.metta")
                    logger.info("Loading MeTTa library from %s", metta_path)
                    self.handler.load_metta_file(metta_path)
                    LOADEDLIB = True

  
    def add_atom(self, atom: str) -> str:  
        return self.handler.process_metta_string(f"!(compileadd {self.kb} {atom})")  
  
    def query(self, atom: str, depth: int = 10) -> List[str]:  
        atoms = self.handler.process_metta_string(  
            f"!(query (fromNumber {depth}) {self.kb} {atom})"  
        )  
        return atoms
    
    def normalizeVar(self, atom: str) -> str:
        return re.sub(r'\$_\d+', '$x', atom)

    def patternToRule(self, patternText: str, idx: int) -> str | None:
        atoms = [self.normalizeVar(a) for a in self.atomRe.findall(patternText or "")]
        if not atoms:
            return None

        stvMatch = self.stvRe.search(patternText or "")
        strength, confidence = (stvMatch.group(1), stvMatch.group(2)) if stvMatch else ("1.0", "1.0")

        consequent = next((a for a in atoms if a.startswith("(engagement ")), atoms[-1])
        antecedents = [a for a in atoms if a != consequent]
        lhs = antecedents[0] if len(antecedents) == 1 else f"(And {' '.join(antecedents)})"

        return f'(: rule_{idx} (-> {lhs} {consequent}) (STV {strength} {confidence}))'

    def formatter(self, minedPatterns):
        """Insert mined patterns as rules."""
        print("DEBUG: formatter received minedPatterns:", minedPatterns)
        try:
            payload = json.loads(minedPatterns) if isinstance(minedPatterns, str) else minedPatterns
            patterns = payload.get("patterns", [])
            insertedRules = []
            for idx, p in enumerate(patterns, start=1):
                patternText = str(p.get("pattern", ""))
                ruleAtom = self.patternToRule(patternText, idx)
                if not ruleAtom:
                    continue
                print("Adding rule:", ruleAtom)
                self.add_atom(ruleAtom)
                insertedRules.append(ruleAtom)

            return {
                "status": "success",
                "insertedRuleCount": len(insertedRules),
                "rules": insertedRules
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "insertedRuleCount": 0
            }
handler = PeTTaChainer()

# Define tools for ASI API
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "mine_pattern",
            "description": "Mines patterns with a specified number of conjunctions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numberOfConjunction": {
                        "type": "integer",
                        "description": "The number of conjunctions to use in pattern mining."
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
            "description": "Starts a mining job with a specified number of conjunctions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conjunction_count": {
                        "type": "integer",
                        "description": "The number of conjunctions to use in pattern mining."
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
                        "default": 5
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
        response = requests.post(ASI_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ASI API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return {"error": str(e)}

def run_metta_with_petta(metta_code: str) -> str:
    """
    Runs MeTTa code using the persistent PeTTa Python wrapper when available,
    otherwise falls back to the petta CLI.
    """
    if init_petta_engine():
        with petta_lock:
            try:
                results = petta_engine.process_metta_string(metta_code)
                print(f"DEBUG: In-process PeTTa results: {results}")
                if isinstance(results, (list, tuple)):
                    # If the runnable was not evaluated, fall back to load_metta_file.
                    if len(results) == 1 and str(results[0]).strip() == metta_code.strip().lstrip("!").strip():
                        raise ValueError("Unevaluated runnable from process_metta_string")
                    return "\n".join(str(r) for r in results)
                return str(results)
            except Exception as e:
                print(f"ERROR: In-process PeTTa failed: {e}")
                # Fallback: run via load_metta_file in-process before CLI.
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.metta', delete=False) as tf:
                        tf.write(metta_code)
                        temp_file_path = tf.name
                    results = petta_engine.load_metta_file(temp_file_path)
                    print(f"DEBUG: In-process PeTTa load_metta_file results: {results}")
                    if isinstance(results, (list, tuple)):
                        return "\n".join(str(r) for r in results)
                    return str(results)
                except Exception as e2:
                    print(f"ERROR: In-process PeTTa load_metta_file failed: {e2}")
                finally:
                    try:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except Exception:
                        pass
                # Fall back to CLI if available

    if not PETTA_PYTHON_AVAILABLE:
        print(f"WARNING: PeTTa Python wrapper unavailable: {PETTA_IMPORT_ERROR}")

    full_metta_script = METTA_SETUP_WSL + "\n" + metta_code

    with tempfile.NamedTemporaryFile(mode='w', suffix='.metta', delete=False) as tf:
        tf.write(full_metta_script)
        temp_file_path = tf.name
    print(full_metta_script)  # Debug: print the full MeTTa script being executed
    print(f"DEBUG: Temporary file created at: {temp_file_path}")

    try:
        # Using bash explicitly as PeTTa/run.sh is a shell script
        # and we are on Windows (likely using git bash or similar environment)
        temp_file_path_wsl = to_wsl_path(temp_file_path)
        cmd = ["bash", PETTA_RUN_SH, temp_file_path_wsl]
        print(f"DEBUG: Running petta command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Petta execution failed: {e.stderr}")
        raise Exception(f"Petta execution failed: {e.stderr}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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

def parse_facts_for_pettachainer(facts_output):  
    """  
    Parse nested facts output and convert to PeTTaChainer-compatible format.  
    Args:  
        facts_output: List containing a single string with nested facts  
    Returns:  
        List of individual fact strings ready for handler.add_atom()  
    """  
    if not facts_output:
        return []
    # Extract the big string
    nested_facts = facts_output[0]
    # Regex to capture valid facts
    pattern = r'\(:\s*fact\d+\s*\([^)]*\)\s*\(STV\s*[\d\.]+\s*[\d\.]+\)\)'
    matches = re.findall(pattern, nested_facts)
    individual_facts = [m.strip() for m in matches]
    return individual_facts
  

def extract_parenthesized_expressions(text: str) -> list[str]:
    """Extract all balanced parenthesized expressions from text."""
    results = []
    start = 0
    while True:
        idx = text.find("(", start)
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

def run_petta_query_lines(metta_code: str) -> list[str]:
    """Run MeTTa code through PeTTa and return cleaned output lines."""
    petta_output = run_metta_with_petta(metta_code)
    return [line for line in parse_petta_output(petta_output) if line and line.lower() != "true"]

def debug_petta_chainer_state(what_to_check: str) -> None:
    """Log a few diagnostic probes to confirm what PeTTa sees."""
    if not PETTA_DEBUG:
        return
    try:
        print("DEBUG: PeTTa probe - get-atoms &res1")
        atoms_lines = run_petta_query_lines(f"!(backward-chain &res1 (S (S Z)) (: $prf {what_to_check}))")
        print("DEBUG: PeTTa atoms lines (first 10):", atoms_lines)

        print("DEBUG: PeTTa probe - direct match")
        match_query = f"!(match &res1 {what_to_check} $x)"
        match_lines = run_petta_query_lines(match_query)
        print("DEBUG: PeTTa match lines:", match_lines)
    except Exception as e:
        print("DEBUG: PeTTa probe failed:", e)

def mine_pattern(numberOfConjunction: int) -> dict:
    """
    Mines patterns with a specified number of conjunctions using PeTTa.

    Args:
        numberOfConjunction: The number of conjunctions to use in pattern mining.

    Returns:
        A dictionary containing the mining results with parsed patterns.
    """
    print(f"Debug: mine pattern function being called with conjunction count {numberOfConjunction}")
    
    try:
        # Run the miner with petta
        query = f"!(pattern-miner &purifiedDbSpace 3 {int(numberOfConjunction)})"
        print(f"DEBUG: Executing PeTTa query: {query}")
        petta_output = run_metta_with_petta(query)
        print(f"DEBUG: PeTTa execution finished. Output length: {len(petta_output)}")
        normalized_query = query.strip().lstrip("!").strip()
        if petta_output.strip() == normalized_query:
            return {
                "status": "error",
                "message": "PeTTa returned the unevaluated expression. The runnable may not have executed.",
                "raw_result": petta_output
            }
        result_lines = parse_petta_output(petta_output)
        
        print(f"Debug: PeTTa result lines: {result_lines}")
        
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
            return {"status": "no_results", "patterns": []}
        
        return {
            "answer": full_answer_str,
            "status": "success",
            "conjunction_count": numberOfConjunction,
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

        if not username:
            return jsonify({"status": "error", "message": "Username is required"}), 400

        print(f"Received ingestion request for user: {username}")
        result = run_ingestion(username=username)

        if result.get("status") == "error":
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        print(f"Ingestion error: {e}")
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
    if init_petta_engine():
        try:
            lines = run_petta_query_lines("!(collapse (get-atoms &res1))")
            joined = " ".join(lines)
            facts = extract_parenthesized_expressions(joined) or lines
            return {"status": "success", "facts": facts}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    try:
        facts = metta4Miner.run("!(collapse (get-atoms &res1))")
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Normalize the returned structure into a flat list of string atoms
    normalized = []
    try:
        # If facts is nested (e.g., [[atom1, atom2]]), flatten one level
        if isinstance(facts, (list, tuple)) and len(facts) == 1 and isinstance(facts[0], (list, tuple)):
            iterable = facts[0]
        else:
            iterable = facts

        if isinstance(iterable, (list, tuple)):
            for item in iterable:
                try:
                    normalized.append(str(item))
                except Exception:
                    normalized.append(repr(item))
        else:
            normalized.append(str(iterable))

    except Exception:
        # Fallback: stringify the whole object
        try:
            return {"status": "success", "facts": [str(facts)]}
        except Exception:
            return {"status": "success", "facts": [repr(facts)]}

    return {"status": "success", "facts": normalized}


def handle_backward_chain_for_message(message: str):
    """Detect simple 'why' questions about an article and run the
    required automatic workflow: fetch facts, rewrite query to canonical
    form, then call the chainer. Returns (response_text, function_calls) or
    (None, None) if not applicable.
    """
    # Quick heuristic: look for 'why' and an article id
    if not re.search(r'\bwhy\b|explain why|prove that', message, re.I):
        return None, None

    m = re.search(r'article\s+(\d+)', message, re.I)
    if not m:
        return None, None

    article_id = m.group(1)

    # Call getAllFactsAndRules to obtain canonical atoms
    print("DEBUG: handle_backward_chain_for_message - calling getAllFactsAndRules()")
    facts_res = getAllFactsAndRules()
    print("DEBUG: facts_res type:", type(facts_res))
    print("DEBUG: facts_res preview:", (facts_res.get('facts')[:5] if isinstance(facts_res, dict) else str(facts_res)[:200]))
    function_calls = [{'name': 'getAllFactsAndRules', 'args': {}, 'result': facts_res}]

    if not isinstance(facts_res, dict) or facts_res.get('status') != 'success':
        return None, None

    facts = facts_res.get('facts', []) or []

    # Ask the LLM to rewrite the user's question into a canonical MeTTa query
    # using the facts we retrieved. The model must output only a single MeTTa
    # expression (e.g. (engagement 1 $what)).
    try:
        facts_text = "\n".join(facts[:200]) if isinstance(facts, list) else str(facts)
        rewrite_prompt = f"""
            You are given the following KB atoms (facts/rules), one per line:
            {facts_text}

            User question: "{message}"

            Task (STRICT):
            - Do NOT narrate or describe any internal steps.
            - Do NOT output anything except a SINGLE canonical MeTTa expression that uses predicate and constant names from the KB above.
            - If mapping is ambiguous, pick the most semantically likely predicate present in the KB.
            - If you cannot produce a valid MeTTa expression, output the single token NO_QUERY and NOTHING ELSE.

            Example mapping (for clarity only, do not output this): facts contain (engagement 1 high) -> question "Why is article 1 high?" -> output: (engagement 1 $what)

            OUTPUT ONLY the MeTTa expression or NO_QUERY.
            """
        print("DEBUG: sending rewrite prompt to LLM (first 300 chars):", rewrite_prompt[:300])
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": rewrite_prompt}
        ]
        response_data = call_asi_api(messages)
        candidate_query = None
        if 'choices' in response_data and response_data['choices']:
            candidate_query = response_data['choices'][0]['message'].get('content', '').strip()

        print("DEBUG: rewrite result:", candidate_query)
        function_calls.append({'name': 'rewrite_query', 'args': {'message': message}, 'result': candidate_query})
    except Exception as e:
        print('DEBUG: rewrite error:', e)
        return None, None

    if not candidate_query or candidate_query.upper() == 'NO_QUERY':
        print('DEBUG: no candidate query produced by LLM')
        return None, None

    # Ensure the candidate looks like a MeTTa expression; if it contains extra text,
    # try to extract the first parenthesized expression.
    mexpr = re.search(r"\([^\)]*\)", candidate_query)
    if mexpr:
        candidate_query = mexpr.group(0).strip()

    print('DEBUG: final candidate_query to send to chainer:', candidate_query)

    # Call the chainer with the rewritten query and include debug output
    try:
        print('DEBUG: calling getChainerResult with', candidate_query)
        chainer_result = getChainerResult(candidate_query)
        print('DEBUG: chainer_result type:', chainer_result)
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
    print('DEBUG: final response text:', resp_text)
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
        "conjunction_size": latest_job.conjunction_count
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
    if init_petta_engine():
        try:
            run_metta_with_petta(f"!(let $atom (main {mined_patterns}) (add-atom &res1 $atom))")
            print("formatter used PeTTa atomspace")
            return
        except Exception as e:
            print(f"formatter: PeTTa path failed, falling back to hyperon: {e}")

    mined_patterns = metta4Miner.parse_single(mined_patterns)
    metta4Miner.run(f""" !(let $atom (main {mined_patterns}) (add-atom &res1 $atom)) """)
    print("formatter ended :-_-:")

def backWardChainer(whatToCheck, depth=5):
    if init_petta_engine():
        debug_petta_chainer_state(whatToCheck)
        query = f"!(backward-chain &res1 (fromNumber 5) (: $prf {whatToCheck}))"
        lines = run_petta_query_lines(query)
        if PETTA_DEBUG:
            print("DEBUG: raw backward chaining output lines:", lines)
        joined = " ".join(lines)
        proofs = extract_parenthesized_expressions(joined)
        return proofs or lines

    whatToCheck = metta4Miner.parse_single(whatToCheck)
    answer = metta4Miner.run(f""" !(backward-chain &res1 (fromNumber 5) (: $prf {whatToCheck})) """)
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
    print("DEBUG: getChainerResult - chainAnswer type:", chainAnswer)
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
        When user asks "why is article 1 did get high engagement?", format query as "(engagement 1 high)" and call getChainerResult. 
        
        If backward chaining returns: [(: ((rule:- (, (engagement 1 high) (topic 1 AI))) (fact:- (topic 1 AI))) (engagement 1 high)), (: ((rule:- (, (engagement 1 high) (length 1 low))) (fact:- (length 1 low))) (engagement 1 high))]
        
        Analyze as: "I found 2 proofs for why article 1 has high engagement:
        
        **Proof 1:** Based on the rule that states 'if an article is about AI, then it has high engagement', and since we have the fact that 'article 1 is about AI', we can conclude that article 1 has high engagement.
        
        **Proof 2:** Based on the rule that states 'if an article is short (low length), then it has high engagement', and since we have the fact that 'article 1 has low length', we can also conclude that article 1 has high engagement.
        
        **Overall Justification:** Article 1's high engagement is well-supported by two independent logical proofs - both its AI topic and its concise length contribute to high engagement according to the rules in our knowledge base."

        The backward chaining system tried to prove the query "{whatToCheck}" and found the above results. Please analyze these results and explain the logical reasoning behind the proof(s).
        """

    try:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ]
        response_data = call_asi_api(messages)
        justification = None
        if 'choices' in response_data and response_data['choices']:
            justification = response_data['choices'][0]['message'].get('content', '')

        return {
            "query": whatToCheck,
            "status": "success",
            "raw_proofs": str(chainAnswer),
            "proof_count": len(chainAnswer),
            "justification": justification or "Unable to generate justification analysis.",
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
    supplied mined patterns. The summary will reference patterns as [N]
    so the frontend can make them clickable for visualization.
    """
    if not patterns:
        return "No patterns to summarize."

    # Build a compact prompt that lists the patterns and asks for a concise
    # analytic summary that includes references like [1], [2]
    prompt_parts = ["Please analyze the following mined patterns and produce a single concise summary that references specific patterns using [N] notation. Explain trends and actionable insights."]
    for i, p in enumerate(patterns, 1):
        patt = p.get('pattern') if isinstance(p, dict) else str(p)
        supp = p.get('support', '') if isinstance(p, dict) else ''
        prompt_parts.append(f"[{i}] Pattern: {patt} -- Support: {supp}")

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

SYSTEM_INSTRUCTION = """You are a friendly and knowledgeable AI assistant with expertise in data mining patterns, knowledge graphs, and pattern analysis. 

        **Your Primary Specialty:**
        You excel at analyzing pattern mining results, explaining conjunctions, and providing insights about relationships in data.

        **When to Use Functions:**
        - User says "Mine rules with X patterns" | "What patterns were found?" | "Show me the patterns" |or something like this → ALWAYS call mine_pattern(job_id: str , with the given conjunct number or default 3) first
        - "Analyze this pattern" / "Explain this pattern" → Use analyze_specific_pattern()
        - "Statistics" / "how many patterns" → Use get_pattern_statistics()
        - "Visualize" / "show me" a pattern → Use visualize_pattern_request()
    - "Why is..." / "Explain why..." / "Prove that..." questions → Use getChainerResult() with the query formatted as a MeTTa expression

    **CRITICAL BACKWARD-CHAINING WORKFLOW (MUST FOLLOW):**
    Before answering why questions, ALWAYS call getChainerResult().

    Example:
    - User: "What is article 1's engagement level?"
    - Assistant: call getChainerResult("(engagement 1 $whatIsIt)").
    - This ensures the chainer is invoked with a query that matches KB atoms and returns useful proofs.

                IMPORTANT: FOR ANY "WHY" / "EXPLAIN" / "PROVE" QUESTIONS (MANDATORY):
                - The assistant MUST enforce this internal workflow, but MUST NOT mention it to the user.
                    * Internally: fetch KB atoms, rewrite the NL question to a canonical MeTTa query, then call the chainer to obtain proofs.
                    * Externally (user-facing): do NOT narrate or reveal any of these internal steps, function calls, or that you fetched facts/rules. Never say "I called...", "I fetched...", or similar.
                - For such questions, the assistant MUST present only the final justification derived from the chainer or the concise statement "No proof was found." if no proof exists.
                - Style: final answers should be friendly, concise, and slightly jokey. No MeTTa expressions, no code blocks, and no internal diagnostic text in the user-facing reply unless the user explicitly requests the raw proof or the MeTTa query.
                - If the assistant cannot map the user's question to a MeTTa query or cannot produce a proof, respond with a short user-facing explanation (e.g. "I couldn't find a logical proof for that.") and offer to show the raw proof only if the user asks for it.

        **CRITICAL: When User Says "Mine rules with X patterns":**
        1. ALWAYS call mine_pattern() immediately to get all patterns
        2. Analyze ALL patterns together to find common themes
        3. Create ONE comprehensive summary (not individual summaries)
        4. In your summary, reference specific patterns using [Rule N] notation where N is the pattern index. Do NOT use comma separated list of rules format, like [Rule 1, Rule 2]; instead, use [Rule 1], [Rule 2]
        5. Format: "Based on the mining results, most of high engagement level is correlated to... [Rule 1] ... the longer the article is ... [Rule 3]"
        6. Focus on insights and trends across ALL patterns

        **Pattern Reference Format:**
        - Use [Rule 1], [Rule 2], etc. to reference patterns in your summary
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

        **Proof 2:** Another supporting rule indicates that [Rule N]:- [rule explanation], combined with the established facts:- [fact explanation], also leads to [conclusion].

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
    # Start formatting in background thread only if we have an answer payload
    print("🔍 DEBUG: Starting formatting thread")
    print("🔍 DEBUG: Result before formatting =", result)
    if isinstance(result, dict) and result.get('answer'):
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
        # directly serializable by Flask's JSON encoder). Reuse the top-level
        # sanitizer which is defensive about types.
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
            print(f"DEBUG: Creating new conversation session '{session_id}'")
        conversations.setdefault(session_id, [])

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Special-case: 'why' questions about articles should run the
        # return the chainer's justification directly.
        try:
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
            mining_fr = next((fr for fr in function_results if fr.get('name') == 'mine_pattern' and isinstance(fr.get('result'), dict)), None)
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