#!/usr/bin/env python3
"""Unified API server wiring for mining, simulation, and chat."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

from flask import Flask
from flask_cors import CORS

# Add workspace root to path to allow imports when running as a script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from experiments.api.asi_client import call_asi_api
from experiments.api.chainer import (
    backWardChainer,
    getAllFactsAndRules,
    getChainerResult as get_chainer_result_impl,
)
from experiments.api.chat.prompts import SYSTEM_INSTRUCTION, build_tools_schema
from experiments.api.chat.support import (
    analyze_pattern as analyze_pattern_impl,
    handle_backward_chain_for_message as handle_backward_chain_for_message_impl,
    handle_mining_for_message as handle_mining_for_message_impl,
    is_backward_chain_intent,
    parse_chat_mining_intent as parse_chat_mining_intent_impl,
    parse_pattern as parse_pattern_impl,
    register_chat_routes,
    summarize_patterns as summarize_patterns_impl,
)
from experiments.ingestion.pipeline import run_ingestion
from experiments.api.config import (
    DEFAULT_CHAIN_DEPTH,
    DEFAULT_CONJUNCTION_COUNT,
    DEFAULT_MIN_SUPPORT,
    dataset_file_path,
)
from experiments.api.routes import register_core_routes
from experiments.api.support import (
    extract_parenthesized_expressions,
    extract_support_of_expressions,
    load_dataset_facts_for_chainer,
    make_json_safe,
    parse_facts_for_pettachainer,
    parse_pattern_string,
    parse_petta_output,
    select_facts_for_prompt,
)
from experiments.api.mining import (
    MiningJob,
    formatter,
    get_mining_results,
    get_pattern_statistics,
    insert_mined_rules_into_chainer,
    mine_pattern,
    mining_jobs,
    run_metta_with_petta,
    run_mining_task,
    run_mining_task_inprocess,
    start_mining_job,
)
from experiments.api.runtime import (
    get_chainer_service,
    invalidate_chainer_dataset,
    reload_petta_dataset_if_ready,
)
from experiments.api.simulation import (
    build_simulation_explanation,
    build_simulation_fact_atoms,
    simulate_engagement,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": False,
    "max_age": 3600,
}})

tools_schema = build_tools_schema(DEFAULT_CHAIN_DEPTH)
conversations: Dict[str, list] = {}


def parse_chat_mining_intent(message: str) -> Optional[Dict[str, int]]:
    return parse_chat_mining_intent_impl(
        message,
        default_conjunction_count=DEFAULT_CONJUNCTION_COUNT,
        default_min_support=DEFAULT_MIN_SUPPORT,
    )


def summarize_patterns(patterns: list) -> str:
    return summarize_patterns_impl(
        patterns,
        call_asi_api=call_asi_api,
        system_instruction=SYSTEM_INSTRUCTION,
        logger=logger,
    )


def handle_mining_for_message(message: str) -> tuple[Optional[str], Optional[list]]:
    return handle_mining_for_message_impl(
        message,
        default_conjunction_count=DEFAULT_CONJUNCTION_COUNT,
        default_min_support=DEFAULT_MIN_SUPPORT,
        start_mining_job=start_mining_job,
        summarize_patterns=summarize_patterns,
    )


def getChainerResult(whatToCheck, depth=DEFAULT_CHAIN_DEPTH):
    return get_chainer_result_impl(whatToCheck, depth=depth, call_asi_api=call_asi_api)


def handle_backward_chain_for_message(message: str) -> tuple[Optional[str], Optional[list]]:
    return handle_backward_chain_for_message_impl(
        message,
        get_all_facts_and_rules=getAllFactsAndRules,
        select_facts_for_prompt=select_facts_for_prompt,
        call_asi_api=call_asi_api,
        system_instruction=SYSTEM_INSTRUCTION,
        get_chainer_result=getChainerResult,
        logger=logger,
    )


def parse_pattern(pattern: str) -> dict:
    return parse_pattern_impl(pattern)


def analyze_pattern(pattern: str, support: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": (
                "Analyze this mined PeTTa rule. Explain the antecedent, conclusion, "
                "and support without claiming causation. Be concise.\n\n"
                f"Pattern: {pattern}\nSupport: {support}"
            ),
        },
    ]
    response_data = call_asi_api(messages)
    if response_data.get("choices"):
        content = response_data["choices"][0]["message"].get("content", "").strip()
        if content:
            return content
    return analyze_pattern_impl(pattern, support)


def analyze_specific_pattern(pattern: str) -> dict:
    properties = parse_pattern(pattern)
    return {
        "pattern": pattern,
        "properties": properties,
        "property_count": len(properties),
        "description": f"Pattern with {len(properties)} properties: {', '.join(properties.keys())}"
    }


def visualize_pattern_request(pattern: str) -> dict:
    return {
        "action": "visualize",
        "pattern": pattern,
        "message": "Pattern visualization requested. The frontend will display this pattern."
    }


available_functions = {
    "mine_pattern": start_mining_job,
    "start_mining_job": start_mining_job,
    "startMiningJob": start_mining_job,
    "minePattern": start_mining_job,
    "get_mining_results": get_mining_results,
    "analyze_specific_pattern": analyze_specific_pattern,
    "get_pattern_statistics": get_pattern_statistics,
    "visualize_pattern_request": visualize_pattern_request,
    "getChainerResult": getChainerResult,
}


register_core_routes(
    app,
    logger=logger,
    run_ingestion=run_ingestion,
    invalidate_chainer_dataset=invalidate_chainer_dataset,
    dataset_file_path=dataset_file_path,
    get_chainer_service=get_chainer_service,
    default_conjunction_count=DEFAULT_CONJUNCTION_COUNT,
    default_min_support=DEFAULT_MIN_SUPPORT,
    default_chain_depth=DEFAULT_CHAIN_DEPTH,
    mining_jobs=mining_jobs,
    mining_job_type=MiningJob,
    run_mining_task=run_mining_task,
    simulate_engagement=simulate_engagement,
    run_chainer_query=backWardChainer,
    make_json_safe=make_json_safe,
)

register_chat_routes(
    app,
    logger=logger,
    conversations=conversations,
    call_asi_api=call_asi_api,
    system_instruction=SYSTEM_INSTRUCTION,
    tools_schema=tools_schema,
    handle_mining_for_message=handle_mining_for_message,
    is_backward_chain_intent=is_backward_chain_intent,
    handle_backward_chain_for_message=handle_backward_chain_for_message,
    available_functions=available_functions,
    summarize_patterns=summarize_patterns,
    analyze_pattern=analyze_pattern,
    make_json_safe=make_json_safe,
)


def create_app():
    """Application factory used by production WSGI servers."""
    return app


if __name__ == '__main__':
    logger.info("Starting Unified API Server (Mining + Chat)")
    create_app()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        threaded=True
    )
