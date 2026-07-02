from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from flask import jsonify, request


def parse_chat_mining_intent(
    message: str,
    *,
    default_conjunction_count: int,
    default_min_support: int,
) -> Optional[dict[str, int]]:
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

    conjunction_count = default_conjunction_count
    min_support = default_min_support

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


def handle_mining_for_message(
    message: str,
    *,
    default_conjunction_count: int,
    default_min_support: int,
    start_mining_job: Callable[..., dict],
    summarize_patterns: Callable[[list], str],
) -> tuple[Optional[str], Optional[list]]:
    """Run pattern mining directly when the chat message asks for it."""
    mining_params = parse_chat_mining_intent(
        message,
        default_conjunction_count=default_conjunction_count,
        default_min_support=default_min_support,
    )
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


ARTICLE_ID_RE = re.compile(r"\b[AH]_[A-Za-z0-9_]+\b")
ENGAGEMENT_LEVEL_RE = re.compile(r"\b(high|medium|low)\b", re.IGNORECASE)


def clean_metta_query(candidate_query: Optional[str]) -> Optional[str]:
    if not candidate_query:
        return None
    query = candidate_query.strip()
    query = re.sub(r"^```(?:metta)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s*```$", "", query)
    query = query.strip().strip("`").strip()
    return query or None


def deterministic_engagement_query(message: str, facts: list[str]) -> Optional[str]:
    """Build exact engagement queries without asking the LLM to rewrite obvious cases."""
    article_match = ARTICLE_ID_RE.search(message or "")
    level_match = ENGAGEMENT_LEVEL_RE.search(message or "")
    if not article_match or not level_match:
        return None

    article_id = article_match.group(0)
    level = level_match.group(1).capitalize()
    target_atom = f'(engagement {article_id} "{level}")'

    # Prefer an exact query even if the fact is absent; broad variable queries
    # are the common source of runaway proof searches.
    return f"(: $prf {target_atom} $tv)"


def safe_chainer_error_message(query: str, depth: int | None = None) -> str:
    depth_text = f" within depth {depth}" if depth is not None else ""
    return (
        "The backward chainer could not safely complete this proof"
        f"{depth_text}. This usually means the query expanded too broadly or hit a recursive/resource limit. "
        f"I tried the concrete query `{query}`; narrow the query or lower the depth if this repeats."
    )


def handle_backward_chain_for_message(
    message: str,
    *,
    get_all_facts_and_rules: Callable[[], dict],
    select_facts_for_prompt: Callable[[list[str], str, int], list[str]],
    call_asi_api: Callable[[list[dict[str, Any]], Optional[list[dict[str, Any]]]], dict[str, Any]],
    system_instruction: str,
    get_chainer_result: Callable[[str], dict],
    logger: Any,
) -> tuple[Optional[str], Optional[list]]:
    """Handle natural language queries using backward chaining with STV support."""
    function_calls = []
    facts_res = get_all_facts_and_rules()
    if not isinstance(facts_res, dict) or facts_res.get("status") != "success":
        return None, None

    facts = facts_res.get("facts", []) or []
    candidate_query = deterministic_engagement_query(message, facts)
    if candidate_query:
        function_calls.append({'name': 'rewrite_query_deterministic', 'args': {'message': message}, 'result': candidate_query})
    else:
        try:
            facts_text = "\n".join(select_facts_for_prompt(facts, message, 200))
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
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": rewrite_prompt},
            ]
            response_data = call_asi_api(messages)
            if 'choices' in response_data and response_data['choices']:
                candidate_query = response_data['choices'][0]['message'].get('content', '').strip()
            function_calls.append({'name': 'rewrite_query', 'args': {'message': message}, 'result': candidate_query})
        except Exception:
            logger.exception("Failed to rewrite a backward-chaining query from chat input")
            return None, None

    candidate_query = clean_metta_query(candidate_query)
    if not candidate_query or candidate_query == "NO_QUERY":
        return None, function_calls

    try:
        chainer_result = get_chainer_result(candidate_query)
    except Exception as exc:
        logger.exception("Backward chainer call failed")
        chainer_result = {'status': 'error', 'justification': safe_chainer_error_message(candidate_query), 'technical_error': str(exc)}

    function_calls.append({'name': 'getChainerResult', 'args': {'whatToCheck': candidate_query}, 'result': chainer_result})

    if isinstance(chainer_result, dict):
        if chainer_result.get('status') == 'error':
            return chainer_result.get('justification') or safe_chainer_error_message(candidate_query), function_calls
        raw_just = chainer_result.get('justification') or chainer_result.get('error') or ''
    else:
        raw_just = str(chainer_result)

    if not raw_just:
        return "No proof was found.", function_calls

    return f"Backward-chainer result:\n\n{raw_just}", function_calls


def build_rule_grounded_summary(patterns: list) -> str:
    lines = ["Here are the mined rules, grounded directly in the PeTTa output:"]
    for index, pattern in enumerate(patterns, 1):
        pattern_text = pattern.get('pattern') if isinstance(pattern, dict) else str(pattern)
        support = pattern.get('support', '') if isinstance(pattern, dict) else ''
        support_text = f"Support {support}" if support != "" else "Support not reported"
        lines.append(f"- [Rule {index}] {support_text}: `{pattern_text}`")
    lines.append(
        "These rules should be read as mined associations unless the rule itself "
        "and the proof chain explicitly support a stronger explanation."
    )
    return "\n".join(lines)


def summarize_patterns(
    patterns: list,
    *,
    call_asi_api: Callable[[list[dict[str, Any]], Optional[list[dict[str, Any]]]], dict[str, Any]],
    system_instruction: str,
    logger: Any,
) -> str:
    if not patterns:
        return "No patterns to summarize."

    prompt_parts = ["""Analyze the following mined PeTTa rules.

Strict requirements:
- Every factual statement or insight must cite at least one rule as [Rule N].
- Do not claim a trend unless it is directly supported by one of the listed rules.
- Explain the antecedent conditions, the conclusion, and the support value.
- Prefer short, concrete bullet points.
- If a rule is only an association, call it an association instead of a causal explanation.
- Do not invent facts that are not present in the rules."""]
    for index, pattern in enumerate(patterns, 1):
        pattern_text = pattern.get('pattern') if isinstance(pattern, dict) else str(pattern)
        support = pattern.get('support', '') if isinstance(pattern, dict) else ''
        prompt_parts.append(f"[Rule {index}]\nPattern: {pattern_text}\nSupport: {support}")

    prompt = "\n\n".join(prompt_parts)
    try:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        response_data = call_asi_api(messages)
        text = None
        if 'choices' in response_data and response_data['choices']:
            text = response_data['choices'][0]['message'].get('content', '')

        if text and "[Rule" in text:
            return text
        return build_rule_grounded_summary(patterns)
    except Exception as exc:
        logger.exception("summarize_patterns failed")
        fallback = build_rule_grounded_summary(patterns)
        return f"{fallback}\n\nSummary generation note: {exc}"


def parse_pattern(pattern: str) -> dict:
    properties = {}
    pattern = pattern.strip()
    if pattern.startswith('(') and pattern.endswith(')'):
        pattern = pattern[1:-1]

    matches = re.findall(r'\((\w+)\s+\$\w+\s+"([^"]+)"\)', pattern)
    for prop, value in matches:
        properties[prop] = value

    return properties


def analyze_pattern(pattern: str, support: str) -> str:
    properties = parse_pattern(pattern)
    if not properties:
        return f"📊 **Pattern Analysis**\n\nPattern: `{pattern}`\nSupport: **{support}**\n\nThis pattern appears {support} times in the dataset."

    description = " AND ".join(f"**{prop}** = `{value}`" for prop, value in properties.items())
    bullets = "\n".join(f"• {prop}: **{value}**" for prop, value in properties.items())

    return f"""📊 **Pattern Analysis**

        **Support:** {support} occurrences

        This pattern identifies topics that have:
        {bullets}

        **Interpretation:**
        Topics matching this pattern combine {description}. 
        The support value of {support} indicates this specific combination appears {support} times in your dataset.

        **Example Use Case:**
        This pattern can help identify content that has this specific combination of characteristics, useful for content recommendation, categorization, or trend analysis.
        """


def register_chat_routes(
    app,
    *,
    logger: Any,
    conversations: dict[str, list[dict[str, str]]],
    call_asi_api: Callable[[list[dict[str, Any]], Optional[list[dict[str, Any]]]], dict[str, Any]],
    system_instruction: str,
    tools_schema: list[dict[str, Any]],
    handle_mining_for_message: Callable[[str], tuple[Optional[str], Optional[list]]],
    is_backward_chain_intent: Callable[[str], bool],
    handle_backward_chain_for_message: Callable[[str], tuple[Optional[str], Optional[list]]],
    available_functions: dict[str, Callable[..., Any]],
    summarize_patterns: Callable[[list], str],
    analyze_pattern: Callable[[str, str], str],
    make_json_safe: Callable[[Any], Any],
    omegaclaw_chat_handler: Optional[Callable[..., dict[str, Any]]] = None,
) -> None:
    @app.route('/api/chat/health', methods=['GET'])
    def chat_health_check():
        return jsonify({'status': 'healthy', 'service': 'chat-api'})

    @app.route('/api/chat/analyze', methods=['POST', 'OPTIONS'])
    def analyze_conjunct():
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
            return response, 200

        try:
            data = request.get_json() or {}
            pattern = data.get('pattern', '')
            support = data.get('support', '0')
            if omegaclaw_chat_handler is not None:
                try:
                    omega_result = omegaclaw_chat_handler(
                        (
                            "Analyze this mined Mindplex pattern for the UI. "
                            "Be concise and explain what the rule suggests.\n"
                            f"Pattern: {pattern}\n"
                            f"Support: {support}"
                        ),
                        session_id=str(data.get('session_id', 'analyze')),
                        history=[],
                    )
                except TimeoutError as exc:
                    logger.warning("OmegaClaw analyze bridge timed out: %s", exc)
                    return jsonify({'error': str(exc), 'backend': 'omegaclaw'}), 504

                summary = str(omega_result.get("response", "")).strip()
                if not summary:
                    summary = "OmegaClaw returned an empty pattern analysis."
                return jsonify({'summary': summary, 'pattern': pattern, 'support': support, 'backend': 'omegaclaw'})

            summary = analyze_pattern(pattern, support)
            return jsonify({'summary': summary, 'pattern': pattern, 'support': support})
        except Exception as exc:
            logger.exception("analyze_conjunct failed")
            return jsonify({'error': str(exc)}), 500

    @app.route('/api/chat/summarize', methods=['POST', 'OPTIONS'])
    def summarize_patterns_endpoint():
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
            return response, 200

        try:
            data = request.get_json() or {}
            patterns = data.get('patterns', [])
            if omegaclaw_chat_handler is not None:
                limited_patterns = patterns[:20] if isinstance(patterns, list) else patterns
                omitted = len(patterns) - len(limited_patterns) if isinstance(patterns, list) else 0
                pattern_payload = json.dumps(limited_patterns, ensure_ascii=True)
                suffix = f"\nOmitted pattern count: {omitted}" if omitted > 0 else ""
                try:
                    omega_result = omegaclaw_chat_handler(
                        (
                            "Summarize these mined Mindplex patterns for the UI. "
                            "Mention the count, strongest signals, and what the user should inspect next.\n"
                            f"Patterns JSON: {pattern_payload}{suffix}"
                        ),
                        session_id=str(data.get('session_id', 'summary')),
                        history=[],
                    )
                except TimeoutError as exc:
                    logger.warning("OmegaClaw summarize bridge timed out: %s", exc)
                    return jsonify({'error': str(exc), 'backend': 'omegaclaw'}), 504

                summary = str(omega_result.get("response", "")).strip()
                if not summary:
                    summary = "OmegaClaw returned an empty pattern summary."
                return jsonify({'summary': summary, 'backend': 'omegaclaw'})

            summary = summarize_patterns(patterns)
            return jsonify({'summary': summary})
        except Exception as exc:
            logger.exception("summarize_patterns_endpoint failed")
            return jsonify({'error': str(exc)}), 500

    @app.route('/api/chat', methods=['POST', 'OPTIONS'])
    def chat():
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

            if session_id not in conversations:
                conversations[session_id] = []

            if not message:
                return jsonify({'error': 'Message is required'}), 400

            logger.info("Chat request received: session=%s chars=%s", session_id, len(message))
            if omegaclaw_chat_handler is not None:
                try:
                    omega_result = omegaclaw_chat_handler(
                        message,
                        session_id=session_id,
                        history=history,
                    )
                except TimeoutError as exc:
                    logger.warning("OmegaClaw bridge timed out: %s", exc)
                    return jsonify({'error': str(exc), 'backend': 'omegaclaw'}), 504

                response_text = str(omega_result.get("response", "")).strip()
                if not response_text:
                    response_text = "OmegaClaw returned an empty response."
                conversations[session_id].append({'role': 'user', 'content': message})
                conversations[session_id].append({'role': 'assistant', 'content': response_text})
                function_calls = [{
                    "name": "omegaclaw_chat",
                    "args": {"session_id": session_id},
                    "result": {
                        "id": omega_result.get("id"),
                        "backend": omega_result.get("backend", "omegaclaw"),
                    },
                }]
                logger.info("Chat routed to OmegaClaw: session=%s response_chars=%s", session_id, len(response_text))
                return jsonify({
                    'response': response_text,
                    'functionCalls': function_calls,
                    'session_id': session_id,
                    'backend': 'omegaclaw',
                })

            mining_text, mining_calls = handle_mining_for_message(message)
            if mining_text is not None:
                conversations[session_id].append({'role': 'user', 'content': message})
                conversations[session_id].append({'role': 'assistant', 'content': mining_text})
                try:
                    safe_calls = make_json_safe(mining_calls)
                except Exception:
                    safe_calls = str(mining_calls)
                logger.info(
                    "Chat shortcut used: session=%s shortcut=mining calls=%s",
                    session_id,
                    [call.get('name') for call in safe_calls] if isinstance(safe_calls, list) else safe_calls,
                )
                return jsonify({'response': mining_text, 'functionCalls': safe_calls, 'session_id': session_id})

            try:
                if is_backward_chain_intent(message):
                    bc_text, bc_calls = handle_backward_chain_for_message(message)
                    if bc_text is not None:
                        conversations[session_id].append({'role': 'user', 'content': message})
                        conversations[session_id].append({'role': 'assistant', 'content': bc_text})
                        try:
                            safe_calls = make_json_safe(bc_calls)
                        except Exception:
                            safe_calls = str(bc_calls)
                        chainer_status = None
                        proof_count = None
                        if isinstance(safe_calls, list):
                            for call in safe_calls:
                                if call.get('name') == 'getChainerResult' and isinstance(call.get('result'), dict):
                                    chainer_status = call['result'].get('status')
                                    proof_count = call['result'].get('proof_count')
                        logger.info(
                            "Chat shortcut used: session=%s shortcut=backward_chain status=%s proofs=%s calls=%s",
                            session_id,
                            chainer_status,
                            proof_count,
                            [call.get('name') for call in safe_calls] if isinstance(safe_calls, list) else safe_calls,
                        )
                        return jsonify({'response': bc_text, 'functionCalls': safe_calls, 'session_id': session_id})
            except Exception:
                logger.exception("Backward chain shortcut handling failed")

            asi_messages = [{"role": "system", "content": system_instruction}]
            for msg in history[-10:]:
                role = msg.get('role')
                if role == 'assistant':
                    role = 'assistant'
                elif role == 'user':
                    role = 'user'
                asi_messages.append({'role': role, 'content': msg.get('content', '')})

            asi_messages.append({'role': 'user', 'content': message})
            response_data = call_asi_api(asi_messages, tools=tools_schema)

            max_iterations = 5
            iteration = 0
            function_results = []

            while iteration < max_iterations:
                iteration += 1

                if 'error' in response_data:
                    logger.warning("ASI API returned an error payload: %s", response_data['error'])
                    break

                if 'choices' not in response_data or not response_data['choices']:
                    break

                choice = response_data['choices'][0]
                message_obj = choice['message']
                asi_messages.append(message_obj)

                if 'tool_calls' not in message_obj or not message_obj['tool_calls']:
                    break

                for tool_call in message_obj['tool_calls']:
                    function_name = tool_call['function']['name']
                    function_args_str = tool_call['function']['arguments']
                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError:
                        function_args = {}

                    if function_name not in available_functions:
                        logger.warning("Unknown function requested by ASI: %s", function_name)
                        function_result = {"error": f"Unknown function {function_name}"}
                    else:
                        try:
                            norm_args = {}
                            for key, value in function_args.items():
                                normalized_key = key
                                if key in ('conjunction_count', 'conjunctions', 'conjunctionCount', 'numberOfConjunction', 'n'):
                                    normalized_key = 'conjunction_count'
                                elif key in ('min_support', 'minimum_support', 'minSupport', 'minimumSupport', 'support'):
                                    normalized_key = 'min_support'

                                if isinstance(value, str) and re.fullmatch(r"\d+", value):
                                    norm_args[normalized_key] = int(value)
                                elif isinstance(value, float) and value.is_integer():
                                    norm_args[normalized_key] = int(value)
                                else:
                                    norm_args[normalized_key] = value

                            function_result = available_functions[function_name](**norm_args)
                            function_results.append({'name': function_name, 'args': norm_args, 'result': function_result})
                            logger.info(
                                "ASI tool call completed: session=%s function=%s args=%s status=%s",
                                session_id,
                                function_name,
                                norm_args,
                                function_result.get('status') if isinstance(function_result, dict) else type(function_result).__name__,
                            )
                        except Exception as func_error:
                            logger.exception("Function call failed: %s", function_name)
                            function_result = {'error': str(func_error)}

                    asi_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": json.dumps(function_result),
                    })

                response_data = call_asi_api(asi_messages, tools=tools_schema)

            response_text = ''
            if 'choices' in response_data and response_data['choices']:
                response_text = response_data['choices'][0]['message'].get('content', '')

            if not response_text:
                mining_function_names = {'mine_pattern', 'start_mining_job', 'startMiningJob', 'minePattern'}
                mining_result = next(
                    (
                        result for result in function_results
                        if result.get('name') in mining_function_names and isinstance(result.get('result'), dict)
                    ),
                    None,
                )
                if mining_result:
                    mining_payload = mining_result['result']
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
                        except Exception:
                            logger.exception("Failed to generate summary after function call")

            if not response_text:
                response_text = "I apologize, but I couldn't generate a proper response. Please try again."

            conversations[session_id].append({'role': 'user', 'content': message})
            conversations[session_id].append({'role': 'assistant', 'content': response_text})

            try:
                safe_function_results = make_json_safe(function_results)
            except Exception:
                logger.exception("Failed to sanitize function_results")
                safe_function_results = str(function_results)

            return jsonify({'response': response_text, 'functionCalls': safe_function_results, 'session_id': session_id})

        except Exception as exc:
            logger.exception("chat endpoint failed")
            return jsonify({'error': str(exc)}), 500

    @app.route('/api/chat/clear', methods=['POST', 'OPTIONS'])
    def clear_chat():
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
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500
