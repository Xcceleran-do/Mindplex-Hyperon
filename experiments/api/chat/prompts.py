from __future__ import annotations


def build_tools_schema(default_chain_depth: int) -> list[dict]:
    return [
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
                "description": "Gets the latest mining results including all patterns and their details.",
                "parameters": {
                    "type": "object",
                    "properties": {}
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
                    "properties": {}
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
                "description": "Gets the result of backward chaining for a specific query.",
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
                            "default": default_chain_depth
                        }
                    },
                    "required": ["whatToCheck"]
                }
            }
        }
    ]


def build_chainer_analysis_prompt(what_to_check: str, proof_text: str, fact_text: str) -> str:
    return f"""Analyze this backward chaining result with STV truth values and provide a clear logical justification.

    Query: {what_to_check}

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
If (audience 3 "Professionals") and (length-bucket 3 "high") then (engagement_level 3 "high")

Supporting Facts:
(audience 3 "Professionals") (STV 1.0 1.0)
(length-bucket 3 "high") (STV 0.9 0.8)

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
