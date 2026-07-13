import logging  
import os  
import re  
import json  
import threading  
import uuid  
from typing import List  
  
from petta import PeTTa  
from langchain_google_genai import ChatGoogleGenerativeAI  


from dotenv import load_dotenv
import pathlib

# Always load .env from project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
  
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
                    # Prefer local file; keep legacy subfolder path as fallback.
                    metta_path = os.path.join(self._base_dir, "petta_chainer.metta")
                    if not os.path.exists(metta_path):
                        metta_path = os.path.join(self._base_dir, "metta", "petta_chainer.metta")
                    logger.info("Loading MeTTa library from %s", metta_path)  
                    self.handler.load_metta_file(metta_path)  
                    LOADEDLIB = True  
  
    def get_all_facts(self):  
        """Helper method to retrieve all facts from the knowledge base"""  
        query_result = self.handler.process_metta_string(f"!(match &kb (: {self.kb} $prf $type $tv) (: {self.kb} $prf $type $tv))")  
        return query_result  
  
    def handle_why_question(self, user_question: str, depth: int = 5):
        """
        Pipeline: user question -> canonical query -> chainer -> LLM analysis (Gemini, langchain_google_genai).
        Returns (final_summary, canonical_query, chainer_result)
        """
        SYSTEM_INSTRUCTION = (
            "You are a friendly and knowledgeable AI assistant with expertise in data mining patterns, knowledge graphs, and pattern analysis.\n"
            "You help users understand why certain patterns or relationships exist in their data by analyzing logical rules and facts."
        )

        canonical_query = None
        try:
            facts = self.get_all_facts()
            facts_text = "\n".join(facts[:200])
            llm = get_llm()
            rewrite_prompt = f"""
You are given the following KB atoms (facts/rules), one per line:
{facts_text}

User question: "{user_question}"

Task (STRICT):
- Do NOT narrate or describe any internal steps.
- Do NOT output anything except a SINGLE canonical MeTTa expression that uses predicate and constant names from the KB above.
- If mapping is ambiguous, pick the most semantically likely predicate present in the KB.
- If you cannot produce a valid MeTTa expression, output the single token NO_QUERY and NOTHING ELSE.

Example mapping (for clarity only, do not output this): if facts contain (engagement articleA "Low") -> question "why article A_16624 has low engagement?" -> output should be like : "(: $prf (engagement A_16624 \"Low\") $tv)"

OUTPUT ONLY the MeTTa expression or NO_QUERY.
"""
            resp = llm.invoke(rewrite_prompt)
            canonical_query = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
            print(f"the canonical query is: {canonical_query}")
            if not canonical_query or canonical_query == 'NO_QUERY' or canonical_query.startswith('[ERROR]'):
                return ("Could not map your question to a valid query.", None, None)
            if not (canonical_query.startswith('(:') and '$prf' in canonical_query and '$tv' in canonical_query):
                return (f"Invalid query format received: {canonical_query}", None, None)
        except Exception as e:
            return (f"Error during query processing: {str(e)}", canonical_query, None)

        try:
            chainer_result = self.query(canonical_query, depth)
            print(f"the chainer result is: {chainer_result}")
        except Exception as e:
            return (f"Error during chainer execution: {str(e)}", canonical_query, None)

        try:
            facts = self.get_all_facts()
            analysis_prompt = f"""
You are an expert reasoning assistant. Given the backward chaining results below, explain to a human WHY the query is true (or not), focusing on the main reasons and contributing factors in plain, human language.

**How to answer:**
- Structure your answer as a sequence of proofs: "Proof 1", "Proof 2", etc.
- For each proof, explicitly quote or paraphrase the actual content of the supporting fact or rule (as shown in the knowledge base or proof), not just describe it.
- For direct facts, show the full fact content (e.g., (engagement articleC "Low") (STV ...)).
- For rule-based inferences, show the full rule content and the facts that satisfy its premises, step by step.
- Explain how each fact or rule contributes to the answer, in clear language.
- If some evidence is much stronger than others, highlight which is most important and why.
- If the rule-based path is weak or negligible, mention this simply.
- Do NOT show detailed calculation steps for STV, but you may mention if evidence is strong or weak.
- Do NOT mention or display any fact or rule names, labels, or IDs (such as 'fact5', 'rule_1', etc). Only use the content itself.

**Example:**
Suppose the user asks "why does article 1 have low engagement?" and the system finds:
- Direct fact: (engagement article1 "Low") (STV 0.9 0.93)
- Rule: (-> (And (author article1 "Hruy") (popularity article1 "Top_10")) (engagement article1 "Low")) (STV ...)
- Supporting facts: (author article1 "Hruy") (STV ...), (popularity article1 "Top_10") (STV ...)

Then answer:
Proof 1 (Direct Fact):
The main reason article 1 has low engagement is that it is directly stated as a fact: (engagement article1 "Low") (STV 0.9 0.93).

Proof 2 (Rule-Based Inference):
There is also a rule: (-> (And (author article1 "Hruy") (popularity article1 "Top_10")) (engagement article1 "Low")), and both (author article1 "Hruy") and (popularity article1 "Top_10") are facts in the knowledge base. This provides additional, but weaker, support for the conclusion.

IMPORTANT: Always extract and display the actual content of rules and facts from the raw proof data. Never mention or display any variable names, labels, or IDs (like 'fact52', 'rule_1', etc). Only use the full rule or fact content in your explanation.

**Query:** {canonical_query}
**Backward Chaining Results:** {chainer_result}
**Facts in the KB:** {facts}
"""
            resp = llm.invoke(analysis_prompt)
            summary = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
            if not summary:
                summary = 'Unable to generate justification analysis.'
        except Exception as e:
            summary = f'Error during analysis: {str(e)}'
        return (summary, canonical_query, chainer_result)
  
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
        try:
            payload = json.loads(minedPatterns) if isinstance(minedPatterns, str) else minedPatterns
            patterns = payload.get("patterns", [])
            insertedRules = []
            for idx, p in enumerate(patterns, start=1):
                patternText = str(p.get("pattern", ""))
                ruleAtom = self.patternToRule(patternText, idx)
                if not ruleAtom:
                    continue
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
  
  
def get_llm():  
    """Initialize Gemini LLM with API key from environment"""  
    api_key = os.getenv("GEMINI_API_KEY")  
    if not api_key:  
        raise RuntimeError("GEMINI_API_KEY not found in environment (.env).")  
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)  
  
  
def load_metta_file_to_chainer(chainer, metta_file_path):
    """Load MeTTa atoms from a file and insert them into the chainer KB, wrapping as (: factN ...)."""
    with open(metta_file_path, 'r') as f:
        fact_id = 1
        for line in f:
            atom = line.strip()
            if atom and not atom.startswith(';;'):
                # Remove outer parentheses if present
                if atom.startswith('(') and atom.endswith(')'):
                    atom_inner = atom[1:-1].strip()
                else:
                    atom_inner = atom
                # Wrap as (: factN ... )
                wrapped = f'(: fact{fact_id} {atom_inner})'
                chainer.add_atom(wrapped)
                fact_id += 1
  
def get_facts(handler):  
    """Get facts using direct match to avoid inference recursion"""  
    query_result = handler.handler.process_metta_string(  
        f"!(match &kb (: {handler.kb} $prf $type $tv) (: {handler.kb} $prf $type $tv))"  
    )  
    return query_result

def main():  
    """Main function to run the interactive why-question handler"""  
    handler = PeTTaChainer()  
    
    print("Loading knowledge base from data.metta ...")
    load_metta_file_to_chainer(handler, "../atomspace_visualizer/public/data.metta")
    print("Knowledge base loaded successfully!")

    facts = get_facts(handler)
    print("DEBUG: get_facts output:", facts[:10])  
  
    mined_patterns = {'patterns': [{'pattern': '(((author $_3514 "Hruy") (authored-by $_3514 "Hruy") (date-period $_3514 "Archived") (audience-expertise $_3514 "Beginner") (engagement $_3514 "Low")) (STV 1.7849705479859582e-9 0.07500000000000001))', 'support': '3'}, {'pattern': '(((author $_3734 "Hruy") (authored-by $_3734 "Hruy") (popularity $_3734 "Top_10") (audience-expertise $_3734 "Beginner") (engagement $_3734 "Low")) (STV 1.7849705479859582e-9 0.07500000000000001))', 'support': '3'}, {'pattern': '(((author $_3952 "Hruy") (date-period $_3952 "Archived") (popularity $_3952 "Top_10") (audience-expertise $_3952 "Beginner") (engagement $_3952 "Low")) (STV 1.7849705479859582e-9 0.07500000000000001))', 'support': '3'}, {'pattern': '(((authored-by $_4172 "Hruy") (date-period $_4172 "Archived") (popularity $_4172 "Top_10") (audience-expertise $_4172 "Beginner") (engagement $_4172 "Low")) (STV 1.7849705479859582e-9 0.07500000000000001))', 'support': '3'}]}
    # add mined patterns to database as rules
    handler.formatter(mined_patterns)
    # Interactive loop for user questions  
    while True:  
        print("\n" + "="*60)  
        print("Enter your 'why' question (e.g., 'Why does articleA have low engagement?')")  
        print("Type 'quit' or 'exit' to stop")  
        user_question = input("Your question: ").strip()  
          
        if user_question.lower() in ['quit', 'exit']:  
            break  
              
        if not user_question:  
            print("Please enter a valid question.")  
            continue  
              
        print(f"\nProcessing question: {user_question}")  
        summary, canonical_query, chainer_result = handler.handle_why_question(user_question)
          
        print("\n" + "="*60)  
        print("Analaysis:")  
        print("="*60)  
        print(f"\n CANONICAL QUERY:")  
        print(canonical_query if canonical_query else "No query generated")  
          
        print(f"\n CHAINER RESULT:")  
        if chainer_result:  
            for i, result in enumerate(chainer_result, 1):  
                print(f"{i}. {result}")  
        else:  
            print("No results from chainer")  
          
        print(f"\n ANALYSIS & SUMMARY:")  
        print(summary)  
  
if __name__ == "__main__":  
    main()
