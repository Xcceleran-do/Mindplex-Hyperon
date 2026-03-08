import os
import json
import google.generativeai as genai

class SemanticAgent:

    def __init__(self, model="gemini-2.5-flash"):
        self.model_name = model
        self.client = None
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("MINDPLEX_API_TOKEN")
        if api_key:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
        else:
            print("Warning: SemanticAgent initialized without GEMINI_API_KEY.")

    def analyze_text(self, text, prompt=None):
        if not self.client:
            return {}
            
        if not prompt:
            prompt = f"""
            Extract topics and difficulty level from this text.
    
            Return JSON with:
            topics: list
            difficulty: beginner/intermediate/advanced
    
            Text:
            {text[:2000]}
            """

        try:
            response = self.client.generate_content(prompt)
            result_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(result_text)
        except Exception as e:
            print(f"Error in SemanticAgent: {e}")
            return {}

    def process(self, records):
        enriched = []

        for r in records:
            text = None
            if "abstract" in r:
                text = r["abstract"]
            elif "description" in r:
                text = r["description"]
            elif "body" in r:
                text = r["body"]

            if text:
                analysis = self.analyze_text(text)
                r["semantic_analysis"] = analysis

            enriched.append(r)

        return enriched