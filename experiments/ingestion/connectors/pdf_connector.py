# ingestion/connectors/pdf_connector.py

from pdfminer.high_level import extract_text
from .base_connector import BaseConnector


class PdfConnector(BaseConnector):

    def fetch(self):
        import os
        from .base_connector import BaseConnector
        from ..agents.semantic_agent import SemanticAgent
        import json

        path = self.config.get("path")
        
        if not os.path.isabs(path):
            # Resolve relative to the experiments/ingestion/ directory
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(script_dir, path)

        text = extract_text(path)
        
        # Use SemanticAgent to extract metadata from the text
        agent = SemanticAgent()
        
        # Override the url/prompt locally to extract paper metadata
        prompt = f"""
        Extract paper metadata from this text.
        Return ONLY a raw JSON object (no markdown formatting, no code blocks) with the following keys:
        - post_title: string
        - author_display_name: string
        - overview: string
        - category: string
        - sentiment: string (e.g., Positive, Negative, Neutral)
        - writing_level: string (e.g., Beginner, Intermediate, Advanced)
        
        Text:
        {text[:2000]}
        """
        
        try:
            import os
            filename = os.path.basename(path)
            safe_name = filename.replace('.pdf', '').replace(' ', '_').lower()
            entity_id = f"paper_{safe_name}"
            
            metadata = agent.analyze_text(text, prompt=prompt)
            metadata["id"] = entity_id
            return [metadata]
        except Exception as e:
            print(f"Error extracting metadata with LLM: {e}")
            return [
                {
                    "id": path, # fallback
                    "text": text,
                    "post_title": "Unknown Title",
                    "author_display_name": "Unknown Author",
                    "overview": text[:200]
                }
            ]