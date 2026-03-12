# experiments/ingestion/analyzer.py
import datetime
import json
import re
import requests
from .config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS, 
    RETENTION_BUCKETS, ANALYSIS_PROMPT_TEMPLATE, ENTITY_EXTRACTION_PROMPT_TEMPLATE,
    CLASSIFICATION_PROMPT_TEMPLATE, SENTIMENT_PROMPT_TEMPLATE, OPEN_IE_PROMPT_TEMPLATE,
    DETERMINISTIC_STV, AI_FAILURE_STV, UNKNOWN_STV
)

ASI_BASE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-mini"

class BaseProcessor:
    def __init__(self, api_key):
        self.api_key = api_key

    def call_asi_api(self, messages):
        if not self.api_key:
            return {"error": "No API key"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": ASI_MODEL,
            "messages": messages,
            "temperature": 0.5 # Lower temperature for better structural consistency
        }
        try:
            response = requests.post(ASI_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ASI API Error: {e}")
            return {"error": str(e)}

    def get_text_snippet(self, article, length=2000):
        raw_content = article.get('content', [])
        if isinstance(raw_content, list):
            text_content = " ".join([item.get('content', '') for item in raw_content if isinstance(item, dict) and item.get('type') == 'p'])
        else:
            text_content = str(raw_content or article.get('overview', ''))
        return text_content[:length]

    def _safe_json_parse(self, response_data, fallback):
        try:
            if 'choices' in response_data and response_data['choices']:
                text = response_data['choices'][0]['message'].get('content', '')
                # Clean up markdown
                text = text.replace('```json', '').replace('```', '').strip()
                return json.loads(text)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
        return fallback

class ClassificationAgent(BaseProcessor):
    def process(self, article):
        snippet = self.get_text_snippet(article)
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            title=article.get('post_title', ''),
            content_snippet=snippet
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"domain": "Other", "format": "Other", "confidence": 0.5})

class FormatConverterAgent(BaseProcessor):
    def process(self, article):
        """Ensures all inputs are in a consistent internal format."""
        file_path = article.get('file_path')
        if not file_path:
            # Standard article object with content list
            raw_content = article.get('content', [])
            if isinstance(raw_content, list):
                text = " ".join([item.get('content', '') for item in raw_content if isinstance(item, dict)])
            else:
                text = str(raw_content or article.get('overview', ''))
            return {"normalized_text": text}

        # Local file handling
        ext = file_path.split('.')[-1].lower()
        text = ""
        try:
            if ext == 'pdf':
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = " ".join([page.extract_text() for page in reader.pages])
            elif ext == 'csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    text = " ".join([" ".join(row) for row in reader])
            elif ext == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data) if not isinstance(data, str) else data
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
        except Exception as e:
            print(f"Format Conversion Error ({ext}): {e}")
            text = str(article.get('content', ''))

        return {"normalized_text": text}

class SemanticAgent(BaseProcessor):
    def process(self, article):
        snippet = self.get_text_snippet(article)
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            title=article.get('post_title', ''),
            content_snippet=snippet
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {
            "tone": {"value": "Unknown", "strength": 0.0, "confidence": 0.0},
            "audience_expertise": {"value": "Unknown", "strength": 0.0, "confidence": 0.0},
            "content_type": {"value": "Unknown", "strength": 0.0, "confidence": 0.0},
            "primary_goal": {"value": "Unknown", "strength": 0.0, "confidence": 0.0},
            "audience_sentiment": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}
        })

class SentimentAgent(BaseProcessor):
    def process(self, article):
        snippet = self.get_text_snippet(article, length=1000) # Smaller snippet for sentiment
        prompt = SENTIMENT_PROMPT_TEMPLATE.format(text=snippet)
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"sentiment": "Neutral", "strength": 0.5, "confidence": 0.5})

class EntityLinkingAgent(BaseProcessor):
    def process(self, article):
        snippet = self.get_text_snippet(article)
        prompt = ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(
            title=article.get('post_title', ''),
            content_snippet=snippet
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"entities": []})

class OpenIEAgent(BaseProcessor):
    def process(self, article):
        snippet = self.get_text_snippet(article, length=1000)
        prompt = OPEN_IE_PROMPT_TEMPLATE.format(text=snippet)
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"triples": []})

class ArticleAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.classifier = ClassificationAgent(api_key)
        self.format_converter = FormatConverterAgent(api_key)
        self.semantic = SemanticAgent(api_key)
        self.sentiment = SentimentAgent(api_key)
        self.entities = EntityLinkingAgent(api_key)
        self.openie = OpenIEAgent(api_key)
        
        if not self.api_key:
            print("Warning: No ASI API key provided. AI enrichment will be limited.")

    def discretize_value(self, value, buckets):
        if value is None: return "Unknown", UNKNOWN_STV
        for label, (condition, stv_range) in buckets.items():
            try:
                if condition(value): return label, stv_range
            except: continue
        return "Unknown", UNKNOWN_STV
    
    def calculate_proportional_stv(self, value, bucket_label, stv_range, bucket_bounds=None):
        if stv_range[0] == stv_range[1]: return stv_range
        if bucket_bounds:
            min_val, max_val = bucket_bounds
            normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            normalized = max(0, min(1, normalized))
        else: normalized = 0.5
        strength = stv_range[0] + normalized * (stv_range[1] - stv_range[0])
        return (round(strength, 3), 0.9)

    def process(self, article, rank_stats=None):
        # 1. Classification
        cls_res = self.classifier.process(article)
        
        # 2. Format Normalization
        norm_res = self.format_converter.process(article)
        full_text = norm_res['normalized_text']
        word_count = len(full_text.split())
        
        # 3. Rule-based analysis
        read_time_str = article.get('min_to_read', '0')
        read_time = int(re.search(r'(\d+)', str(read_time_str)).group(1)) if re.search(r'(\d+)', str(read_time_str)) else 0
        
        len_label, len_range = self.discretize_value(word_count, LENGTH_BUCKETS)
        rt_label, rt_range = self.discretize_value(read_time, READING_TIME_BUCKETS)
        
        metadata = {
            "domain": {"value": cls_res['domain'], "stv": (cls_res['confidence'], 1.0)},
            "length": {"value": len_label, "stv": self.calculate_proportional_stv(word_count, len_label, len_range)},
            "reading_time": {"value": rt_label, "stv": self.calculate_proportional_stv(read_time, rt_label, rt_range)},
            "author": {"value": article.get('author_username', 'unknown'), "stv": DETERMINISTIC_STV},
            "category": {"value": article.get('categories', [{}])[0].get('slug', 'general'), "stv": DETERMINISTIC_STV},
            "title": {"value": article.get('post_title', 'untitled'), "stv": DETERMINISTIC_STV}
        }

        # 4. Semantic & Sentiment
        sem_res = self.semantic.process(article)
        for key, res in sem_res.items():
            metadata[key] = {"value": res['value'], "stv": (res.get('strength', 0.5), res.get('confidence', 0.5))}
            
        sent_res = self.sentiment.process(article)
        metadata['sentiment'] = {"value": sent_res['sentiment'], "stv": (sent_res['strength'], sent_res['confidence'])}

        # 5. Entity Linking
        ent_res = self.entities.process(article)
        metadata['entities'] = ent_res.get('entities', [])

        # 6. OpenIE
        ie_res = self.openie.process(article)
        metadata['openie_triples'] = ie_res.get('triples', [])

        article['enriched_metadata'] = metadata
        return article
