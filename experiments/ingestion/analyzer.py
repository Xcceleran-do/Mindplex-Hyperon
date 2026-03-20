# experiments/ingestion/analyzer.py
import datetime
import json
import re
import requests
from .config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS, 
    RETENTION_BUCKETS, ANALYSIS_PROMPT_TEMPLATE, ENTITY_EXTRACTION_PROMPT_TEMPLATE,
    CLASSIFICATION_PROMPT_TEMPLATE, SENTIMENT_PROMPT_TEMPLATE, OPEN_IE_PROMPT_TEMPLATE,
    METADATA_SCHEMA_PROMPT_TEMPLATE, METADATA_EXTRACTION_PROMPT_TEMPLATE, ENTITY_TYPE_PROMPT_TEMPLATE,
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

    def get_text_snippet(self, document, length=2000):
        raw_content = document.get('content', [])
        if isinstance(raw_content, list):
            text_content = " ".join([item.get('content', '') for item in raw_content if isinstance(item, dict) and item.get('type') == 'p'])
        else:
            text_content = str(raw_content or document.get('overview', ''))
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
    def __init__(self, api_key):
        super().__init__(api_key)
        try:
            from transformers import pipeline
            self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        except Exception as e:
            print(f"Warning: Could not load transformers. Fallback to API. Error: {e}")
            self.classifier = None
            
        self.domain_labels = ["News", "Research Paper", "Tutorial", "Opinion", "Review", "Interview", "Movie", "Social Media", "Documentation", "Other"]
        self.format_labels = ["Article", "PDF", "Video", "Post", "Book", "Code", "Other"]

    def process(self, document):
        snippet = self.get_text_snippet(document, length=1500)
        
        if self.classifier:
            if not snippet.strip():
                return {"domain": "Other", "format": "Other", "confidence": 0.5}
            try:
                res = self.classifier(snippet, candidate_labels=self.domain_labels)
                domain = res['labels'][0]
                conf = res['scores'][0]
                return {"domain": domain, "format": "Document", "confidence": conf}
            except Exception as e:
                print(f"Classification Pipeline Error: {e}")
                
        # API Fallback
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(content_snippet=snippet)
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"domain": "Other", "format": "Other", "confidence": 0.5})

class FormatConverterAgent(BaseProcessor):
    def process(self, document):
        """Ensures all inputs are in a consistent internal format."""
        file_path = document.get('file_path')
        if not file_path:
            # Standard document object with content list
            raw_content = document.get('content', [])
            if isinstance(raw_content, list):
                text = " ".join([item.get('content', '') for item in raw_content if isinstance(item, dict)])
            else:
                text = str(raw_content or document.get('overview', ''))
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
            text = str(document.get('content', ''))

        return {"normalized_text": text}

class DynamicMetadataAgent(BaseProcessor):
    def process(self, document):
        snippet = self.get_text_snippet(document)
        
        # 1. Identify relevant metadata fields
        schema_prompt = METADATA_SCHEMA_PROMPT_TEMPLATE.format(content_snippet=snippet)
        schema_res = self.call_asi_api([{"role": "user", "content": schema_prompt}])
        schema_data = self._safe_json_parse(schema_res, {"metadata_fields": []})
        fields = schema_data.get("metadata_fields", [])
        
        if not fields:
            return {}
            
        # 2. Extract those fields
        extract_prompt = METADATA_EXTRACTION_PROMPT_TEMPLATE.format(
            fields_list=", ".join(fields),
            content_snippet=snippet
        )
        extract_res = self.call_asi_api([{"role": "user", "content": extract_prompt}])
        return self._safe_json_parse(extract_res, {})

class SentimentAgent(BaseProcessor):
    def __init__(self, api_key):
        super().__init__(api_key)
        try:
            from transformers import pipeline
            self.sentiment_pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        except Exception:
            self.sentiment_pipe = None

    def process(self, document):
        snippet = self.get_text_snippet(document, length=1000)
        
        if self.sentiment_pipe:
            if not snippet.strip():
                return {"sentiment": "Neutral", "strength": 0.5, "confidence": 0.5}
            try:
                res = self.sentiment_pipe(snippet[:1500])[0]
                sentiment = "Positive" if res['label'] == "POSITIVE" else "Negative"
                score = res['score']
                return {"sentiment": sentiment, "strength": max(0.5, score), "confidence": score}
            except Exception as e:
                print(f"Sentiment Pipeline Error: {e}")

        # API Fallback
        prompt = SENTIMENT_PROMPT_TEMPLATE.format(text=snippet)
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"sentiment": "Neutral", "strength": 0.5, "confidence": 0.5})

class EntityLinkingAgent(BaseProcessor):
    def process(self, document):
        snippet = self.get_text_snippet(document)
        
        # 1. Identify entity types
        type_prompt = ENTITY_TYPE_PROMPT_TEMPLATE.format(content_snippet=snippet)
        type_res = self.call_asi_api([{"role": "user", "content": type_prompt}])
        type_data = self._safe_json_parse(type_res, {"entity_types": []})
        entity_types = type_data.get("entity_types", [])
        
        if not entity_types:
            return {"entities": []}
            
        # 2. Extract entities of those types
        prompt = ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(
            entity_types=", ".join(entity_types),
            content_snippet=snippet
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"entities": []})

class OpenIEAgent(BaseProcessor):
    def __init__(self, api_key):
        super().__init__(api_key)
        try:
            import spacy
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("Warning: SpaCy model not found. OpenIE agent will fallback to API.")
            self.nlp = None

    def process(self, document):
        snippet = self.get_text_snippet(document, length=1500)
        
        if self.nlp:
            if not snippet.strip():
                return {"triples": []}
            try:
                doc_spacy = self.nlp(snippet)
                triples = []
                for sent in doc_spacy.sents:
                    subj, verb, obj = None, None, None
                    for token in sent:
                        if "subj" in token.dep_: subj = token.text
                        if "obj" in token.dep_: obj = token.text
                        if token.pos_ == "VERB": verb = token.text
                    if subj and verb and obj:
                        triples.append({"subject": subj, "predicate": verb, "object": obj, "confidence": 0.8})
                return {"triples": triples[:10]}
            except Exception as e:
                print(f"SpaCy OpenIE Error: {e}")
                
        # API Fallback
        prompt = OPEN_IE_PROMPT_TEMPLATE.format(text=snippet)
        messages = [{"role": "user", "content": prompt}]
        response = self.call_asi_api(messages)
        return self._safe_json_parse(response, {"triples": []})

class DocumentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.classifier = ClassificationAgent(api_key)
        self.format_converter = FormatConverterAgent(api_key)
        self.dynamic_metadata = DynamicMetadataAgent(api_key)
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

    def process(self, document, rank_stats=None):
        # 1. Classification
        cls_res = self.classifier.process(document)
        
        # 2. Format Normalization
        norm_res = self.format_converter.process(document)
        full_text = norm_res['normalized_text']
        word_count = len(full_text.split())
        
        # 3. Rule-based analysis
        read_time_str = document.get('min_to_read', '0')
        read_time = int(re.search(r'(\d+)', str(read_time_str)).group(1)) if re.search(r'(\d+)', str(read_time_str)) else 0
        
        len_label, len_range = self.discretize_value(word_count, LENGTH_BUCKETS)
        rt_label, rt_range = self.discretize_value(read_time, READING_TIME_BUCKETS)
        
        metadata = {
            "domain": {"value": cls_res['domain'], "stv": (cls_res['confidence'], 1.0)},
            "length": {"value": len_label, "stv": self.calculate_proportional_stv(word_count, len_label, len_range)},
            "reading_time": {"value": rt_label, "stv": self.calculate_proportional_stv(read_time, rt_label, rt_range)},
            "author": {"value": document.get('author_username', 'unknown'), "stv": DETERMINISTIC_STV},
            "category": {"value": document.get('categories', [{}])[0].get('slug', 'general'), "stv": DETERMINISTIC_STV},
            "title": {"value": document.get('post_title', 'untitled'), "stv": DETERMINISTIC_STV}
        }

        # 4. Dynamic Metadata & Sentiment
        dyn_meta_res = self.dynamic_metadata.process(document)
        for key, res in dyn_meta_res.items():
            if isinstance(res, dict) and 'value' in res:
                metadata[key] = {"value": res['value'], "stv": (res.get('strength', 0.5), res.get('confidence', 0.5))}
            else:
                metadata[key] = {"value": res, "stv": UNKNOWN_STV}
            
        sent_res = self.sentiment.process(document)
        metadata['sentiment'] = {"value": sent_res.get('sentiment', 'Neutral'), "stv": (sent_res.get('strength', 0.5), sent_res.get('confidence', 0.5))}

        # 5. Entity Linking
        ent_res = self.entities.process(document)
        metadata['entities'] = ent_res.get('entities', [])

        # 6. OpenIE
        ie_res = self.openie.process(document)
        document['openie_triples'] = ie_res.get('triples', [])

        document['enriched_metadata'] = metadata
        return document
