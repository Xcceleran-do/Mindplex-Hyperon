# experiments/ingestion/analyzer.py
import datetime
import json
import re
import requests
from .config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS, 
    ANALYSIS_PROMPT_TEMPLATE
)

ASI_BASE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-mini"

class ArticleAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key:
            print("Warning: No ASI API key provided. AI enrichment will be skipped.")

    def call_asi_api(self, messages):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": ASI_MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            response = requests.post(ASI_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ASI API Error: {e}")
            return {"error": str(e)}

    def discretize_value(self, value, buckets):
        if value is None:
            return "Unknown"
        for label, condition in buckets.items():
            try:
                if condition(value):
                    return label
            except:
                continue
        return "Unknown"

    def calculate_date_period(self, date_str):
        if not date_str:
            return "Unknown"
        try:
            # Format: "2024-11-04 18:02:23"
            pub_date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            # Make it timezone aware (UTC) for comparison if needed, or just naive
            now = datetime.datetime.now()
            days_diff = (now - pub_date).days

            if days_diff <= 7: return "Current"
            if days_diff <= 30: return "Recent"
            if days_diff <= 90: return "Older"
            return "Archived"
        except ValueError:
            return "Unknown"

    def parse_read_time(self, read_time_str):
        """Extracts minutes from string like '20 min read.'"""
        if not read_time_str:
            return 0
        match = re.search(r'(\d+)', str(read_time_str))
        if match:
            return int(match.group(1))
        return 0

    def enrich_with_ai(self, article):
        """Uses ASI API to extract Tone, Audience Expertise, etc."""
        if not self.api_key:
            return {
                "tone": "Unknown", "audience_expertise": "Unknown", 
                "content_type": "Unknown", "primary_goal": "Unknown", 
                "audience_sentiment": "Unknown"
            }

        # Prepare content snippet (first 2000 chars)
        raw_content = article.get('content', [])
        if isinstance(raw_content, list):
            # Extract text from list of dicts
            text_content = " ".join([item.get('content', '') for item in raw_content if isinstance(item, dict) and item.get('type') == 'p'])
        else:
            text_content = str(raw_content or article.get('overview', ''))

        snippet = text_content[:2000]
        
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            title=article.get('post_title', ''),
            content_snippet=snippet
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes articles and outputs JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response_data = self.call_asi_api(messages)
            
            text = None
            if 'choices' in response_data and response_data['choices']:
                 text = response_data['choices'][0]['message'].get('content', '')
            
            if not text:
                raise Exception("No content in response")

            # Clean up json string if markdown is present
            text = text.replace('```json', '').replace('```', '')
            ai_metadata = json.loads(text)
            return ai_metadata
        except Exception as e:
            print(f"AI Analysis failed for article {article.get('id')}: {e}")
            return {
                "tone": "Unknown", "audience-expertise": "Unknown", 
                "content_type": "Unknown", "primary_goal": "Unknown", 
                "audience_sentiment": "Unknown"
            }

    def process(self, article, rank_stats=None):
        """Main processing method."""
        # 1. Intrinsic (Calculated)
        # Word count might not be directly available, estimate from content length
        raw_content = article.get('content', [])
        if isinstance(raw_content, list):
            full_text = " ".join([item.get('content', '') for item in raw_content if isinstance(item, dict)])
        else:
            full_text = str(raw_content or '')
            
        word_count = len(full_text.split())
        
        read_time_str = article.get('min_to_read', '0')
        read_time = self.parse_read_time(read_time_str)
        
        metadata = {
            "length": self.discretize_value(word_count, LENGTH_BUCKETS),
            "reading_time": self.discretize_value(read_time, READING_TIME_BUCKETS),
            "date_period": self.calculate_date_period(article.get('published_timestamp')),
        }

        # 2. Relational (Calculated)
        views = int(article.get('views', 0))
        likes = int(article.get('likes', 0))
        comments = int(article.get('comments', 0))
        # Avoid division by zero
        engagement_ratio = likes + views + comments
        
        metadata['engagement'] = self.discretize_value(engagement_ratio, ENGAGEMENT_BUCKETS)
        
        # Popularity (Rank based)
        if rank_stats:
            rank = rank_stats.get(article.get('id'), 101)
            if rank <= 10: metadata['popularity'] = "Top_10"
            elif rank <= 40: metadata['popularity'] = "High"
            elif rank <= 70: metadata['popularity'] = "Medium"
            else: metadata['popularity'] = "Low"
        else:
             metadata['popularity'] = "Unknown"

        # 3. AI Enrichment
        ai_data = self.enrich_with_ai(article)
        metadata.update(ai_data)

        # Merge with original article
        article['enriched_metadata'] = metadata
        return article
