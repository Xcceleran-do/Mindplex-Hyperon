# experiments/ingestion/analyzer.py
import datetime
import json
import re
import requests
from .config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS, 

    RETENTION_BUCKETS, ANALYSIS_PROMPT_TEMPLATE,
    DETERMINISTIC_STV, AI_FAILURE_STV, UNKNOWN_STV
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
        """Discretize value and return (bucket, stv_range) tuple."""
        if value is None:
            return "Unknown", UNKNOWN_STV
        for label, (condition, stv_range) in buckets.items():
            try:
                if condition(value):
                    return label, stv_range
            except:
                continue
        return "Unknown", UNKNOWN_STV
    
    def calculate_proportional_stv(self, value, bucket_label, stv_range, bucket_bounds=None):
        """Calculate proportional STV within bucket range."""
        if stv_range[0] == stv_range[1]:
            return stv_range
        
        if bucket_bounds:
            min_val, max_val = bucket_bounds
            if max_val == min_val:
                normalized = 0.5
            else:
                normalized = (value - min_val) / (max_val - min_val)
                normalized = max(0, min(1, normalized))  # Clamp to [0,1]
        else:
            # Simple linear interpolation within range
            normalized = 0.5  # Default to middle if no bounds
        
        strength = stv_range[0] + normalized * (stv_range[1] - stv_range[0])
        confidence = 0.9  # High confidence for rule-based calculations
        return (round(strength, 3), round(confidence, 3))

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
        """Uses ASI API to extract Tone, Audience Expertise, etc. with AI-assigned STV values."""
        if not self.api_key:
            return {
                "tone": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "audience_expertise": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "content_type": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "primary_goal": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "audience_sentiment": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}
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
            {"role": "system", "content": "You are a helpful assistant that analyzes articles and outputs JSON with AI-assigned STV values."},
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
            
            # Validate and normalize AI response with STV values
            normalized_metadata = {}
            for key in ["tone", "audience_expertise", "content_type", "primary_goal", "audience_sentiment"]:
                if key in ai_metadata:
                    ai_data = ai_metadata[key]
                    if isinstance(ai_data, dict) and 'value' in ai_data and 'strength' in ai_data and 'confidence' in ai_data:
                        # Validate strength and confidence are in valid range
                        strength = max(0.0, min(1.0, float(ai_data['strength'])))
                        confidence = max(0.0, min(1.0, float(ai_data['confidence'])))
                        normalized_metadata[key] = {
                            "value": ai_data['value'],
                            "strength": strength,
                            "confidence": confidence
                        }
                    else:
                        # Fallback for old format or missing STV components
                        normalized_metadata[key] = {
                            "value": str(ai_data) if not isinstance(ai_data, dict) else ai_data.get('value', 'Unknown'),
                            "strength": 0.5,  # Default moderate strength
                            "confidence": 0.5  # Default moderate confidence
                        }
                else:
                    normalized_metadata[key] = {"value": "Unknown", "strength": 0.0, "confidence": 0.0}
            
            return normalized_metadata
            
        except Exception as e:
            print(f"AI Analysis failed for article {article.get('id')}: {e}")
            return {
                "tone": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "audience_expertise": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "content_type": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "primary_goal": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}, 
                "audience_sentiment": {"value": "Unknown", "strength": 0.0, "confidence": 0.0}
            }


    def process(self, article, rank_stats=None):
        """Main processing method."""
        # 1. Intrinsic (Calculated) with proportional STVs
        raw_content = article.get('content', [])
        if isinstance(raw_content, list):
            full_text = " ".join([item.get('content', '') for item in raw_content if isinstance(item, dict)])
        else:
            full_text = str(raw_content or '')
            
        word_count = len(full_text.split())
        
        read_time_str = article.get('min_to_read', '0')
        read_time = self.parse_read_time(read_time_str)
        
        # Discretize with STV ranges
        length_bucket, length_stv_range = self.discretize_value(word_count, LENGTH_BUCKETS)
        reading_time_bucket, reading_time_stv_range = self.discretize_value(read_time, READING_TIME_BUCKETS)
        
        # Calculate proportional STVs
        length_bounds = (0, 500) if length_bucket == "Short" else (
                       (500, 1500) if length_bucket == "Medium" else (1500, 3000))
        length_stv = self.calculate_proportional_stv(word_count, length_bucket, length_stv_range, length_bounds)
        
        reading_time_bounds = (0, 2) if reading_time_bucket == "Very_Short" else (
                            (2, 5) if reading_time_bucket == "Short" else (
                            (5, 10) if reading_time_bucket == "Medium" else (10, 20)))
        reading_time_stv = self.calculate_proportional_stv(read_time, reading_time_bucket, reading_time_stv_range, reading_time_bounds)
        
        metadata = {
            "length": {"value": length_bucket, "stv": length_stv},
            "reading_time": {"value": reading_time_bucket, "stv": reading_time_stv},
            "date_period": {"value": self.calculate_date_period(article.get('published_timestamp')), "stv": DETERMINISTIC_STV},
        }

        # 2. Relational (Calculated) with proportional STVs
        views = int(article.get('views', 0))
        likes = int(article.get('likes', 0))
        comments = int(article.get('comments', 0))
        engagement_score = likes + views + comments  # Simplified engagement score
        
        engagement_bucket, engagement_stv_range = self.discretize_value(engagement_score, ENGAGEMENT_BUCKETS)
        
        # Calculate proportional STV for engagement
        engagement_bounds = (0, 30) if engagement_bucket == "Low" else (
                          (30, 50) if engagement_bucket == "Medium" else (
                          (50, 100) if engagement_bucket == "High" else (100, 1000)))
        engagement_stv = self.calculate_proportional_stv(engagement_score, engagement_bucket, engagement_stv_range, engagement_bounds)
        
        metadata['engagement'] = {"value": engagement_bucket, "stv": engagement_stv}
        
        # Popularity (Rank based) with deterministic STV
        if rank_stats:
            rank = rank_stats.get(article.get('id'), 101)
            if rank <= 10: 
                popularity_val = "Top_10"
            elif rank <= 40: 
                popularity_val = "High"
            elif rank <= 70: 
                popularity_val = "Medium"
            else: 
                popularity_val = "Low"
            metadata['popularity'] = {"value": popularity_val, "stv": DETERMINISTIC_STV}
        else:
            metadata['popularity'] = {"value": "Unknown", "stv": UNKNOWN_STV}


        # 3. AI Enrichment with AI-assigned STV values
        ai_data = self.enrich_with_ai(article)
        
        # Use AI's own STV assessment directly
        for key, ai_result in ai_data.items():
            value = ai_result['value']
            strength = ai_result['strength']
            confidence = ai_result['confidence']
            
            if value == "Unknown" or strength == 0.0:
                # Complete failure
                stv = AI_FAILURE_STV
            else:
                # Use AI's own assessment
                stv = (strength, confidence)
            
            metadata[key] = {"value": value, "stv": stv}
        
        # Add mandatory core facts with deterministic STV
        metadata['author'] = {"value": article.get('author_username', 'unknown'), "stv": DETERMINISTIC_STV}
        
        categories = article.get('categories', [])
        if categories:
            category_value = categories[0].get('slug', 'general')
        else:
            category_value = 'general'
        metadata['category'] = {"value": category_value, "stv": DETERMINISTIC_STV}
        
        metadata['title'] = {"value": article.get('post_title', 'untitled'), "stv": DETERMINISTIC_STV}
        
        # Topic defaults to Unknown with weak STV
        if 'topic' not in metadata:
            metadata['topic'] = {"value": "Unknown", "stv": UNKNOWN_STV}

        # Merge with original article
        article['enriched_metadata'] = metadata
        return article
