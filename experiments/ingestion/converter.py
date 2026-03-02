# experiments/ingestion/converter.py
from .config import DETERMINISTIC_STV, UNKNOWN_STV

class JsonToMetta:
    def convert(self, articles):
        metta_lines = []
        
        for article in articles:
            art_id = f"A_{article.get('id')}"
            meta = article.get('enriched_metadata', {})
            
            # Helper to add line with STV
            def add_prop(prop, meta_key, default_stv=UNKNOWN_STV):
                if meta_key in meta:
                    prop_data = meta[meta_key]
                    if isinstance(prop_data, dict) and 'value' in prop_data and 'stv' in prop_data:
                        value = prop_data['value']
                        stv = prop_data['stv']
                    else:
                        # Fallback for old format
                        value = prop_data
                        stv = default_stv
                    
                    if value and value != "Unknown":
                        # Sanitize string for MeTTa
                        safe_val = str(value).replace('"', '\\"').replace('\n', ' ')
                        strength, confidence = stv
                        metta_lines.append(f"(({prop} {art_id} \"{safe_val}\") (STV {strength} {confidence}))")

            # Add all properties with their STVs
            add_prop("length", "length")
            add_prop("reading-time", "reading_time")
            add_prop("tone", "tone")
            add_prop("audience-expertise", "audience_expertise")
            add_prop("content-type", "content_type")
            add_prop("date-period", "date_period")
            add_prop("primary-goal", "primary_goal")
            add_prop("audience-sentiment", "audience_sentiment")
            add_prop("popularity", "popularity")
            add_prop("engagement", "engagement")
            add_prop("author", "author")
            add_prop("category", "category")
            add_prop("title", "title")
            add_prop("topic", "topic")
            
            # Add authored-by as alias for author
            if 'author' in meta:
                author_data = meta['author']
                if isinstance(author_data, dict) and 'value' in author_data:
                    value = author_data['value']
                    stv = author_data['stv']
                    if value and value != "unknown":
                        safe_val = str(value).replace('"', '\\"').replace('\n', ' ')
                        strength, confidence = stv
                        metta_lines.append(f"((authored-by {art_id} \"{safe_val}\") (STV {strength} {confidence}))")

        return "\n".join(metta_lines)
