# experiments/ingestion/converter.py

class JsonToMetta:
    def convert(self, articles):
        metta_lines = []
        
        for article in articles:
            art_id = f"A_{article.get('id')}"
            meta = article.get('enriched_metadata', {})
            
            # Helper to add line
            def add_prop(prop, value):
                if value and value != "Unknown":
                    # Sanitize string for MeTTa
                    safe_val = str(value).replace('"', '\\"').replace('\n', ' ')
                    metta_lines.append(f"({prop} {art_id} \"{safe_val}\")")

            # Intrinsic
            add_prop("length", meta.get('length'))
            add_prop("reading-time", meta.get('reading_time'))
            add_prop("tone", meta.get('tone'))
            add_prop("audience-expertise", meta.get('audience_expertise'))
            add_prop("content-type", meta.get('content_type'))
            add_prop("date-period", meta.get('date_period'))
            add_prop("primary-goal", meta.get('primary_goal'))
            
            # Category is a list in the JSON, take the first one or slug
            # "categories": [{"name": "AI", "slug": "ai"}]
            categories = article.get('categories', [])
            if categories:
                cat_slug = categories[0].get('slug')
                add_prop("category", cat_slug)

            # Relational
            add_prop("popularity", meta.get('popularity'))
            add_prop("engagement", meta.get('engagement'))
            add_prop("audience-sentiment", meta.get('audience_sentiment'))
            
            # Author
            author_username = article.get('author_username', 'unknown')
            add_prop("authored-by", author_username)
            
            # Title (optional, good for debugging)
            add_prop("title", article.get('post_title'))

        return "\n".join(metta_lines)
