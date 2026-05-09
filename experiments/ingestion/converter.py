# experiments/ingestion/converter.py
from .config import DETERMINISTIC_STV, UNKNOWN_STV
from .utils import metta_predicate, normalize_property_name, sanitize_atom_id

class JsonToMetta:
    def __init__(self, include_author_alias=True, excluded=None):
        self.include_author_alias = include_author_alias
        self.excluded = {
            normalize_property_name(item)
            for item in (excluded or ())
        }

    def convert(self, articles):
        metta_lines = []
        
        for article in articles:
            art_id = f"A_{sanitize_atom_id(article.get('id'))}"
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

            # Add every planned property. Metadata keys use Python-safe
            # underscores; MeTTa predicates use kebab-case.
            for meta_key in meta.keys():
                predicate = metta_predicate(meta_key)
                if predicate in self.excluded:
                    continue
                add_prop(predicate, meta_key)
            
            # Add authored-by as alias for author
            if self.include_author_alias and 'author' in meta and "authored-by" not in self.excluded:
                author_data = meta['author']
                if isinstance(author_data, dict) and 'value' in author_data:
                    value = author_data['value']
                    stv = author_data['stv']
                    if value and value != "unknown":
                        safe_val = str(value).replace('"', '\\"').replace('\n', ' ')
                        strength, confidence = stv
                        metta_lines.append(f"((authored-by {art_id} \"{safe_val}\") (STV {strength} {confidence}))")

        return "\n".join(metta_lines)
