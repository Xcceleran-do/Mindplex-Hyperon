# experiments/ingestion/converter.py
from .config import DETERMINISTIC_STV, UNKNOWN_STV

class JsonToMetta:
    def convert(self, documents):
        metta_lines = []
        
        for document in documents:
            doc_id = f"A_{document.get('id')}"
            meta = document.get('enriched_metadata', {})
            
            def add_prop(prop, value, stv=UNKNOWN_STV):
                if value and value != "Unknown" and value != "unknown" and value != "":
                    # Sanitize string for MeTTa
                    safe_val = str(value).replace('"', '\\"').replace('\n', ' ')
                    strength, confidence = stv
                    metta_lines.append(f"(({prop} {doc_id} \"{safe_val}\") (STV {strength} {confidence}))")

            # Dynamically add all properties
            for key, prop_data in meta.items():
                # Skip specially handled keys
                if key in ['entities', 'author', 'openie_triples']:
                    continue
                    
                if isinstance(prop_data, dict) and 'value' in prop_data:
                    value = prop_data.get('value')
                    stv = prop_data.get('stv', UNKNOWN_STV)
                else:
                    value = prop_data
                    stv = UNKNOWN_STV
                    
                prop_name = str(key).replace('_', '-').replace(' ', '-').lower()
                add_prop(prop_name, value, stv)
            
            # Add authored-by as alias for author
            if 'author' in meta:
                author_data = meta['author']
                if isinstance(author_data, dict) and 'value' in author_data:
                    value = author_data['value']
                    stv = author_data['stv']
                    if value and value != "unknown":
                        safe_val = str(value).replace('"', '\\"').replace('\n', ' ')
                        strength, confidence = stv
                        metta_lines.append(f"((authored-by {doc_id} \"{safe_val}\") (STV {strength} {confidence}))")

            # Add entities/topics
            if 'entities' in meta:
                for ent in meta['entities']:
                    val = ent.get('value')
                    if val:
                        safe_val = str(val).replace('"', '\\"').replace('\n', ' ')
                        strength = ent.get('strength', 0.5)
                        confidence = ent.get('confidence', 0.5)
                        metta_lines.append(f"((has-topic {doc_id} \"{safe_val}\") (STV {strength} {confidence}))")

            # Add OpenIE triples
            openie_triples = document.get('openie_triples', [])
            if not openie_triples and 'openie_triples' in meta:
                openie_triples = meta['openie_triples']
                
            for triple in openie_triples:
                s = str(triple.get('subject', '')).replace('"', '\\"').replace('\n', ' ')
                p = str(triple.get('predicate', '')).replace('"', '\\"').replace('\n', ' ')
                o = str(triple.get('object', '')).replace('"', '\\"').replace('\n', ' ')
                conf = triple.get('confidence', 0.5)
                if s and p and o:
                    metta_lines.append(f"(({p} \"{s}\" \"{o}\") (STV 1.0 {conf}))")

        return "\n".join(metta_lines)
