import yaml


class OntologyMapper:

    def __init__(self, mapping_file):
        import os
        if not os.path.isabs(mapping_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # mapping_file usually comes as 'mapping/xxx.yaml', so go up one level
            mapping_file = os.path.join(os.path.dirname(script_dir), mapping_file)
            
        with open(mapping_file) as f:
            self.mapping = yaml.safe_load(f)

    def map_records(self, records):

        triples = []

        id_field = self.mapping["id_field"]

        for r in records:

            entity_id = f"E_{r.get(id_field)}"

            for field, predicate in self.mapping["mappings"].items():

                if field in r and r[field]:

                    value = str(r[field]).replace('"', '')

                    triples.append(
                        f"({predicate} {entity_id} \"{value}\")"
                    )

        return triples