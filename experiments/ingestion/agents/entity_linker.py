class EntityLinker:
    def __init__(self):
        pass

    def process(self, records):
        """
        Mock entity linking process: matches extracted terms to known ontologies.
        Returns the records with linked entities.
        """
        enriched = []
        for r in records:
            # Mock implementation of entity linking
            if "entities" not in r:
                r["entities"] = []
            
            # Example logic: add a linked ontology flag
            r["entities_linked"] = True
            enriched.append(r)

        return enriched
