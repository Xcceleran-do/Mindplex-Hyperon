class SchemaDetector:

    def detect(self, records):

        if not records:
            return {}

        sample = records[0]

        schema = {}

        for k, v in sample.items():
            schema[k] = type(v).__name__

        return schema