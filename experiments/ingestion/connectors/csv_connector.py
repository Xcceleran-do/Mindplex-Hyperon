# ingestion/connectors/csv_connector.py

import csv
from .base_connector import BaseConnector


class CsvConnector(BaseConnector):

    def fetch(self):
        import os

        path = self.config.get("path")
        
        if not os.path.isabs(path):
            # Resolve relative to the experiments/ingestion/ directory
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(script_dir, path)

        data = []

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                data.append(row)

        return data