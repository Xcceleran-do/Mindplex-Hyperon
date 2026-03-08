# ingestion/schema/field_normalizer.py

import re
from typing import Dict, List, Any


class FieldNormalizer:

    def normalize(self, records):

        normalized = []

        for r in records:

            new_record = {}

            for k, v in r.items():

                key = k.strip().lower()

                new_record[key] = v

            normalized.append(new_record)

        return normalized