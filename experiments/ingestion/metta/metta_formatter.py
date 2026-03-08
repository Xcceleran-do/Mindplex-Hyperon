class MettaFormatter:

    def write(self, triples, output_file):
        import os
        
        existing_triples = set()
        
        # 1. Read existing triples if the file exists to prevent duplicates
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    existing_triples.add(line.strip())

        # 2. Append only unique ones
        with open(output_file, "a", encoding="utf-8") as f:
            for t in triples:
                clean_t = t.strip()
                if clean_t and clean_t not in existing_triples:
                    f.write(clean_t + "\n")
                    existing_triples.add(clean_t)