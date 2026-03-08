class TripleBuilder:

    @staticmethod
    def build(triple_dict):
        predicate = triple_dict["predicate"]
        subject = triple_dict["subject"]
        obj = triple_dict["object"]

        if isinstance(obj, str):
            obj = f'"{obj}"'

        return f"({predicate} {subject} {obj})"