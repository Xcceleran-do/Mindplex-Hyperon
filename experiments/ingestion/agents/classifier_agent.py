class ClassificationAgent:

    def classify(self, records):
        """
        Determine dataset type using schema hints.
        Returns a string domain type.
        """

        if not records:
            return "generic"

        sample = records[0]
        fields = set(sample.keys())

        if "director" in fields or "releaseYear" in fields:
            return "movie"

        if "post_title" in fields or "author_display_name" in fields:
            return "paper"

        if "text" in fields and "user" in fields:
            return "tweet"

        return "generic"