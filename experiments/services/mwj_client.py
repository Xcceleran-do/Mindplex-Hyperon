import requests

class MWJClient:
    def __init__(self, url="http://localhost:5001/metta"):
        self.url = url
        self.headers = {
            "Content-Type": "text/plain; charset=utf-8"
        }
    def process_metta_string(self, query):
        print("\n=== MWJ QUERY ===")
        print(query)
        response = requests.post(
            self.url,
            headers=self.headers,
            data=query.encode("utf-8"),
            timeout=30
        )
        print("\n=== MWJ RESPONSE ===")
        print(response.text)
        response.raise_for_status()
        return response.text

