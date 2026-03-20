import json
from experiments.ingestion.analyzer import DocumentAnalyzer
from experiments.ingestion.converter import JsonToMetta
import os
from dotenv import load_dotenv

def test_mixed_inputs():
    load_dotenv()
    api_key = os.getenv("ASI_API_KEY", "test-key")
    if api_key == "test-key":
        print("Warning: ASI_API_KEY not found in .env, using dummy key!")
        
    analyzer = DocumentAnalyzer(api_key=api_key)
    converter = JsonToMetta()
    
    docs = [
        {
            "id": "pdf-1",
            "post_title": "Attention Is All You Need",
            "content": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        },
        {
            "id": "tweet-1",
            "post_title": "Daily Update",
            "content": "Just had the best coffee at Downtown Cafe! ☕️ The new espresso blend is fire. Highly recommend checking it out if you're in the area. #coffee #morning"
        },
        {
            "id": "recipe-1",
            "post_title": "Classic Pancakes",
            "content": "Ingredients: 1 cup flour, 2 tbsp sugar, 1 cup milk, 1 egg. \nInstructions: Mix dry ingredients. Whisk egg and milk. Combine. Cook on griddle until bubbly, then flip."
        }
    ]
    
    print("Starting Analysis...")
    analyzed_docs = []
    for doc in docs:
        print(f"Analyzing {doc['id']}...")
        result = analyzer.process(doc)
        analyzed_docs.append(result)
        print(f"Dynamic Metadata extracted: {list(result.get('enriched_metadata', {}).keys())}")
        
    print("\nStarting Conversion...")
    metta_output = converter.convert(analyzed_docs)
    
    print("\n--- MeTTa Output ---\n")
    print(metta_output)

if __name__ == "__main__":
    test_mixed_inputs()
