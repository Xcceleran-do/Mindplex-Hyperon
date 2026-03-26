# experiments/ingestion/pipeline.py
import os
import sys
from dotenv import load_dotenv

# Add workspace root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.ingestion.fetcher import MindplexFetcher, FileFetcher
from experiments.ingestion.analyzer import DocumentAnalyzer
from experiments.ingestion.converter import JsonToMetta
from experiments.ingestion.config import DEFAULT_USERNAME

def run_ingestion(username=None, source=None):
    load_dotenv()
    
    if not username:
        username = os.getenv("MINDPLEX_USERNAME", DEFAULT_USERNAME)
        
    print(f"Starting ingestion pipeline. Target: {source or username}")

    if source:
        print(f"1. Fetching local files from {source}...")
        fetcher = FileFetcher(source)
    else:
        print(f"1. Fetching documents for user: {username}...")
        fetcher = MindplexFetcher(username=username)
        
    documents = fetcher.fetch_all(limit=50) # Start small for testing
    print(f"   Fetched {len(documents)} items.")
    
    if not documents:
        print("No documents found. Exiting.")
        return {"status": "error", "message": "No documents found"}

    print("2. Calculating Rankings...")
    # Sort by views descending to determine popularity rank
    # Ensure views is int
    for doc in documents:
        try:
            doc['views'] = int(doc.get('views', 0))
        except:
            doc['views'] = 0
            
    documents.sort(key=lambda x: x.get('views', 0), reverse=True)
    rank_map = {doc.get('id'): i+1 for i, doc in enumerate(documents)}

    print("3. Analyzing and Enriching (this may take time)...")
    # Use ASI_API_KEY as per new requirement
    api_key = os.getenv("ASI_API_KEY")
    analyzer = DocumentAnalyzer(api_key=api_key)
    
    enriched_documents = []
    for i, doc in enumerate(documents):
        print(f"   Processing [{i+1}/{len(documents)}]: {doc.get('post_title', 'Untitled')[:30]}...")
        enriched = analyzer.process(doc, rank_stats=rank_map)
        enriched_documents.append(enriched)
        
        # Sleep to avoid hitting ASI API rate limits if necessary
        # if i < len(documents) - 1:
        #     print("   Sleeping for rate limit...")
        #     time.sleep(1)

    print("4. Converting to MeTTa...")
    converter = JsonToMetta()
    metta_output = converter.convert(enriched_documents)

    output_path = "experiments/atomspace_visualizer/public/data.metta"
    print(f"5. Saving to {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(metta_output)
    
    print("Done!")
    return {"status": "success", "message": f"Ingested {len(documents)} documents for user {username}"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mindplex Ingestion Pipeline")
    parser.add_argument("--username", help="Mindplex username to fetch documents for")
    parser.add_argument("--source", help="Local directory or file path for ingestion")
    args = parser.parse_args()
    
    run_ingestion(username=args.username, source=args.source)

if __name__ == "__main__":
    main()
