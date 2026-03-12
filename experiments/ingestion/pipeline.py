# experiments/ingestion/pipeline.py
import os
import sys
from dotenv import load_dotenv

# Add workspace root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.ingestion.fetcher import MindplexFetcher, FileFetcher
from experiments.ingestion.analyzer import ArticleAnalyzer
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
        print(f"1. Fetching articles for user: {username}...")
        fetcher = MindplexFetcher(username=username)
        
    articles = fetcher.fetch_all(limit=50) # Start small for testing
    print(f"   Fetched {len(articles)} items.")
    
    if not articles:
        print("No articles found. Exiting.")
        return {"status": "error", "message": "No articles found"}

    print("2. Calculating Rankings...")
    # Sort by views descending to determine popularity rank
    # Ensure views is int
    for art in articles:
        try:
            art['views'] = int(art.get('views', 0))
        except:
            art['views'] = 0
            
    articles.sort(key=lambda x: x.get('views', 0), reverse=True)
    rank_map = {art.get('id'): i+1 for i, art in enumerate(articles)}

    print("3. Analyzing and Enriching (this may take time)...")
    # Use ASI_API_KEY as per new requirement
    api_key = os.getenv("ASI_API_KEY")
    analyzer = ArticleAnalyzer(api_key=api_key)
    
    enriched_articles = []
    for i, art in enumerate(articles):
        print(f"   Processing [{i+1}/{len(articles)}]: {art.get('post_title', 'Untitled')[:30]}...")
        enriched = analyzer.process(art, rank_stats=rank_map)
        enriched_articles.append(enriched)
        
        # Sleep to avoid hitting ASI API rate limits if necessary
        # if i < len(articles) - 1:
        #     print("   Sleeping for rate limit...")
        #     time.sleep(1)

    print("4. Converting to MeTTa...")
    converter = JsonToMetta()
    metta_output = converter.convert(enriched_articles)

    output_path = "experiments/atomspace_visualizer/public/data.metta"
    print(f"5. Saving to {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(metta_output)
    
    print("Done!")
    return {"status": "success", "message": f"Ingested {len(articles)} articles for user {username}"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mindplex Ingestion Pipeline")
    parser.add_argument("--username", help="Mindplex username to fetch articles for")
    parser.add_argument("--source", help="Local directory or file path for ingestion")
    args = parser.parse_args()
    
    run_ingestion(username=args.username, source=args.source)

if __name__ == "__main__":
    main()
