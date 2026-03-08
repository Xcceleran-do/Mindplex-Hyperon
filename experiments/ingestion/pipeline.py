import os
import sys
import yaml
from dotenv import load_dotenv

# Add workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# ORIGINAL COMPONENTS
from experiments.ingestion.fetcher import MindplexFetcher
from experiments.ingestion.analyzer import ArticleAnalyzer
from experiments.ingestion.converter import JsonToMetta
from experiments.ingestion.config import DEFAULT_USERNAME

# NEW AGENTS
from experiments.ingestion.agents.classifier_agent import ClassificationAgent
from experiments.ingestion.agents.semantic_agent import SemanticAgent
from experiments.ingestion.agents.entity_linker import EntityLinker

# NEW INFRASTRUCTURE
from experiments.ingestion.connectors import get_connector
from experiments.ingestion.schema.schema_detector import SchemaDetector
from experiments.ingestion.schema.field_normalizer import FieldNormalizer
from experiments.ingestion.mapping.ontology_mapper import OntologyMapper
from experiments.ingestion.metta.metta_formatter import MettaFormatter


def load_config(path="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline():

    load_dotenv()

    config = load_config()

    print("========== INGESTION PIPELINE START ==========")

    # ------------------------------------------------
    # 1 CONNECTOR LAYER
    # ------------------------------------------------

    source_type = config["source"]["type"]
    source_config = config["source"]["config"]

    connector = get_connector(source_type, source_config)

    raw_data = connector.fetch()

    print(f"Fetched {len(raw_data)} records")

    if not raw_data:
        print("No data found.")
        return

    # ------------------------------------------------
    # 2 SCHEMA DETECTION
    # ------------------------------------------------

    detector = SchemaDetector()
    schema_info = detector.detect(raw_data)

    print("Detected schema:", schema_info)

    # ------------------------------------------------
    # 3 FIELD NORMALIZATION
    # ------------------------------------------------

    normalizer = FieldNormalizer()
    normalized_data = normalizer.normalize(raw_data)

    # ------------------------------------------------
    # 4 CLASSIFICATION AGENT
    # ------------------------------------------------

    classifier = ClassificationAgent()
    domain = classifier.classify(normalized_data)

    print("Detected domain:", domain)

    # ------------------------------------------------
    # 5 SEMANTIC AGENT (LLM)
    # ------------------------------------------------

    semantic_agent = SemanticAgent()
    enriched_data = semantic_agent.process(normalized_data)

    # ------------------------------------------------
    # 6 ENTITY LINKING
    # ------------------------------------------------

    linker = EntityLinker()
    linked_data = linker.process(enriched_data)

    # ------------------------------------------------
    # 7 ONTOLOGY MAPPING
    # ------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_file = os.path.join(script_dir, f"mapping/{domain}_mapping.yaml")
    
    if not os.path.exists(mapping_file):
        print(f"Mapping file {mapping_file} not found, falling back to generic_mapping.yaml")
        mapping_file = os.path.join(script_dir, "mapping/generic_mapping.yaml")

    mapper = OntologyMapper(mapping_file)

    triples = mapper.map_records(linked_data)

    # ------------------------------------------------
    # 8 METTA FORMATTER
    # ------------------------------------------------

    output_file = config.get("output_file", "output/knowledge.metta")
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)

    formatter = MettaFormatter()

    formatter.write(triples, output_file)

    print("Knowledge graph updated.")

    print("========== PIPELINE FINISHED ==========")


def run_ingestion(username=None):
    import time
    
    load_dotenv()
    
    if not username:
        username = os.getenv("MINDPLEX_USERNAME", DEFAULT_USERNAME)
        
    print(f"Starting ingestion pipeline for user: {username}")

    print("1. Fetching articles...")
    fetcher = MindplexFetcher(username=username)
    articles = fetcher.fetch_all(limit=50) # Start small for testing
    print(f"   Fetched {len(articles)} articles.")
    
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

    print("4. Converting to MeTTa...")
    converter = JsonToMetta()
    metta_output = converter.convert(enriched_articles)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "../../experiments/atomspace_visualizer/public/data.metta")
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
    parser.add_argument("--legacy", action="store_true", help="Run the legacy MindplexFetcher ingestion pipeline")
    
    args = parser.parse_args()
    
    if args.legacy:
        run_ingestion()
    else:
        run_pipeline()

if __name__ == "__main__":
    main()