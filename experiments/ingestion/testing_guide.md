# Mindplex Ingestion Pipeline Testing Guide

Welcome to the `experiments/ingestion` directory. This module acts as the Source-Agnostic extraction pipeline responsible for scraping arbitrary APIs, classifying the metadata, processing contextual properties using LLMs (Gemini), and outputting relational `.metta` graphs for the Mindplex application.

## 1. Setup

Before running the pipeline, guarantee your virtual environment is loaded with the necessary packages:

```bash
# Create a venv (if you haven't)
python3 -m venv venv
source venv/bin/activate

# Install the extraction dependencies
pip install -r requirements.txt
```

You must also configure your API keys. Copy the provided `.env` format:

```bash
# .env file
MINDPLEX_API_TOKEN="XYZ"
ASI_API_KEY="XYZ"
GEMINI_API_KEY="AI..."
```

## 2. Using the Generic Pipeline (LLMs, Connectors, classification)

The new pipeline is orchestrated entirely by `config.yaml`. 
For instance, to run a test on JSON data via an API Endpoint (`jsonplaceholder`), your configuration should look like this:

```yaml
# config.yaml (Excerpt)
source:
  type: json_api
  config:
    url: "https://jsonplaceholder.typicode.com/posts"
    limit: 10 # Start small for testing!

mapping:
  # The Mapping will default to generic_mapping.yaml if the schema matches no existing domains
  file: mapping/generic_mapping.yaml

output_file: output/knowledge.metta
```

Once the configuration is set up, run the standard pipeline:
```bash
python3 experiments/ingestion/pipeline.py
```
> This will orchestrate the fetching, normalization, domain classification, and graph mapping, pushing tuples to `output/knowledge.metta`.

## 3. Legacy Mindplex Script

The original code you saw that specifically targeted `MindplexFetcher`, `ArticleAnalyzer`, and `JsonToMetta` (converting directly into `experiments/atomspace_visualizer/public/data.metta`) actually had different business logic.

If you need to retrieve actual Mindplex production articles over `--legacy` infrastructure, simply run:

```bash
python3 experiments/ingestion/pipeline.py --legacy
```

**Note:** The legacy pipeline relies purely on fetching username-specific content and assumes valid `MINDPLEX_API_TOKEN` and `ASI_API_KEY` environmental variables are heavily injected to process the analytics without rate limiting.
