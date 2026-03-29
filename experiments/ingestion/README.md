# Standalone Multi-Agent Ingestion Tool

This module ingests heterogeneous resources (JSON, CSV, text, HTML, URL, or directory), lets an agent infer which properties are relevant from the observed schema/value patterns, converts continuous values into symbolic buckets, assigns STV values, and writes MeTTa triples to a dedicated `.metta` file.

## Output format

Each generated line follows:

```metta
((predicate subject "object") (STV confidence strength))
```

Example:

```metta
((length A_16624 "Medium") (STV 0.694 0.9))
```

## Agent pipeline

1. `SourceResolutionAgent`: Resolves files, URLs, directories, and fallback pseudo-sources.
2. `RecordExtractionAgent`: Converts payloads into normalized record dictionaries.
3. `SentimentAnalysisAgent`: Scores audience sentiment from textual evidence.
4. `ContentClassificationAgent`: Classifies content style (instructional/analytical/opinionated/informational).
5. `SemanticParsingAgent`: Extracts semantic keywords and condensed topical signals.
6. `RecommendationSignalAgent`: Derives recommendation-oriented signals (utility, novelty, complexity).
7. `SchemaProfilerAgent`: Profiles coverage, type shape, cardinality, and numeric distributions.
8. `PropertySelectionAgent`: Chooses properties with sufficient observed coverage (no hardcoded domain list).
9. `DiscretizationAgent`: Applies generic quantile bins for continuous fields.
10. `TripleConstructionAgent`: Builds `(predicate subject object)` facts with STV.
11. `FactValidationAgent`: Drops malformed or out-of-range STV facts.
12. `FactPersistenceAgent`: Writes validated facts to `.metta`.

All agents use explicit deterministic tools through a `ToolRouter`, so the pipeline does not depend on one general LLM for all tasks.

## Module Layout

- `pipeline.py`: Thin public API (`run_ingestion`) used by CLI/API.
- `orchestrator.py`: Agent orchestration, execution guards, and runtime telemetry.
- `agent_registry.py`: Ordered, pluggable registry used to compose the agent pipeline.
- `tool_router.py`: Deterministic tools for source loading, parsing, profiling, scoring, and writing.
- `models.py`: Shared dataclasses (`IngestionConfig`, `IngestionState`, `Fact`, etc.).
- `constants.py`: Shared domain-agnostic constants and token sets.
- `agents/base.py`: Base agent interface.
- `agents/io_agents.py`: Source resolution, record extraction, and persistence agents.
- `agents/analysis_agents.py`: Sentiment, classification, semantic parsing, recommendation signal agents.
- `agents/schema_profiler_agent.py`: Schema profile agent.
- `agents/property_selection_agent.py`: Property selection agent.
- `agents/discretization_agent.py`: Numeric discretization agent.
- `agents/triple_construction_agent.py`: Triple builder agent.
- `agents/fact_validation_agent.py`: Fact validation agent.
- `agents/transformation_agents.py`: Compatibility exports for transformation agents.
- `multimedia_ingester.py`: Placeholder extension point for multimedia ingestion.

## Extending With New Agents

1. Implement a class that extends `agents.base.Agent`.
2. Register it using `AgentRegistry.register(...)`, `insert_before(...)`, or `insert_after(...)`.
3. Construct `MultiAgentIngestionOrchestrator` with your customized registry.

This keeps feature growth modular and avoids editing the core orchestrator loop.

## Run from repo root

```bash
python -m experiments.ingestion.cli --input experiments/atomspace_visualizer/public/data.metta --output experiments/ingestion/outputs/from_existing_data.metta
```

For structured ingestion:

```bash
python -m experiments.ingestion.cli --input datasets/MIND/train/news.tsv --output experiments/ingestion/outputs/mind_data.metta
```

For API compatibility fallback:

```bash
python -m experiments.ingestion.cli --username Hruy --output experiments/ingestion/outputs/user_profile.metta
```

Full runtime report:

```bash
python -m experiments.ingestion.cli \
  --input datasets/MIND/train/news.tsv \
  --output experiments/ingestion/outputs/mind_data.metta \
  --min-property-coverage 0.2 \
  --include-agent-reports
```

## Real Data Test: Kaggle Book-Crossing

Dataset page:
- `https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset`

1. Download and extract the dataset so a folder contains:
- `BX-Books.csv`
- `BX-Users.csv`
- `BX-Book-Ratings.csv`

2. Run the dedicated adapter + ingestion pipeline:

```bash
python -m experiments.ingestion.run_bookcrossing_ingestion \
  --dataset-dir <PATH_TO_EXTRACTED_FOLDER> \
  --output-metta experiments/ingestion/outputs/bookcrossing_data.metta \
  --prepared-jsonl experiments/ingestion/outputs/bookcrossing_prepared.jsonl
```

3. Optional test:

```bash
python -m unittest experiments.ingestion.tests.bookcrossing_adapter_test
```

## Notes

- Multimedia ingestion is reserved for a future phase and currently exposed as a placeholder extension point in `multimedia_ingester.py`.
- STV values use source reliability as strength and confidence from extraction mode/model certainty.
- Continuous values are discretized so they are suitable for symbolic reasoning.
- No fixed list like sentiment/read-time/category is required; extracted properties are inferred from the data itself.
- The output JSON includes per-agent telemetry (`agent_reports`) for observability and debugging.
