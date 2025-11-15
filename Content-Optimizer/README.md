# Content Optimization Recommender (Neo4j + GraphSAGE + Hybrid Ranker)

Small scaffold for a recommender that combines Neo4j GDS GraphSAGE embeddings, sentence-transformer text embeddings, and a hybrid LightGBM ranker.

Quickstart
- Copy `.env.sample` to `.env` and edit credentials.
- Start Neo4j with Docker Compose: `docker-compose -f infra/docker-compose.yml up -d`.
- Run Cypher import scripts in `neo4j/cypher/` to create schema and sample data.
- Run the notebook `backend/notebooks/00_pipeline_demo.ipynb` to demo embedding & training steps.
- Train a model using `backend/train/train_ranker.py` after preparing data with `prepare_dataset.py`.
- Start the API: `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000`.

Project layout
See `step_by_step_guide.md` for a full list of files and the intended pipeline.

Ports and creds
- Neo4j Bolt: 7687, HTTP: 7474
- Default ADMIN_TOKEN and NEO4J_PASSWORD live in `.env` (do not commit secrets)

Notes
- This repository currently contains scaffolding and placeholder implementations. Use the step-by-step guide to implement each module.
