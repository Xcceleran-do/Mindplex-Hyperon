
# Content Optimization Recommender (Neo4j + GraphSAGE + Hybrid Ranker)

This repository is a scaffold for a full-stack content-optimizer recommender that combines
collaborative signals from a Neo4j graph (GraphSAGE embeddings via the Graph Data Science plugin)
with content signals (sentence-transformer text embeddings) and a downstream hybrid ranker
(e.g., LightGBM). The goal is to suggest content ideas and optimizations to creators.

Contents (high level)

- `infra/`: Docker Compose and Neo4j config
- `neo4j/`: import CSVs and `cypher/` scripts (schema, sample data, GDS GraphSAGE projection)
- `backend/`: FastAPI service, training scripts, and helper services
- `frontend/`: minimal React UI (Vite)
- `tests/`: simple end-to-end smoke tests

Prerequisites

- Docker & Docker Compose (for Neo4j)
- Python 3.10+ (for backend scripts and API)
- Node.js & npm/yarn (to run the frontend dev server)

.env (secrets)

Copy `.env.sample` -> `.env` and update values before starting services. Do NOT commit `.env` to git.

Core ports & endpoints

- Neo4j Browser / HTTP: http://localhost:7474
- Neo4j Bolt: bolt://localhost:7687
- Backend (FastAPI) default: http://localhost:8000
	- Health: GET /health
	- Recommendations: POST /recommendations
- Frontend dev (Vite) default: http://localhost:5173

Quickstart (local, minimal)

1) Copy and edit environment file

```powershell
copy .env.sample .env
# Edit .env and set NEO4J_PASSWORD and ADMIN_TOKEN
```

2) Start Neo4j using Docker Compose

```powershell
docker-compose -f infra/docker-compose.yml up -d
```

3) Confirm Neo4j is up

Open http://localhost:7474 in your browser and log in with the credentials you set in `.env`.

4) Load schema and sample data (one-off)

You can run the Cypher files in the `neo4j/cypher/` folder from Neo4j Browser or using the `neo4j` CLI or a client script.

From Neo4j Browser (copy/paste each file content):

- `neo4j/cypher/00_create_schema.cypher` — creates constraints and indexes
- `neo4j/cypher/01_sample_data.cypher` — inserts a small sample graph

Or use the `neo4j` cypher-shell (example):

```powershell
# Example using Docker container's cypher-shell
docker exec -it content_optimizer_neo4j cypher-shell -u neo4j -p yourpassword -f /var/lib/neo4j/cypher/00_create_schema.cypher
docker exec -it content_optimizer_neo4j cypher-shell -u neo4j -p yourpassword -f /var/lib/neo4j/cypher/01_sample_data.cypher
```

5) (Optional) Project GDS graph & run GraphSAGE

The `neo4j/cypher/02_gds_project_and_gsage.cypher` file contains example calls to `gds.graph.project` and
GraphSAGE write APIs. The exact call may vary by GDS version — check `CALL gds.version()` in Neo4j Browser.

6) Create & activate Python environment, install backend deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

7) Start the backend (FastAPI)

```powershell
# from repository root
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

8) Start the frontend (dev server)

```powershell
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

Expected minimal outputs (after quickstart)

- Neo4j: sample nodes and relationships are present (verify with simple Cypher queries in Browser)
- Backend: GET /health returns {"status":"ok"}
- Frontend: basic UI shows an input for `creatorId` and a button to request recommendations (initially placeholder responses)

Project status and next steps

This repository currently contains scaffolding and working placeholders for the main components. The next development tasks are:

1. Implement robust `neo4j_client` functions and test the Cypher execution path
2. Implement `embeddings.compute_text_embedding` integration with sentence-transformers
3. Implement `prepare_dataset.py` and `train_ranker.py` to produce and persist models
4. Wire the `/recommendations` endpoint to load the latest model and return scored candidates
5. Add tests and CI to validate end-to-end functionality

Troubleshooting & tips

- If GraphSAGE calls fail, ensure the Neo4j image includes the GDS plugin version you need. Check `CALL gds.version()`.
- If Bolt connection is refused, confirm the container is running and `7687` is mapped and not blocked.
- For faster text embedding runs, try smaller sentence-transformer models on CPU (e.g., `paraphrase-MiniLM-L3-v2`).

License & security

Do not commit secrets (`.env`). Use environment variable management for production.

Contact / Links

See `step_by_step_guide.md` for a complete implementation plan and file-by-file prompts.
