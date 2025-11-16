
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

## Final Acceptance Checklist (Step 12)

Use this checklist to validate the full end-to-end functionality before tagging a release.

### 1. Infrastructure & Data
1. `docker-compose -f infra/docker-compose.yml up -d` starts Neo4j (verify container healthy).
2. Run `neo4j/cypher/00_create_schema.cypher` then `01_sample_data.cypher` (constraints succeed; sample nodes appear).
3. (Optional) Run `02_gds_project_and_gsage.cypher` to generate GraphSAGE embeddings; confirm `Content.embedding` (or `gsage_embedding`) is populated via:
	```cypher
	MATCH (c:Content) RETURN c.contentId, size(coalesce(c.embedding, c.gsage_embedding)) AS embDim LIMIT 5;
	```

### 2. Embeddings & Dataset
4. If text embeddings missing, run notebook `backend/notebooks/00_pipeline_demo.ipynb` or a script to populate `c.text_embedding`.
5. Run dataset prep:
	```powershell
	.\.venv\Scripts\Activate.ps1
	python backend/train/prepare_dataset.py --output data/dataset.parquet --test_size 0.2
	```
	Expected: `data/dataset.parquet` written; columns include `contentId,title,features,target`.

### 3. Model Training
6. Train ranker:
	```powershell
	python backend/train/train_ranker.py --input data/dataset.parquet --out_dir models
	```
	Expected: `models/model_<timestamp>.joblib` + matching `.json` meta (contains version & metrics).

### 4. API & Frontend
7. Start backend:
	```powershell
	uvicorn backend.app.main:app --reload --port 8000
	```
	Health check: `curl http://localhost:8000/health` -> JSON with keys `status`, `neo4j_version`, `has_embeddings`.
8. Start frontend:
	```powershell
	cd frontend
	npm install
	npm run dev
	```
	Open http://localhost:5173 and enter a sample `creatorId` (or leave blank) then request recommendations.

### 5. Recommendations Verification
9. Call recommendations endpoint directly:
	```powershell
	curl "http://localhost:8000/recommendations?topK=5"
	```
	Expected response structure:
	```json
	{
	  "recommendations": [
		 {
			"contentId": "...",
			"title": "...",
			"score": 0.1234,
			"explanation": {
			  "score": 0.1234,
			  "model": "model_<timestamp>",
			  "nearest_examples": [{"contentId":"...","similarity":0.98,"score":0.12}, ...],
			  "top_features": [12, 3, 57, 4, 1]
			}
		 }
	  ],
	  "modelVersion": "model_<timestamp>"
	}
	```
	Acceptance: At least 3 recommendations returned (if sample dataset loaded) and each contains `explanation.nearest_examples` (<=3 items) and `explanation.top_features` (<=5 indices).

### 6. Tests
10. Run automated tests:
	 ```powershell
	 pytest -q
	 ```
	 Expected: All tests pass (including acceptance test if added) with no errors; minor warnings acceptable.

### 7. Debugging Tips
- Fewer than 3 recommendations? Verify sample data import and embeddings creation; ensure model artifacts exist (`models/`).
- Empty `nearest_examples`? Happens when fewer than 2 content items or embeddings mismatch sizes.
- All scores zero? Indicates model not loaded; check models directory and retrain.
- Feature importances empty? LightGBM not used or model lacks `feature_importances_`; retrain with LightGBM.
- Neo4j connection errors? Confirm env vars `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` match running container.

### 8. Ready to Ship
Ship when: checklist items 1–10 succeed, recommendations stable across refreshes, and no critical errors in logs.

---
*This acceptance checklist is generated as part of Step 12.*
