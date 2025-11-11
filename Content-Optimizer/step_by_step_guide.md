# Step by Step Guide

Nice — below is a **step-by-step, copy-pasteable prompt plan for GitHub Copilot**. Each step includes the exact prompt(s) you should paste into your editor (or use as a commit message / file header) so Copilot will produce the file/code you need. I assume you want a full-stack repo: Neo4j (GDS GraphSAGE), Python data pipeline & training (notebook + scripts), a FastAPI recommendation service, and a minimal React creator UI. I include file names, folder structure, required libs, and small examples of expected outputs so you can validate Copilot’s results.

Use this as a checklist: paste each prompt into a new file (or start a new editor buffer), accept/adjust Copilot suggestions, run tests. If Copilot drifts, paste the next prompt to re-generate the file with more constraints.

---

# Repo layout (target)

```
content-opt-recommender/
├─ infra/
│  ├─ docker-compose.yml
│  └─ neo4j/neo4j.conf
├─ neo4j/
│  ├─ import/ (sample CSVs)
│  └─ cypher/
│     ├─ 00_create_schema.cypher
│     ├─ 01_sample_data.cypher
│     └─ 02_gds_project_and_gsage.cypher
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ api/
│  │  │  ├─ recommendations.py
│  │  │  └─ admin.py
│  │  ├─ services/
│  │  │  ├─ neo4j_client.py
│  │  │  ├─ embeddings.py
│  │  │  └─ ranker.py
│  │  └─ models/
│  │     └─ schemas.py
│  ├─ notebooks/
│  │  └─ 00_pipeline_demo.ipynb
│  ├─ requirements.txt
│  └─ train/
│     ├─ prepare_dataset.py
│     ├─ train_ranker.py
│     └─ predict.py
├─ frontend/
│  ├─ package.json
│  └─ src/
│     ├─ App.jsx
│     └─ components/RecommendationsPanel.jsx
├─ tests/
│  └─ test_end_to_end.py
└─ README.md

```

---

# Required libraries & tools

- Neo4j with GDS plugin (>= GDS version that supports GraphSAGE)
- Python 3.10+
- Python libs: `neo4j`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `sentence-transformers`, `fastapi`, `uvicorn`, `requests`, `python-dotenv`
- Frontend: `react`, `vite` or `create-react-app`, `axios`

---

# HOW TO USE THESE Copilot prompts

1. Create the file named as specified.
2. Paste the **exact** prompt (quoted below) into the file as a comment header or as the file content (Copilot will read/edit and propose content).
3. Accept or iterate on Copilot suggestions. If it generates partial code, paste additional prompt(s) to refine.
4. Run the tests in `tests/` and manually run a small data ingest to validate end-to-end.

---

# STEP 0 — Repo README / Project description (paste as repo root README.md)

Prompt to paste into README.md:

```
# Content Optimization Recommender (Neo4j + GraphSAGE + Hybrid Ranker)

Generate a README describing a project that uses Neo4j (GDS GraphSAGE), sentence-transformer text embeddings, and a LightGBM hybrid ranker. Include quickstart: start Neo4j with Docker Compose, import sample data, run Python notebook to compute embeddings, train ranker, start FastAPI server, and start frontend. Provide required ports and credentials and expected outputs.

```

---

# STEP 1 — Infrastructure: docker-compose + Neo4j (infra/docker-compose.yml)

Prompt:

```
# Create a docker-compose.yml that runs Neo4j with the Graph Data Science plugin enabled, and a Python service placeholder.
# Use Neo4j official image with GDS. Expose Bolt 7687 and HTTP 7474.
# Create a single compose file with services: neo4j and backend (python). Set default password via env var and mount ./neo4j/import and ./neo4j/cypher.

```

What to expect: a docker-compose you can run `docker-compose up -d` to start Neo4j.

---

# STEP 2 — Neo4j schema & sample data (neo4j/cypher/00_create_schema.cypher)

Prompt:

```
/*
Create Neo4j schema: constraints and indexes for Creator, Content, Topic, Tag, AudienceSegment.
Add uniqueness constraints on creatorId, contentId, topicId, tagId, segmentId.
Create sample indexes for Content(createdAt) and Content(format).
This Cypher should be idempotent and safe to run multiple times.
*/

```

Then create `01_sample_data.cypher`:

```
/*
Insert a small sample graph: 2 creators, 6 content nodes with titles and lengths, a few topics/tags, and ENGAGED_WITH edges from AudienceSegment nodes with properties views, likes, watch_time.
This will be used to demo training.
*/

```

---

# STEP 3 — Graph projection + GraphSAGE commands (neo4j/cypher/02_gds_project_and_gsage.cypher)

Prompt:

```
/*
Write a Cypher file that:
1) Projects a GDS graph named contentGraph_demo including Content, AudienceSegment, Topic, Creator and relationships ENGAGED_WITH, HAS_TOPIC, CREATED.
2) Runs GraphSAGE unsupervised training with featureProperties ['lengthSec', 'format_onehot', 'platform_onehot'], embeddingDimension 128, epochs 10, learningRate 0.01, numSamples [25,10], aggregator 'mean'.
3) Writes the embedding back to content nodes as property gsage_embedding_v1.
Make commands robust with checks (drop graph if exists, etc.).
*/

```

Note: Copilot should generate `CALL gds.graph.project` + `CALL gds.beta.graphSage.write(...)` or equivalent depending on GDS version. Validate in your Neo4j.

---

# STEP 4 — Backend: Neo4j client & helpers (backend/app/services/neo4j_client.py)

Prompt:

```
# Create a Python module neo4j_client.py that:
# - uses neo4j.Driver from neo4j package
# - reads connection details from environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
# - exposes functions: get_driver(), run_cypher(query, parameters=None), fetch_content_embeddings(limit=None)
# - fetch_content_embeddings returns pandas DataFrame with contentId, title, lengthSec, gsage_embedding, text_embedding (if exists), and aggregated engagement stats (views, likes, watch_time)
# Include basic logging and connection pool handling.

```

---

# STEP 5 — Backend: Embeddings utilities (backend/app/services/embeddings.py)

Prompt:

```
# Create embeddings.py:
# - functions:
#   - compute_text_embedding(text_list, model_name='all-MiniLM-L6-v2') -> numpy array using sentence-transformers
#   - save_text_embeddings_to_neo4j(driver, content_embeddings: List[dict]) -> upsert content.text_embedding property
#   - combine_embeddings(graph_emb, text_emb, numeric_features) -> concatenated numpy vector
# - include type hints and docstrings
# - keep GPU optional (auto-detect)

```

---

# STEP 6 — Backend: Data prep & training scripts

Create `backend/train/prepare_dataset.py`:

Prompt:

```
# Create a Python script prepare_dataset.py that:
# - connects to Neo4j using neo4j_client
# - fetches content with gsage_embedding and text embedding (compute text embeddings if missing via embeddings.compute_text_embedding)
# - computes numeric features per-content (agg views, avg_watch_time, likes_normalized)
# - assembles a training dataset: feature vector and target engagement_score = 0.5*views_norm + 0.5*(watch_time_norm)
# - writes out a Parquet or numpy file for train and test splits
# - includes CLI args for output path and test_size

```

Create `backend/train/train_ranker.py`:

Prompt:

```
# Create a Python script train_ranker.py that:
# - loads dataset prepared by prepare_dataset.py
# - trains a LightGBM regressor (or sklearn GradientBoostingRegressor) to predict engagement_score
# - performs simple train/test split and logs R2 and NDCG@10 (approximate via sklearn metrics)
# - saves the model artifact (joblib or pickle) to disk with versioned filename including timestamp and metrics JSON
# - exposes minimal CLI to specify input file, model output dir, and hyperparameters

```

Create `backend/train/predict.py`:

Prompt:

```
# Create a predict.py that:
# - loads a saved model artifact
# - given a set of candidate content vectors (graph+text+meta), outputs scored list sorted descending
# - returns top K results and optionally nearest neighbors by cosine similarity from existing content embeddings
# - include CLI and function api for programmatic use

```

---

# STEP 7 — Notebook demo (backend/notebooks/00_pipeline_demo.ipynb)

Prompt:

```
# Create a Jupyter notebook that runs end-to-end on the small sample dataset:
# 1) Connect to Neo4j, run the sample cypher import scripts
# 2) Project the GDS graph and run GraphSAGE (or node2vec fallback)
# 3) Compute text embeddings for content titles using sentence-transformers
# 4) Assemble hybrid vectors
# 5) Train a small LightGBM model and show simple evaluation metrics and an example recommendation for a creator
# Add markdown cells describing each step and expected outputs (tables and plots).

```

---

# STEP 8 — FastAPI service (backend/app/main.py + api/recommendations.py)

Create `backend/app/main.py`:

Prompt:

```
# Create a FastAPI app in main.py:
# - include health endpoint GET /health
# - include router for /recommendations from api/recommendations.py
# - include router for /admin endpoints (trigger retrain, project graph) from api/admin.py
# - load config from .env
# - include CORS middleware allowing frontend origin
# - run with uvicorn in if __name__ == '__main__' block

```

Create `backend/app/api/recommendations.py`:

Prompt:

```
# Create a FastAPI router that implements POST /recommendations
# Input JSON schema: {creatorId: str, constraints?: {...}, topK?: int}
# Behavior:
# - Build candidate set: either take candidate IDs passed in request or fetch recent topics/content from Neo4j
# - For each candidate compute hybrid vector by fetching gsage_embedding or using GraphSAGE inductive infer, and text_embedding
# - Call the trained ranker (predict.py logic) to score candidates
# - Return topK items with fields {contentId, title, score, explanation: {nearest_examples:[{id,score}], top_features:[]}}
# Use Pydantic schemas for request/response models (in models/schemas.py)
# Add logging and error handling.

```

Create `backend/app/api/admin.py`:

Prompt:

```
# Create a small admin router that offers:
# POST /admin/retrain -> triggers prepare_dataset.py and train_ranker.py (calls subprocess or function)
# POST /admin/project_graph -> runs the cypher file to project graph and run GraphSAGE (executes Cypher via neo4j_client.run_cypher)
# GET /admin/status -> returns last run timestamps and model versions
# Secure these endpoints with a simple token-based header 'X-ADMIN-TOKEN' checked against env var ADMIN_TOKEN.

```

---

# STEP 9 — Models / Pydantic schemas (backend/app/models/schemas.py)

Prompt:

```
# Create Pydantic schemas for:
# - RecommendationRequest { creatorId: str, constraints: Optional[dict], topK: Optional[int]=10 }
# - RecommendationItem { contentId: str, title: str, score: float, explanation: dict }
# - RecommendationResponse { recommendations: List[RecommendationItem], modelVersion: str }
# - Admin triggers schemas as needed

```

---

# STEP 10 — Frontend minimal UI (frontend/src/components/RecommendationsPanel.jsx & App.jsx)

Create `frontend/package.json` with Vite/React minimal setup (Copilot can generate).

Prompt for `frontend/src/App.jsx`:

```
/*
Create a minimal React App that:
- Has a simple input for creatorId and a "Get Recommendations" button
- On click, POSTs to backend /recommendations and shows results in a RecommendationsPanel component
- Use axios for HTTP calls
*/

```

Prompt for `frontend/src/components/RecommendationsPanel.jsx`:

```
/*
Create a React component RecommendationsPanel that:
- Accepts an array of recommendations
- Renders a list with title, score, a "Why?" toggle that shows explanation.nearest_examples and top_features
- Include a 'Save idea' button (no backend behavior required initially)
*/

```

---

# STEP 11 — Tests (tests/test_end_to_end.py)

Prompt:

```
# Create an end-to-end test script using pytest that:
# - Starts with a running Neo4j (assume docker compose is up)
# - Calls the Neo4j cypher sample import via neo4j_client.run_cypher to ensure sample data present
# - Triggers GDS projection and GraphSAGE via admin endpoint or directly with neo4j_client
# - Runs the backend training pipeline (prepare_dataset -> train_ranker)
# - Starts the FastAPI test client and calls /recommendations for creatorId sample and asserts at least 1 recommendation returned
# Keep tests simple and able to run locally with small sample dataset.

```

---

# STEP 12 — Utilities: model registry & versioning

Prompt:

```
# Create a small module backend/app/services/ranker.py that wraps model saving/loading:
# - save_model(model, metrics, out_dir) -> writes model.joblib and metadata JSON with timestamp and metrics
# - load_latest_model(out_dir) -> returns model and metadata
# - list_models(out_dir) -> returns available versions
# Add unit tests for these functions.

```

---

# STEP 13 — Observability & job scheduling (optional)

Prompt:

```
# Create a simple scheduler script that:
# - Reads CRON config from .env (DAILY_RETRAIN=true/false)
# - If enabled, schedules a daily retrain by calling admin /admin/retrain
# - Logs outputs to a file in logs/
# Provide a systemd service file or a small README note on how to run in production.

```

---

# STEP 14 — Security and config (.env.sample)

Prompt:

```
# Create a .env.sample file listing:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=yourpassword
# ADMIN_TOKEN=changeme
# MODEL_DIR=./models
# FRONTEND_ORIGIN=http://localhost:5173
# OTHER config defaults

```

---

# STEP 15 — Final acceptance tests & checklist

Prompt (to paste in README or tests):

```
# Create a checklist script or README section describing manual validation steps:
# 1) docker-compose up (neo4j)
# 2) run cypher 00_create_schema.cypher and 01_sample_data.cypher
# 3) run notebook pipeline to compute text embeddings and GraphSAGE embeddings
# 4) run prepare_dataset.py and train_ranker.py
# 5) start backend: uvicorn app.main:app --reload
# 6) start frontend: npm run dev
# 7) In frontend, enter sample creatorId and verify at least 3 recommendations are returned with explanations
# Provide expected outputs and quick debugging tips.

```

---

# Helpful Copilot prompt tips (to get better results)

- When a file is long, paste the filename as a comment at the top and add: `# Use type hints, docstrings, and include a basic unit test.` Copilot responds better with explicit constraints.
- For Neo4j GDS GraphSAGE, if Copilot produces `beta` or `alpha` API calls incompatible with your GDS version, manually replace with the correct call (I included the expected patterns in earlier messages).
- If Copilot generates pseudocode for heavy compute (GraphSAGE training), insist on returning embeddings as lists of floats and persisting to node properties.

Example meta-prompt to paste at top of a file before using Copilot:

```
# File: backend/app/services/embeddings.py
# Goal: Production-ready Python functions to compute and persist text embeddings and assemble hybrid vectors.
# Requirements: use sentence-transformers, numpy, pandas, neo4j driver. Include type hints and docstrings. Add a short __main__ demo that computes embeddings for two sample texts.

```

---

# Example tiny snippet you can paste as a Copilot seed to generate the GraphSAGE write call

Prompt:

```
# Create a Python function `run_gds_graphsage(driver, graph_name='contentGraph_demo', write_property='gsage_embedding_v1')`
# that executes the following Cypher on Neo4j via driver:
# 1) drop existing graph if exists
# 2) gds.graph.project(...)
# 3) gds.beta.graphSage.write(...) or gds.graphSage.write depending on GDS version
# Return a dict summary with node_count and embedding_property.
# Include robust error handling.

```

---

# What I expect Copilot to generate (sanity checks)

- `neo4j_client.run_cypher()` correctly uses `session.run(query, parameters)` and returns records.
- GraphSAGE Cypher uses `gds.beta.graphSage.write` or `gds.graphSage.write` with `writeProperty`.
- `embeddings.compute_text_embedding()` uses `SentenceTransformer` and returns numpy arrays.
- Training script saves model + metadata JSON and prints validation metrics.
- FastAPI `/recommendations` returns Pydantic-validated JSON and uses model registry to load the latest model.

---

# Quick debugging tips

- If GraphSAGE call errors: confirm GDS plugin is installed and check exact method name with `CALL gds.version()` in Neo4j Browser.
- If text embeddings are slow on CPU, switch to smaller model `paraphrase-MiniLM-L3-v2` or run on a machine with GPU.
- If Neo4j connection refused: check docker-compose logs and that bolt port 7687 is bound.

---
