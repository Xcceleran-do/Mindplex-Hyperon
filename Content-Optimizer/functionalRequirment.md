# Full Functional Requirements — Content-Optimization Recommender using Neo4j + GraphSAGE

Below are **detailed, testable functional requirements** (FRs) for the content-optimization recommender you asked for. This spec assumes Neo4j + GDS GraphSAGE for inductive graph embeddings, content text embeddings (Sentence Transformers), and a downstream hybrid ranker (e.g., LightGBM or MLP). Each FR includes intent, inputs, outputs, behavior, and acceptance criteria.

---

# 1. Project summary (1-line)

A recommender that **suggests content ideas / optimizations to a creator** (topics, formats, titles, lengths, publish times) by combining GraphSAGE graph embeddings (collaborative signals) with content text/metadata embeddings (content signals) into a hybrid ranking model.

---

# 2. Actors

- **Creator** — receives recommendations and explanation.
- **Admin / Data Engineer** — loads data, triggers embedding/training jobs, views system health.
- **System** — Neo4j DB + GDS, ML training service, REST API, UI.

---

# 3. High-level capabilities (epics)

1. **Data Ingestion & Graph Creation** — store creators, content, topics, tags, audience segments, and engagement edges.
2. **Graph Embedding Generation (GraphSAGE)** — compute and persist inductive embeddings via Neo4j GDS.
3. **Content Embeddings** — compute text embeddings for title/description.
4. **Hybrid Feature Assembly** — join graph + text + metadata into per-content vectors.
5. **Train Ranker** — train supervised model to predict engagement target; support retraining.
6. **Generate Recommendations** — given a creator and optional constraints, produce a ranked list of content suggestions with explanations.
7. **UI / API** — expose recommendations and diagnostics.
8. **Monitoring, Logging & Retraining** — track performance and allow scheduled or on-demand retrain.
9. **Privacy & Security** — access control and audit logs.

---

# 4. Data model (Neo4j schema) — concrete fields

## Nodes (required)

- `:Creator {creatorId: STRING, name: STRING, createdAt: DATETIME}`
- `:Content {contentId: STRING, title: STRING, body: STRING, createdAt: DATETIME, lengthSec: INT, format: STRING, language: STRING, platform: STRING, status: STRING, text_embedding: LIST<FLOAT>, gsage_embedding: LIST<FLOAT>}`
- `:Topic {topicId: STRING, name: STRING}`
- `:Tag {tagId: STRING, name: STRING}`
- `:AudienceSegment {segmentId: STRING, name: STRING, attributes: MAP}`

## Relationships (required)

- `(:Creator)-[:CREATED]->(:Content)`
- `(:Content)-[:HAS_TOPIC]->(:Topic)`
- `(:Content)-[:TAGGED]->(:Tag)`
- `(:AudienceSegment)-[:ENGAGED_WITH {views: INT, likes: INT, watch_time: INT, conversions: INT, timestamp: DATETIME}]->(:Content)`

## Indexes & constraints

- Unique constraints on `creatorId`, `contentId`, `topicId`, `tagId`, `segmentId`.
- Index on `:Content(createdAt)`, `:Content(format)`, `:AudienceSegment(name)`.

**Acceptance:** DB has constraints & indexes; sample import creates nodes and edges and passes uniqueness tests.

---

# 5. Data ingestion FRs

### FR 5.1 — Batch ingest content metadata

- **Description:** System accepts CSV/JSON with content metadata and upserts `:Content` nodes and `:Creator` nodes.
- **Input:** CSV/JSON with contentId, creatorId, title, body, lengthSec, format, createdAt, platform, language, tags, topics.
- **Behavior:** Upsert nodes; create relationships to topics/tags/creator; validate required fields.
- **Errors:** Missing contentId or creatorId → reject row with error message.
- **Acceptance:** Import of 10k rows completes and nodes count equals expected; log contains row-level success/failure.

### FR 5.2 — Batch ingest engagement edges

- **Description:** Import audience engagement (views, likes, watch_time).
- **Acceptance:** Engaged edges are stored with aggregation, and duplicates update the numeric fields (either replaced or accumulated by configured policy).

---

# 6. Graph projection & GraphSAGE embedding FRs

### FR 6.1 — Project subgraph for GDS

- **Description:** System projects a named GDS graph (e.g., `contentGraph_v{version}`) containing selected node labels and relationship types.
- **Inputs/Config:** Node labels to include, rel types, edge properties to keep.
- **Behavior:** Call `gds.graph.project()` with chosen labels and relationships. Use ephemeral graph names for test runs.
- **Acceptance:** Graph projection returns expected node/edge counts within 0.5% of the Neo4j counts for selected labels.

### FR 6.2 — Train GraphSAGE embeddings (unsupervised)

- **Description:** Run GraphSAGE with configurable hyperparameters and persist embeddings on `:Content` nodes (property `gsage_embedding`).
- **Config parameters:**
    - `featureProperties` — list of node properties to use (e.g., `['lengthSec','format_onehot','platform_onehot']`)
    - `embeddingDimension` — default 128
    - `epochs` — default 10
    - `learningRate` — default 0.01
    - `batchSize` — default chosen by GDS
    - `numSamples` (neighbors sampling per layer) — default `[25,10]` or similar
    - `aggregator` — e.g., `mean` or `pool`
    - `seed` — for reproducibility
- **Behavior:** Train in unsupervised mode (or supervised later) and write embeddings back with `gds.beta.graphSage.write(...)` or stream and write properties.
- **Outputs:** `gsage_embedding` stored on each content node.
- **Acceptance:** For a test dataset, embeddings are produced for >99% of projected content nodes and `gsage_embedding` is a list of length `embeddingDimension`.

### FR 6.3 — Recompute embeddings on schedule or on-demand

- **Description:** Support scheduled retrain (cron) and on-demand retrain via API with versioning (append version suffix to embedding property or graph name).
- **Acceptance:** Scheduled job runs and produces new embedding property `gsage_embedding_v20251111_01`.

---

# 7. Content text embeddings FRs

### FR 7.1 — Compute sentence-transformer embeddings for title/body

- **Description:** Compute dense text embeddings for each content `title` and optionally `body` using a SentenceTransformer model (e.g., `all-MiniLM-L6-v2`) and store as `text_embedding`.
- **Behavior:** If `body` present and `useBody` flag true, create a combined embedding (concatenate or mean of title/body embeddings).
- **Acceptance:** Embeddings exist for >99% of content nodes and size consistent with chosen model (e.g., 384 dims).

---

# 8. Hybrid feature assembly FRs

### FR 8.1 — Build feature vector for training/prediction

- **Description:** For each content node, assemble vector = `[gsage_embedding, text_embedding, numeric_features, one_hot_metadata]`.
- **Numeric features:** lengthSec, views (aggregated), watch_time_avg, likes, conversions rates (normalized).
- **One-hot metadata:** format, platform, language (must be reproducible across train/predict).
- **Output:** Single vector stored temporarily when exporting to training service.
- **Acceptance:** Assembly script produces identical-length vectors for all rows; vector schema is logged.

---

# 9. Downstream training FRs

### FR 9.1 — Export training dataset

- **Description:** Export assembled feature vectors + target label to a training environment in CSV/Parquet or via direct API.
- **Target definition:** configurable, default is `engagement_score = alpha*views_normalized + beta*(watch_time / lengthSec) + gamma*likes_normalized`.
- **Acceptance:** Export contains X rows, no null vectors, and matches the number of labeled historical contents.

### FR 9.2 — Train ranker/regressor

- **Description:** Train LightGBM (or MLP) to predict `engagement_score` or rank pairwise; support hyperparameter tuning and model versioning.
- **Behavior:** Save model artifacts and schema, log training metrics (R², NDCG@k on holdout).
- **Acceptance:** Model achieves minimum baseline metrics on validation set (configurable).

### FR 9.3 — Model versioning & rollback

- **Description:** Store trained model with version, timestamp, metrics; support rollback to previous version via API.
- **Acceptance:** API can switch active model and return previous model id.

---

# 10. Recommendation generation FRs

### FR 10.1 — Recommend content ideas for a creator (core)

- **Endpoint:** `POST /recommendations`
- **Input:** `{ creatorId, constraints?: {format, topics[], maxLengthSec}, topK?: int (default 10), candidate_pool?: [contentIds or topics], modelVersion?: string }`
- **Behavior:**
    1. Build candidate set (all content types or topic-constrained).
    2. For each candidate compute hybrid vector (if new content idea, compute text_embedding & fallback graph embedding via GraphSAGE inductively).
    3. Score candidates with active ranker.
    4. Return topK with explanation (feature contributions, nearest neighbors in embedding space).
- **Output:** List `[{contentIdea, score, explanation: {nearest_existing: [...], feature_importances}}]`.
- **Acceptance:** API responds in <2s for candidate pool up to 1000 (target), returns topK and explanation.

### FR 10.2 — Generate novel content idea suggestions

- **Description:** Suggest new topic/format combinations not in candidate pool by sampling high-scoring directions: take centroids of high-performing clusters and produce human-readable suggestions (e.g., "Short video on X, 3–5 minutes, publish Mon 9AM").
- **Acceptance:** Top 5 novel suggestions contain at least 3 that are not exact duplicates of existing content titles.

### FR 10.3 — Provide explanation & neighbor examples

- **Description:** For each recommendation show:
    - Top 3 nearest existing content nodes (embedding similarity).
    - Key features that influenced the score (SHAP or feature permutation).
- **Acceptance:** Explanation returned for every recommendation; nearest neighbors are real content nodes with similarity score.

---

# 11. UI FRs (creator dashboard)

### FR 11.1 — Recommendations panel

- **UI elements:** Top suggestions list, score badge, explanation toggle, "Why this?" modal, nearest examples thumbnails, action buttons (save idea, schedule, share).
- **Acceptance:** Clicking a suggestion opens modal with explanation + nearest examples + suggested tags.

### FR 11.2 — Feedback loop (creator signals)

- **Description:** Creator can mark suggestions: `useful`, `not useful`, or `already did`. This feedback is stored as events and used as supervised signal for future training.
- **Acceptance:** UI captures feedback events and they appear in the ingestion logs.

---

# 12. Monitoring, evaluation & metrics FRs

### FR 12.1 — Offline metrics dashboard

- **Displays:** R², MAE, NDCG@5/10, Precision@K, training/validation loss across versions.
- **Acceptance:** Dashboard updates after each training job.

### FR 12.2 — Online A/B experiments tracking

- **Description:** Ability to tag a set of creators as experiment/control, track lift in watch_time, engagement rate, and creator adoption.
- **Acceptance:** Experiment reports show statistically significant results and 95% CIs.

---

# 13. Performance & nonfunctional requirements

- **Latency:** Recommendation API median latency < 500ms for candidate pool ≤ 200; <2s for ≤1000 (as earlier).
- **Embedding generation:** GraphSAGE training time and memory documented; for up to 1M nodes must support distributed GDS or chunked pipelines (documented scaling strategy).
- **Availability:** API 99.9% monthly uptime.
- **Throughput:** Support 200 req/min sustained for recommendations.
- **Storage:** Persist embeddings and model artifacts with metadata; ensure backups.

**Acceptance:** Load tests simulate expected QPS and meet latency & availability targets.

---

---

---

# 14. Testing & acceptance criteria (summary)

- **Unit tests:** for ingestion parsers, feature assembly, model export.
- **Integration tests:** Graph projection → GraphSAGE produces embeddings → text embeddings present → vector assembly → train small model end-to-end on sample dataset.
- **E2E test:** From content import to UI recommendation for a creator returns topK and explanation.
- **Performance tests:** Load and latency tests as specified.
- **Usability tests:** Creator understands the explanation and acts on recommendations in a usability session.

---

# 15. Example Neo4j GraphSAGE commands (implementation notes)

These are the canonical Cypher-style commands for implementation (already referenced in the pipeline):

1. **Project the graph**

```
CALL gds.graph.project(
  'contentGraph_v1',
  ['Content','AudienceSegment','Topic','Creator'],
  {
    ENGAGED_WITH: {properties:['views','likes','watch_time']},
    HAS_TOPIC: {},
    CREATED: {}
  }
);

```

1. **Train GraphSAGE (beta API) — unsupervised, streaming embeddings**

```
CALL gds.beta.graphSage.train.stream('contentGraph_v1', {
  featureProperties: ['lengthSec','platform_onehot','format_onehot'],
  embeddingDimension: 128,
  epochs: 10,
  learningRate: 0.01,
  batchSize: 512,
  numSamples: [25, 10],
  aggregator: 'mean',
  seed: 42
}) YIELD nodeId, embedding
RETURN gds.util.asNode(nodeId).contentId AS contentId, embedding LIMIT 10;

```

1. **Persist embeddings**

```
CALL gds.beta.graphSage.write('contentGraph_v1', {
  writeProperty: 'gsage_embedding',
  featureProperties: ['lengthSec','platform_onehot','format_onehot'],
  embeddingDimension: 128,
  epochs: 10
});

```

> Implementation note: use beta / available GDS API version consistent with your Neo4j/GDS version. Include versioned property names like gsage_embedding_v20251111_01 for traceability.
> 

---

# 16. Feature engineering rules (concrete)

- **Normalization:** numeric engagement metrics normalized per-day and per-platform.
- **One-hot encoding:** `format`, `platform`, `language`.
- **Categorical hashing:** for high-cardinality fields like `tag` (if included).
- **Temporal features:** day-of-week, hour-of-day of creation, recency of topic trend (e.g., rolling 7-day growth).
- **Audience aggregation:** compute per-content aggregated metrics across segments and store as node properties.

**Acceptance:** feature pipeline has unit tests and produces stable transforms between train/inference.

---

# 19. Edge cases & business constraints

- Cold-start: New content without ENGAGED_WITH edges uses text_embedding + metadata; GraphSAGE inductive capability uses feature-only fallback.
    
    ---
    
- Sparse engagement: If content has < N engagement events, downweight or mark as low-confidence; surface to creator with disclaimer.
- Diversity control: Add tunable lambda for diversification to avoid repetitive suggestions.