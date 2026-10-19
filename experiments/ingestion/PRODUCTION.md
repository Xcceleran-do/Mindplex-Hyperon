# Production ingestion

Run `metadata-extractor2PLN` as an independently deployed service and configure
`METADATA_EXTRACTOR_BASE_URL` and `METADATA_EXTRACTOR_API_KEY` in the mining API.
The mining API does not need `GEMINI_API_KEY`; that secret belongs only to the
extractor deployment.

`INGESTION_ENABLED=false` rejects new ingestion while preserving the current
dataset. Successful ingestion writes freshness metadata beside `data.metta`.
The same Mindplex username and requested limit reuse that dataset for three days
by default. Configure the window with `INGESTION_CACHE_TTL_DAYS`; `0` disables
caching and `force=true` bypasses it.

The generated dataset and its cache metadata should be stored on a persistent
volume. A remote extraction failure never replaces the current dataset.

Only Mindplex is a supported source here. Arbitrary request URLs, local LLM
planning, and local MeTTa conversion have been removed. The remote response is
treated as untrusted: every atom must match the narrow fact grammar, truth
values must be within `[0,1]`, and every configured required property must be
present for every record before the dataset is written atomically.

For local Docker Compose, the mining container uses host networking and can
reach an extractor published on `127.0.0.1:8080`. In a distributed deployment,
set `METADATA_EXTRACTOR_BASE_URL` to the extractor's internal TLS/service URL.
