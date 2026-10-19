# Mindplex ingestion adapter

Mindplex-Hyperon owns only the source side of ingestion:

1. authenticate with Mindplex and refresh/cache its access token;
2. fetch and normalize articles for a requested username;
3. reuse a dataset produced within the configured freshness window;
4. send normalized JSON records to `metadata-extractor2PLN`;
5. validate the returned fact contract and atomically replace `data.metta`.

Metadata planning, deterministic extraction, model calls, evidence validation,
STV generation, and PeTTaChainer fact compilation live exclusively in the
headless metadata extractor service.

## Required configuration

```env
METADATA_EXTRACTOR_BASE_URL=http://127.0.0.1:8080
METADATA_EXTRACTOR_API_KEY=mindplex:replace-with-the-configured-server-secret
METADATA_EXTRACTOR_NAMESPACE=A
METADATA_EXTRACTOR_TIMEOUT_SECONDS=150
METADATA_EXTRACTOR_CHUNK_SIZE=10
METADATA_EXTRACTOR_USE_MODEL=true
METADATA_EXTRACTOR_REQUIRED_PROPERTIES=engagement,audience-expertise
```

The API key is the complete `owner-id:secret` bearer token accepted by
`metadata-extractor2PLN`. It is not a Gemini key.

The adapter discovers one plan from at most 20 samples, then extracts records in
bounded chunks. Any remote error, unsafe atom, missing required property, or
partial result aborts the operation and preserves the previous dataset.

The extractor batches semantic classification once per chunk. With the default
chunk size, ingesting 50 articles therefore uses about six model requests: one
for plan discovery and five for extraction, instead of one request per article.
Increase `METADATA_EXTRACTOR_CHUNK_SIZE` cautiously when article sizes and the
provider token limit allow it; the server accepts at most 100 records per extract
request.

Mindplex engagement counters are normalized to `views`, `likes`, `comments`,
`shares`, and `reactions`. The remote service aggregates those counters using
its canonical weighted-interaction calculation.
