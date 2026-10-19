# Mindplex Hyperon

Mindplex Hyperon turns Mindplex articles into a symbolic knowledge graph for visual exploration, pattern mining, and explainable reasoning.

This branch uses these components:

- **Mindplex Hyperon** fetches articles, serves the UI, and coordinates the workflow.
- [**metadata-extractor2PLN**](https://github.com/yotors/metadata-extracter2PLN) converts article metadata and text into PeTTa-formatted facts.
- [**NL2PLN**](https://github.com/yotors/NL2PLN) translates natural-language statements and questions into validated PeTTaChainer input.
- [**PeTTaChainer**](https://github.com/yotors/PeTTaChainer) persists knowledge bases and provides backward and forward reasoning.
- [**PeTTa**](https://github.com/yotors/PeTTa) runs the pattern miner locally as a Git submodule.

## Requirements

- Docker with Docker Compose
- Mindplex credentials or a valid Mindplex API token
- A Gemini key for NL2PLN and an ASI:One or Gemini key for metadata extraction

## Setup

Clone the repository and initialize PeTTa, which is required by the local pattern miner:

```bash
git submodule update --init --recursive PeTTa
cp .env.example .env
```

Set the generated service secrets, provider credentials, and Mindplex credentials in `.env`:

```dotenv
POSTGRES_PASSWORD=replace-with-a-random-password
METADATA_EXTRACTOR_API_KEY=mindplex:replace-with-at-least-32-random-characters
NL2PLN_API_KEY=replace-with-at-least-32-random-characters
PETTACHAINER_API_KEY=replace-with-at-least-32-random-characters

GEMINI_API_KEY=
ASI_ONE_API_KEY=
ASI_API_KEY=

# Use a Mindplex service account or existing tokens
MINDPLEX_SERVICE_EMAIL=
MINDPLEX_SERVICE_PASSWORD=
MINDPLEX_API_TOKEN=
MINDPLEX_API_REFRESH_TOKEN=
```

Start the application:

```bash
docker compose up --build
```

Compose builds the dependency repositories from GitHub and starts the complete stack. Open [http://localhost:3001](http://localhost:3001). The health endpoint is [http://localhost:3001/api/health](http://localhost:3001/api/health).

## Basic workflow

1. Enter a Mindplex username and ingest their articles.
2. Explore the extracted attributes in the knowledge graph.
3. Select multiple attributes to show articles matching all selections.
4. Run pattern mining to discover supported rules.
5. Query the backward chainer for explanations grounded in the dataset and mined rules.

Ingestion results are cached for three days by default. Use **Force refresh** in the UI when a fresh extraction is required.

## Useful commands

```bash
# Follow application logs
docker compose logs -f

# Rebuild only the frontend
docker compose build frontend
docker compose up -d --no-deps frontend

# Stop the application
docker compose down
```

Runtime datasets and authentication tokens are stored in Docker-mounted data locations and are not committed to the repository.
