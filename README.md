
# Mindplex Hyperon

Mindplex Hyperon is a robust, transparent, and explainable recommendation engine that leverages [AtomSpace](https://github.com/opencog/atomspace/), a metagraph-based knowledge graph developed by the OpenCog project. AtomSpace provides a flexible, semantic framework for representing and querying relationships between concepts, supporting advanced reasoning and AI-driven applications.

The core aim of Mindplex Hyperon is to build recommendations that are not only accurate but also interpretable and trustworthy. By utilizing AtomSpace, the engine can mimic user behavior from historical data, constructing an agent that predicts how a user will interact with future content or features. This agent-based modeling enables the system to provide recommendations with clear, logical explanations, helping users understand the reasoning behind each suggestion.

Key features include:
- Transparent and explainable recommendations
- User-centric agent modeling based on interaction history
- Integration with AtomSpace for semantic knowledge representation
- Support for logic-based graph processing and advanced analytics

Mindplex Hyperon sets a new standard for recommendation systems by focusing on transparency, explainability, and robust knowledge graph integration. The project is designed for extensibility, allowing researchers and developers to build upon its foundation for a wide range of AI and data-driven applications.
## This Branch

This branch is dedicated to the implementation and experimentation of the framework with [asi1](https://asi1.ai/).

**Guidelines for contributors:**
- Select a symbolic, graph-based recommendation method or algorithm for implementation.
- Create a dedicated folder named after the symbolic the algorithm you are implementing, in the root directory for your implementation.
- Document the selected approach, your implementation details, and analysis  as .md file in your folder. Include a summary of the method, core algorithms or logic, and how it is adapted or integrated into Mindplex Hyperon.
- Ensure your code is modular and adheres to the project's contribution standards.
- Provide clear explanations and comments to support understanding and future enhancements.

This branch is a collaborative environment for advancing symbolic AI research and practical applications, supporting transparent and explainable recommendation systems.
## Testing

Each feature has associated test cases located in the `features/tests` directory. The test files are named with a `-test` suffix to facilitate CI/CD recognition. 

- **Feature One Tests**: Located in `features/tests/FeatureOne-test.metta`
- **Feature Two Tests**: Located in `features/tests/FeatureTwo-test.metta`

## CI/CD

This project utilizes GitHub Actions for continuous integration and deployment. The CI/CD workflow is configured to run all test files upon every pull request to ensure that new changes do not break existing functionality.

## Contributing

We welcome contributions to Mindplex Hyperon! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute, including the requirement for tests in every pull request and naming conventions.

## Setup Instructions

To set up the project locally, clone the repository and install any necessary dependencies. Follow the instructions in the `CONTRIBUTING.md` file for detailed setup and contribution guidelines.

For the mining API and PeTTa integration, run these commands from the repository root. Install the Python dependencies and SWI-Prolog >= 9.3.x. The default Ubuntu package can be too old, so use the official SWI-Prolog devel PPA on WSL/Ubuntu:

```bash
sudo apt update
sudo apt install software-properties-common build-essential python3-dev
sudo add-apt-repository ppa:swi-prolog/devel
sudo apt update
sudo apt install swi-prolog
swipl --version
python -m pip uninstall -y janus-swi
python -m pip cache purge
python -m pip install --no-cache-dir -r experiments/requirements.txt
python -c "from petta import PeTTa; print('petta ok')"
```

If you use `uv`, the root `pyproject.toml` declares `petta` as an editable local dependency from `PeTTa`. Running `uv sync` from the repository root will generate/update `uv.lock`.

`janus-swi` is listed in `experiments/requirements.txt`, but it is only the Python bridge. The `libswipl.so.9` library in Janus import errors comes from the OS-level SWI-Prolog package, and PeTTa requires SWI-Prolog 9.3.0 or newer.

Reinstall `janus-swi` after installing or upgrading SWI-Prolog. Janus builds/links against the `swipl` found on your PATH, so a cached install can keep pointing at the wrong `libswipl.so.*`.

The mining API uses the Janus-backed PeTTa engine directly. It does not fall back to Hyperon or the PeTTa shell script when the engine is unavailable.

Backward chaining is provided by an independently deployed PeTTaChainer API;
Mindplex does not initialize an in-process chainer. Configure the client with:

```dotenv
PETTACHAINER_BASE_URL=http://127.0.0.1:8000
PETTACHAINER_API_KEY=THE_CLIENT_SECRET_FROM_THE_SERVER_ALLOWLIST
PETTACHAINER_KB_PREFIX=mindplex
```

Ingestion only writes the dataset and invalidates local chainer metadata. The
first subsequent chainer query creates or reuses a content-addressed remote
knowledge base and uploads the dataset in atomic batches. PeTTa and SWI-Prolog
remain Mindplex dependencies for local pattern mining and the legacy simulator,
not for normal backward chaining.

## Usage Examples

Refer to the individual feature files for usage examples and implementation details. Each feature is designed to be modular and can be integrated into larger systems as needed.
