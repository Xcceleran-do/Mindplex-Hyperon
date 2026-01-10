# Mindplex-Hyperon AI Agent Instructions

## Project Overview
Mindplex-Hyperon is a transparent, explainable recommendation engine using **AtomSpace** (metagraph-based knowledge graph from OpenCog) and **MeTTa** (a meta-language for symbolic AI). The system mines frequent patterns, performs logical reasoning via backward chaining, and visualizes knowledge graphs with AI chat assistance.

## Architecture Components

### 1. **MeTTa Language & PeTTa Runtime**
- **MeTTa** (.metta files): Domain-specific language for symbolic reasoning, pattern matching, and knowledge representation
- **PeTTa** (`/PeTTa/`): Prolog-based MeTTa interpreter (`sh run.sh <file>.metta` to execute)
- **Hyperon** Python bindings: `from hyperon import MeTTa` for Python↔MeTTa interop

### 2. **Experiments Pipeline** (`/experiments/`)
- **Pattern Mining**: `pattern-miner/` wraps `frequent-pattern-miner/` to discover frequent conjunctions (e.g., `(supportOf (, (topic $x "AI") (length $x "low")) 3)`)
- **Backward Chainer**: `chainer/` performs logical inference from rules+facts using proof trees
- **Ingestion**: `ingestion/` fetches Mindplex articles, enriches with Gemini AI analysis, converts to MeTTa atoms
- **Visualizer**: `atomspace_visualizer/` React/SolidJS frontend (port 3000) with D3.js graph visualization + AI chat
- **API Server**: `mining_api.py` (port 5000) exposes pattern mining, chat, and chainer endpoints via Flask

### 3. **Key Data Flow**
```
Articles (Mindplex) → ingestion/pipeline.py → MeTTa atoms → &tempo space
                                                   ↓
                              pattern-miner (MeTTa) → frequent patterns
                                                   ↓
                              mining_api.py → Flask API → visualizer (chat + graph)
                                                   ↓
                              Gemini AI ← chat queries → chainer (backward reasoning)
```

## Critical Developer Workflows

### Running Tests
```bash
# From project root (runs all *-test.metta files)
python3 run-tests.py

# Single MeTTa test file (from PeTTa directory)
cd PeTTa && time sh run.sh ../experiments/pattern-miner/tests/test-pattern-miner.metta

# Or using metta command directly
time metta experiments/pattern-miner/tests/test-pattern-miner.metta
```

### Starting Services
```bash
# Quick start (backend + frontend)
cd experiments && ./start_all.sh
# Backend: http://localhost:5000 | Frontend: http://localhost:3000

# Stop all
./stop_all.sh
```

### Module System (MeTTa)
```metta
;; Register root module path (use relative paths from file location)
! (register-module! ../../../experiments)

;; Import modules using colon-separated paths
! (import! &self experiments:pattern-miner:pattern-miner)
! (import! &self experiments:utils:common-utils)
! (import! &tempo experiments:atomspace_visualizer:public:data)  ; &tempo holds article atoms
```

### Python↔MeTTa Integration
```python
from hyperon import MeTTa

metta = MeTTa()
metta.run("! (register-module! experiments)")
metta.run("! (import! &self experiments:pattern-miner:pattern-miner)")

# Execute queries
result = metta.run("!(pattern-miner purifiedDbSpace 3 2)")

# Parse strings to MeTTa atoms
atom = metta.parse_single("(topic 0 \"AI\")")
```

## Project-Specific Conventions

### File/Function Naming (from CONVENTION.md)
- **Files/folders**: camelCase (`userProfile.metta`, `ingestion/`)
- **Test files**: `-test` suffix (`pattern-miner-test.metta`)
- **Functions/variables**: camelCase (`getUserHistory`, `$userHistory`)
- **Comments**: Always `;;` (double semicolon), never `;`

### MeTTa Formatting Rules
```metta
;; Short parameters: single line
(foo $x $y)

;; Long parameters: function name sticks to opening bracket
(foo 
    (and (== $x $y) (< $x 100))
    (X is valid)
    (X is not valid)
)

;; Multi-line if: condition on first line, branches below
(if (== $x 1)
    True
    (complex-expression ...)
)

;; Function definitions: no empty lines between type signature and body
(: getUserHistory (-> Atom Atom))
(= (getUserHistory $user) 
   (match &db (history $user $item) $item)
)
```

### Testing Requirements
- **All new features must include tests** in `<feature>/tests/` directory
- Tests must use `-test` suffix (e.g., `featureName-test.metta`)
- CI/CD runs on every PR using GitHub Actions
- Exit code 0 = success, non-zero = failure

### Commit Messages
```
feat: implement admin middleware

- added userAuth.metta to handle JWT validation
- revised mining_api.py to include authentication decorator
- removed deprecated session handling code from chat endpoint (fixes #42)
```
(See [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/))

## Integration Points

### API Endpoints (mining_api.py)
- `GET /api/health` - Health check
- `POST /api/mine` - Start pattern mining `{ "conjunction_count": 2 }`
- `POST /api/chat` - Chat with AI `{ "message": "...", "history": [...] }`
- `POST /api/chat/analyze` - Analyze patterns `{ "result": [{pattern, support}] }`
- `POST /api/chainer` - Backward chaining `{ "what_to_check": "(likes Abe soda)" }`

### Environment Configuration
- **Config**: `experiments/config.py` loads `.env.local` for API URLs
- **Gemini API**: Uses `GEMINI_API_KEY4` environment variable
- **Dev Container**: Ubuntu 24.04 with `docker`, `git`, `gh`, `kubectl` pre-installed

### External Dependencies
- **AtomSpace/OpenCog**: Knowledge graph framework (conceptual - using MeTTa spaces)
- **Gemini AI**: Powers chat analysis, article enrichment, and function calling
- **D3.js**: Graph visualization in frontend
- **Flask + CORS**: Backend API server
- **React/SolidJS**: Frontend UI framework

## Common Patterns & Gotchas

### MeTTa Spaces (Stateful Knowledge Stores)
```metta
;; Create new space
!(bind! &mySpace (new-space))

;; Add atoms to space
!(add-atom &mySpace (topic 0 "AI"))

;; Query space
!(match &mySpace (topic $x $y) ($x $y))

;; Get all atoms
!(get-atoms &mySpace)
```

### Pattern Mining Parameters
```metta
;; (pattern-miner $kb $db $minsup $depth)
;; $minsup: minimum support count (e.g., 3 = pattern must appear ≥3 times)
;; $depth: conjunction size (2=pairs, 3=triples)
!(pattern-miner purifiedDbSpace 3 2)
```

### Python-MeTTa Result Parsing
```python
# MeTTa returns nested atom structures
answer = metta.run("!(some-query)")
result_atom = answer[0][0]  # First result, first atom
children = result_atom.get_children()  # Extract sub-atoms
pattern_str = str(children[1])  # Convert to string
```

### Debugging Tips
- **MeTTa errors**: Often exit code 134 (assertion failure) - check module paths and imports
- **Chat not opening**: Verify `z-index: 9999` and `display: flex` in CSS
- **CORS issues**: Ensure Flask CORS enabled for localhost:3000
- **Module not found**: Check `register-module!` uses correct relative path
- **Performance degradation**: AtomSpace operates on RAM; slowness and crashes occur with larger datasets. Pattern mining scales poorly beyond test data sizes. Consider data filtering/sampling strategies when working with production-scale article collections.

## Key Files to Reference
- **CONTRIBUTING.md**: PR guidelines, branch naming, test requirements
- **CONVENTION.md**: MeTTa formatting, naming conventions, commit standards
- **experiments/README.md**: Service architecture, endpoints, troubleshooting
- **pattern-miner/README.md**: Pattern mining API and usage examples
- **atomspace_visualizer/CHAT_FEATURE_README.md**: Chat+mining implementation details

## When Making Changes
1. **New features**: Create folder in `experiments/`, add `-test.metta` in `tests/` subfolder
2. **MeTTa code**: Follow formatting rules (double semicolon comments, camelCase, proper indentation)
3. **Python↔MeTTa**: Use `metta.parse_single()` for string→atom, `.get_children()` for extraction
4. **PRs**: Must include tests, follow naming conventions, get 1+ approval before merge
