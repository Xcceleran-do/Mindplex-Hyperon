from fastapi import APIRouter, Header, HTTPException
import os
from pathlib import Path

from typing import Optional, List, Dict, Any

router = APIRouter()

ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', 'changeme')

def check_token(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail='invalid admin token')

@router.post('/admin/retrain')
async def retrain(x_admin_token: str = Header(None)):
    check_token(x_admin_token)
    # trigger dataset prepare and training synchronously (best-effort)
    try:
        from backend.train.prepare_dataset import assemble_dataset
        from backend.train.train_ranker import main as train_main
        df = assemble_dataset(limit=1000)
        data_dir = Path('data'); data_dir.mkdir(exist_ok=True)
        dataset_path = data_dir / 'dataset.parquet'
        df.to_parquet(dataset_path, index=False)
        models_dir = Path('models'); models_dir.mkdir(exist_ok=True)
        train_main(input_path=str(dataset_path), out_dir=str(models_dir))
        # refresh recommendation cache
        from backend.app.api.recommendations import refresh_cache_programmatically
        refresh_cache_programmatically()
        return {'status': 'retrained', 'dataset': str(dataset_path), 'models_dir': str(models_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'retrain failed: {e}')

@router.post('/admin/project_graph')
async def project_graph(x_admin_token: str = Header(None)):
    check_token(x_admin_token)
    # TODO: run cypher via neo4j client
    return {'status': 'project_triggered'}

@router.get('/admin/status')
async def status(x_admin_token: str = Header(None)):
    check_token(x_admin_token)
    return {'models': [], 'last_retrain': None}


@router.post('/admin/orchestrate')
async def orchestrate(x_admin_token: str = Header(None), limit: Optional[int] = None):
    """Run the end-to-end pipeline: assemble dataset -> train -> refresh cache.

    This is a synchronous orchestrator suitable for local development.
    """
    check_token(x_admin_token)
    try:
        from backend.train.prepare_dataset import assemble_dataset
        from backend.train.train_ranker import main as train_main
        df = assemble_dataset(limit=limit or 1000)
        data_dir = Path('data'); data_dir.mkdir(exist_ok=True)
        dataset_path = data_dir / 'dataset.parquet'
        df.to_parquet(dataset_path, index=False)
        models_dir = Path('models'); models_dir.mkdir(exist_ok=True)
        train_main(input_path=str(dataset_path), out_dir=str(models_dir))
        from backend.app.api.recommendations import refresh_cache_programmatically
        refresh_cache_programmatically()
        return {'status': 'ok', 'dataset': str(dataset_path), 'models_dir': str(models_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'orchestrate failed: {e}')


# --- Central full E2E Orchestrator -------------------------------------------------

def _read_cypher_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f'Cypher file not found: {path}')
    return path.read_text(encoding='utf-8')

def _split_cypher_statements(content: str) -> List[str]:
    """Very simple splitter: remove block comments, drop // lines, split on semicolons.

    Note: Neo4j driver executes one statement per run; we run them sequentially.
    """
    import re
    # Remove /* ... */ block comments
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    # Remove // line comments
    lines = []
    for ln in content.splitlines():
        stripped = ln.strip()
        if stripped.startswith('//'):
            continue
        lines.append(ln)
    content = "\n".join(lines)
    # Split on semicolons
    parts = [p.strip() for p in content.split(';')]
    # Filter empty parts
    return [p for p in parts if p]

def _run_cypher_script(path: Path) -> List[dict]:
    from backend.app.services.neo4j_client import run_write
    content = _read_cypher_file(path)
    statements = _split_cypher_statements(content)
    results = []
    for stmt in statements:
        try:
            res = run_write(stmt)
            results.append({'ok': True, 'rows': len(res)})
        except Exception as e:  # provide context which statement failed
            raise HTTPException(status_code=500, detail=f'Cypher failed in {path.name}: {e}')
    return results

def _compute_and_persist_text_embeddings(batch_size: int = 500):
    """Compute text embeddings for all Content nodes and persist in batches.

    Requires sentence-transformers installed. Batches by LIMIT/OFFSET-like pattern.
    """
    from backend.app.services.embeddings import compute_text_embedding, save_text_embeddings_to_neo4j
    from backend.app.services.neo4j_client import run_read, get_driver

    # Fetch all content ids and titles
    rows = run_read("MATCH (c:Content) RETURN c.contentId AS contentId, c.title AS title ORDER BY c.contentId")
    if not rows:
        return {'persisted': 0}
    total = 0
    driver = get_driver()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        titles = [r['title'] or '' for r in batch]
        emb = compute_text_embedding(titles)
        payload = [{'contentId': batch[j]['contentId'], 'embedding': emb[j]} for j in range(len(batch))]
        save_text_embeddings_to_neo4j(driver, payload)
        total += len(batch)
    return {'persisted': total}


def run_full_pipeline(dataset_limit: Optional[int] = 5000) -> Dict[str, Any]:
    """Run the entire pipeline end-to-end. Returns a summary dict.

    Steps:
      1) Create schema (constraints, indexes)
      2) Import CSV into Neo4j from neo4j/import/sample_content.csv
      3) Project GDS graph and compute GraphSAGE embeddings
      4) Compute & persist text embeddings for all Content
      5) Assemble dataset (optionally capped by dataset_limit)
      6) Train ranker and save artifacts under models/
      7) Refresh recommendation cache
    """
    # 1-3: Cypher scripts
    cypher_dir = Path('neo4j') / 'cypher'
    steps = [
        cypher_dir / '00_create_schema.cypher',
        cypher_dir / '01_import_from_csv.cypher',
        cypher_dir / '02_gds_project_and_gsage.cypher',
    ]
    cypher_results: Dict[str, Any] = {}
    for p in steps:
        cypher_results[p.name] = _run_cypher_script(p)

    # 4: Text embeddings
    emb_result = _compute_and_persist_text_embeddings()

    # 5-6: Dataset + training
    from backend.train.prepare_dataset import assemble_dataset
    from backend.train.train_ranker import main as train_main
    df = assemble_dataset(limit=dataset_limit)
    if df.empty:
        raise RuntimeError('Assembled dataset is empty after import/embeddings')
    data_dir = Path('data'); data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / 'dataset.parquet'
    df.to_parquet(dataset_path, index=False)
    models_dir = Path('models'); models_dir.mkdir(exist_ok=True)
    train_main(input_path=str(dataset_path), out_dir=str(models_dir))

    # 7: Refresh cache
    from backend.app.api.recommendations import refresh_cache_programmatically
    refresh_cache_programmatically()

    return {
        'status': 'ok',
        'cypher': cypher_results,
        'embeddings': emb_result,
        'dataset': str(dataset_path),
        'models_dir': str(models_dir)
    }


@router.post('/admin/orchestrate_full')
async def orchestrate_full(x_admin_token: str = Header(None), dataset_limit: Optional[int] = 5000):
    """HTTP wrapper to run the entire pipeline end-to-end in one call."""
    check_token(x_admin_token)
    try:
        return run_full_pipeline(dataset_limit=dataset_limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'orchestrate_full failed: {e}')
