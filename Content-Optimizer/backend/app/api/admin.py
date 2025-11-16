from fastapi import APIRouter, Header, HTTPException
import os
from pathlib import Path

from typing import Optional

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
