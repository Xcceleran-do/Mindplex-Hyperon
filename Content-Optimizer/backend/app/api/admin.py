from fastapi import APIRouter, Header, HTTPException
import os

router = APIRouter()

ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', 'changeme')

def check_token(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail='invalid admin token')

@router.post('/admin/retrain')
async def retrain(x_admin_token: str = Header(None)):
    check_token(x_admin_token)
    # TODO: trigger dataset prepare and training
    return {'status': 'retrain_triggered'}

@router.post('/admin/project_graph')
async def project_graph(x_admin_token: str = Header(None)):
    check_token(x_admin_token)
    # TODO: run cypher via neo4j client
    return {'status': 'project_triggered'}

@router.get('/admin/status')
async def status(x_admin_token: str = Header(None)):
    check_token(x_admin_token)
    return {'models': [], 'last_retrain': None}
