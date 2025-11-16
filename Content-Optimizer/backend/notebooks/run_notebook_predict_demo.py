import sys
from pathlib import Path
import json
import subprocess

# compute repo root relative to this script location (works independent of cwd)
repo_root = Path(__file__).resolve().parents[2]
# prefer notebook parquet then test then dataset
parquet_paths = [repo_root / 'data' / 'notebook_dataset.parquet', repo_root / 'data' / 'test_dataset.parquet', repo_root / 'data' / 'dataset.parquet']
parquet = None
for p in parquet_paths:
    if p.exists():
        parquet = p
        break

if parquet is None:
    print('No parquet dataset found in data/. Run prepare_dataset.py first.')
    sys.exit(2)

import pandas as pd
import numpy as np

df = pd.read_parquet(parquet)
if 'features' not in df.columns:
    print('Parquet dataset does not contain "features" column. Aborting.')
    sys.exit(3)

N = min(10, len(df))
candidates = np.vstack([np.array(f) for f in df['features'].iloc[:N]])

candidates_path = repo_root / 'data' / 'notebook_candidates.npy'
np.save(candidates_path, candidates)
print('Saved candidates', candidates_path, 'shape=', candidates.shape)

# find model
models_dir = repo_root / 'models'
model_files = list(models_dir.glob('*.joblib'))
if not model_files:
    print('No model .joblib found in', models_dir)
    sys.exit(4)

model_file = str(model_files[0])
print('Using model', model_file)

# call predict.py CLI
cmd = [sys.executable, str(repo_root / 'backend' / 'train' / 'predict.py'), '--model', model_file, '--candidates', str(candidates_path), '--topk', '5']
print('$', ' '.join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print('returncode=', res.returncode)
print(res.stdout)
if res.stderr:
    print('STDERR:\n', res.stderr)

# print top-k result file if predict wrote anything to stdout as json
try:
    out = json.loads(res.stdout)
    print(json.dumps(out, indent=2))
except Exception:
    pass
