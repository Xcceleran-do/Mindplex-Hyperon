import joblib
import json
from pathlib import Path
from typing import Tuple, Dict

MODEL_DIR = Path('./models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, metrics: Dict, out_dir: str = './models') -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    version = f"model_{int(__import__('time').time())}"
    model_path = out / f"{version}.joblib"
    joblib.dump(model, model_path)
    meta = {'version': version, 'metrics': metrics}
    (out / f"{version}.json").write_text(json.dumps(meta))
    return version


def load_latest_model(out_dir: str = './models') -> Tuple[object, Dict]:
    out = Path(out_dir)
    files = sorted(out.glob('*.joblib'))
    if not files:
        return None, {}
    model = joblib.load(files[-1])
    meta_path = files[-1].with_suffix('.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta
