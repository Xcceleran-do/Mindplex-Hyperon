"""Train ranker: load prepared dataset, train a LightGBM regressor, and save model + metadata."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def train_model(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> lgb.LGBMRegressor:
    params = params or {}
    model = lgb.LGBMRegressor(**{"n_estimators": 100, **params})
    model.fit(X, y)
    return model


def save_model(model, metrics: dict, out_dir: str) -> Tuple[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    version = f"model_{int(__import__('time').time())}"
    model_path = out / f"{version}.joblib"
    meta_path = out / f"{version}.json"
    joblib.dump(model, model_path)
    meta = {"version": version, "metrics": metrics}
    meta_path.write_text(json.dumps(meta))
    return version, model_path


def main(input_path: str = 'data/dataset.parquet', out_dir: str = './models'):
    df = load_dataset(input_path)
    # expand features into matrix
    X = np.vstack(df['features'].values)
    y = df['target'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    preds = model.predict(X_val)
    r2 = float(r2_score(y_val, preds))
    mse = float(mean_squared_error(y_val, preds))
    metrics = {"r2": r2, "mse": mse}

    version, model_path = save_model(model, metrics, out_dir)
    print("Trained model:", version)
    print("Metrics:", metrics)
    print("Model saved to:", model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/dataset.parquet')
    parser.add_argument('--out_dir', default='./models')
    args = parser.parse_args()
    main(input_path=args.input, out_dir=args.out_dir)
