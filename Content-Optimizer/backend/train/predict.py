"""Predict helper: load a trained model and score candidate feature vectors.

Usage:
  python backend/train/predict.py --model ./models/model_XXXXX.joblib --candidates data/candidates.npy --topk 10
"""
from __future__ import annotations

import argparse
import json
from typing import List
import numpy as np
import joblib


def load_model(path: str):
    return joblib.load(path)


def score_candidates(model, X: np.ndarray) -> List[float]:
    return model.predict(X).tolist()


def main(model_path: str, candidates_path: str | None, topk: int = 10):
    model = load_model(model_path)
    if candidates_path:
        X = np.load(candidates_path)
    else:
        print('No candidates file provided; expecting --candidates path to .npy')
        return
    scores = score_candidates(model, X)
    idx = np.argsort(scores)[::-1][:topk]
    results = [{"index": int(i), "score": float(scores[i])} for i in idx]
    print(json.dumps({"topk": results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()
    main(args.model, args.candidates, args.topk)
