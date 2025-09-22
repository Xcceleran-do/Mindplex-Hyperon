import os
import pandas as pd
import pytest

import model
import recommender
import db


def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        engagement = (
            row['likes'] * db.LIKES_WEIGHT
            + row['shares'] * db.SHARES_WEIGHT
            + row['comments'] * db.COMMENTS_WEIGHT
            + row['views'] * db.VIEWS_WEIGHT
        )
        data.append({
            'id': str(row['post_id']),
            'text': row['text'],
            'length': int(row['length']),
            'engagement_score': float(engagement),
            'topic': row['topic'],
            'style': row['style'],
            'structure': row['structure'],
        })
    return data


def test_end_to_end_recommendation_runs_quickly(tmp_path):
    """End-to-end smoke test:
    - Load CSV data (no Neo4j)
    - Train the model
    - Run recommender for a topic and assert we get a recommendation string
    """
    # Locate sample_data.csv by walking up from this file's directory —
    # tests may be executed with different working directories.
    cur = os.path.abspath(os.path.dirname(__file__))
    csv_path = None
    for _ in range(6):
        candidate = os.path.join(cur, 'CCRS', 'data', 'sample_data.csv')
        if os.path.exists(candidate):
            csv_path = candidate
            break
        candidate2 = os.path.join(cur, 'data', 'sample_data.csv')
        if os.path.exists(candidate2):
            csv_path = candidate2
            break
        cur = os.path.dirname(cur)
    assert csv_path is not None, "CSV not found: searched up from test file for CCRS/data/sample_data.csv"

    data = load_data_from_csv(csv_path)
    # Train model (uses small dataset; function will warn if small)
    model_obj, mean, std, feature_cols = model.train_model(data)
    assert model_obj is not None
    # Generate recommendation for topic 'AI'
    msg, top, cand_df, top3 = recommender.recommend_for_topic('AI', model_obj, mean, std, feature_cols)
    assert isinstance(msg, str)
    assert 'Recommended' in msg or 'Recommended:' in msg
