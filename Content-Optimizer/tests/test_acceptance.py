"""Final acceptance automated checks (Step 12).

These tests supplement the smoke tests by asserting that enriched recommendation
explanations (nearest_examples + top_features) are present when data & model exist.
They are intentionally tolerant of missing data (skip conditions) so they can run
early in development.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


@pytest.mark.acceptance
def test_recommendations_rich_explanations():
    r = client.get('/recommendations', params={'topK': 5})
    assert r.status_code == 200
    data = r.json()
    assert 'recommendations' in data
    recs = data['recommendations']
    # If dataset insufficient, skip richer checks
    if len(recs) < 3:
        pytest.skip('Less than 3 recommendations available; sample dataset may not be loaded.')
    first = recs[0]
    assert 'explanation' in first
    expl = first['explanation']
    for key in ['score', 'model', 'nearest_examples', 'top_features']:
        assert key in expl
    assert isinstance(expl['nearest_examples'], list)
    assert isinstance(expl['top_features'], list)
    assert len(expl['top_features']) <= 5
    # nearest examples should not reference the same contentId as the recommendation
    for ex in expl['nearest_examples']:
        assert ex['contentId'] != first['contentId']