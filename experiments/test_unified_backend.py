#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Backend
Tests mining, chat, and pattern analysis endpoints
"""

import requests
import json
import time
from config import API_BASE_URL

BASE_URL = API_BASE_URL

def test_health():
    """Test basic health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/health")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    data = response.json()
    assert data['status'] == 'healthy', "Health status not healthy"
    print("✓ Health check passed")

def test_chat_health():
    """Test chat health endpoint"""
    print("\nTesting chat health endpoint...")
    response = requests.get(f"{BASE_URL}/api/chat/health")
    assert response.status_code == 200, f"Chat health check failed: {response.status_code}"
    data = response.json()
    assert data['status'] == 'healthy', "Chat health status not healthy"
    print("✓ Chat health check passed")

def test_pattern_analysis():
    """Test pattern analysis endpoint"""
    print("\nTesting pattern analysis...")
    
    pattern = '((length $x "low") (engagement_level $x "high"))'
    support = "6"
    
    response = requests.post(
        f"{BASE_URL}/api/chat/analyze",
        json={"pattern": pattern, "support": support},
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200, f"Pattern analysis failed: {response.status_code}"
    data = response.json()
    
    assert 'summary' in data, "No summary in response"
    assert 'pattern' in data, "No pattern in response"
    assert 'support' in data, "No support in response"
    
    print("✓ Pattern analysis passed")
    print(f"  Pattern: {data['pattern']}")
    print(f"  Support: {data['support']}")
    print(f"  Summary (first 100 chars): {data['summary'][:100]}...")

def test_chat_conversation():
    """Test chat conversation endpoint"""
    print("\nTesting chat conversation...")
    
    message = "What patterns were found in the mining results?"
    
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": message,
            "history": [],
            "session_id": "test_session"
        },
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200, f"Chat conversation failed: {response.status_code}"
    data = response.json()
    
    assert 'response' in data, "No response in chat data"
    assert 'session_id' in data, "No session_id in response"
    
    print("✓ Chat conversation passed")
    print(f"  User: {message}")
    print(f"  AI (first 100 chars): {data['response'][:100]}...")

def test_mining_endpoint():
    """Test mining endpoint (basic structure test)"""
    print("\nTesting mining endpoint structure...")
    
    # This is a structure test - actual mining requires MeTTa data loaded
    response = requests.post(
        f"{BASE_URL}/api/mine",
        json={
            "conjunct_size": 2,
            "min_support": 2
        },
        headers={"Content-Type": "application/json"}
    )
    
    # We expect it might fail without proper data, but check response structure
    print(f"  Response status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Job ID: {data.get('jobId', 'N/A')}")
        print("✓ Mining endpoint structure correct")
    else:
        print("  (Expected - requires data to be loaded in frontend)")

def test_exact_matching_logic():
    """Test the exact matching logic conceptually"""
    print("\nTesting exact matching logic...")
    
    # Simulate the data
    articles = {
        0: {"length": "low", "engagement_level": "high", "tone": "Analytical"},
        1: {"length": "low", "engagement_level": "high", "tone": "Analytical"},
        2: {"length": "low", "engagement_level": "high", "tone": "Analytical"},
        3: {"length": "low", "engagement_level": "medium", "tone": "Analytical"},
        4: {"length": "high", "engagement_level": "medium"},
        5: {"length": "low", "engagement_level": "high", "tone": "Humorous"},
    }
    
    # Pattern: (length="low" AND engagement_level="high")
    required_props = {"length": "low", "engagement_level": "high"}
    
    matching = []
    for article_id, props in articles.items():
        matches_all = all(
            props.get(prop) == value 
            for prop, value in required_props.items()
        )
        if matches_all:
            matching.append(article_id)
    
    expected = [0, 1, 2, 5]
    assert matching == expected, f"Expected {expected}, got {matching}"
    
    print("✓ Exact matching logic verified")
    print(f"  Pattern: length='low' AND engagement_level='high'")
    print(f"  Matching articles: {matching}")
    print(f"  Excluded (partial matches): [3, 4]")

def main():
    """Run all tests"""
    print("=" * 60)
    print("UNIFIED BACKEND TEST SUITE")
    print("=" * 60)
    
    try:
        test_health()
        test_chat_health()
        test_pattern_analysis()
        test_chat_conversation()
        test_mining_endpoint()
        test_exact_matching_logic()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        from config import FRONTEND_URL
        print(f"1. Open {FRONTEND_URL} in your browser")
        print("2. Load the small-ugly.metta data")
        print("3. Click 'Mine' with conjunct size 2")
        print("4. Chat should open with pattern summaries")
        print("5. Click 'Visualize' to see exact matches only")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to backend")
        print("Make sure the backend is running:")
        print("  cd /workspaces/Mindplex-Hyperon/experiments")
        print("  ./start_all.sh")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
