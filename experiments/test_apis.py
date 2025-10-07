#!/usr/bin/env python3
"""
Test script to verify all APIs are working correctly
"""

import requests
import json
import time

def test_health_checks():
    """Test health endpoints for both APIs"""
    print("🔍 Testing Health Endpoints...")
    
    # Test Mining API
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Mining API (port 5000) is healthy")
        else:
            print(f"❌ Mining API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Mining API is not accessible: {e}")
    
    # Test Chat API
    try:
        response = requests.get('http://localhost:5001/api/chat/health', timeout=5)
        if response.status_code == 200:
            print("✅ Chat API (port 5001) is healthy")
        else:
            print(f"❌ Chat API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Chat API is not accessible: {e}")
    
    print()

def test_mining():
    """Test the mining endpoint"""
    print("⛏️  Testing Mining Endpoint...")
    
    try:
        response = requests.post(
            'http://localhost:5000/api/mine',
            json={'conjunction_count': 2},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Mining completed successfully")
            print(f"   Job ID: {data.get('jobId', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Patterns found: {len(data.get('result', []))}")
            
            if data.get('result'):
                print(f"\n   First pattern:")
                print(f"   {json.dumps(data['result'][0], indent=6)}")
            
            return data.get('result', [])
        else:
            print(f"❌ Mining failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Mining test failed: {e}")
        return []
    
    print()

def test_chat_analyze(pattern, support):
    """Test the chat analyze endpoint"""
    print("🤖 Testing Chat Analyze Endpoint...")
    
    try:
        response = requests.post(
            'http://localhost:5001/api/chat/analyze',
            json={
                'pattern': pattern,
                'support': support
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pattern analysis completed")
            print(f"\n   Summary preview:")
            summary_lines = data.get('summary', '').split('\n')[:5]
            for line in summary_lines:
                print(f"   {line}")
            print(f"   ...")
            return True
        else:
            print(f"❌ Chat analyze failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat analyze test failed: {e}")
        return False
    
    print()

def test_chat():
    """Test the chat endpoint"""
    print("💬 Testing Chat Endpoint...")
    
    try:
        response = requests.post(
            'http://localhost:5001/api/chat',
            json={
                'message': 'Hello! Can you help me understand pattern mining?',
                'history': []
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat completed successfully")
            print(f"\n   AI Response preview:")
            response_preview = data.get('response', '')[:200]
            print(f"   {response_preview}...")
            return True
        else:
            print(f"❌ Chat failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False
    
    print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 AtomSpace Visualizer API Test Suite")
    print("=" * 60)
    print()
    
    # Test 1: Health checks
    test_health_checks()
    
    # Test 2: Mining
    patterns = test_mining()
    
    # Test 3: Chat analyze (if we have patterns)
    if patterns:
        print()
        test_chat_analyze(patterns[0]['pattern'], patterns[0]['support'])
    
    # Test 4: Chat
    print()
    test_chat()
    
    print()
    print("=" * 60)
    print("🎉 Test Suite Completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Open http://localhost:3000 in your browser")
    print("2. Click 'Mine Neural Gold' button")
    print("3. Watch the chat interface automatically open")
    print("4. Click 'Visualize' on any pattern to see exact matches")
    print()

if __name__ == '__main__':
    main()
