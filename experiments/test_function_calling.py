#!/usr/bin/env python3
"""
Test Automatic Function Calling
This test verifies that the AI can automatically call functions
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def wait_for_backend():
    """Wait for backend to be ready"""
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=2)
            if response.status_code == 200:
                print("✓ Backend is ready")
                return True
        except:
            print(f"  Waiting for backend... ({i+1}/{max_retries})")
            time.sleep(2)
    return False

def test_function_calling():
    """Test automatic function calling with various queries"""
    
    print("\n" + "="*60)
    print("AUTOMATIC FUNCTION CALLING TESTS")
    print("="*60)
    
    test_cases = [
        {
            "name": "Get Mining Results",
            "message": "What patterns were found in the latest mining?",
            "expected_function": "get_mining_results"
        },
        {
            "name": "Pattern Statistics",
            "message": "Give me statistics about the patterns",
            "expected_function": "get_pattern_statistics"
        },
        {
            "name": "Analyze Pattern",
            "message": "Analyze this pattern: ((length $x \"low\") (engagement_level $x \"high\"))",
            "expected_function": "analyze_specific_pattern"
        },
        {
            "name": "Visualize Request",
            "message": "Can you visualize the pattern ((tone $x \"Analytical\"))?",
            "expected_function": "visualize_pattern_request"
        },
        {
            "name": "General Question (No Function)",
            "message": "What is pattern mining?",
            "expected_function": None
        }
    ]
    
    session_id = f"test_{int(time.time())}"
    history = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        print(f"User: {test_case['message']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json={
                    "message": test_case['message'],
                    "history": history,
                    "session_id": session_id
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"✗ Failed: HTTP {response.status_code}")
                print(f"  Error: {response.text}")
                continue
            
            data = response.json()
            
            # Check for function calls
            function_calls = data.get('functionCalls', [])
            ai_response = data.get('response', '')
            
            if function_calls:
                print(f"🔧 Function calls made: {len(function_calls)}")
                for fc in function_calls:
                    print(f"   - {fc['name']}({fc.get('args', {})})")
                    print(f"     Result: {fc.get('result', {})}")
            else:
                print("  No function calls made")
            
            # Verify expected function was called
            if test_case['expected_function']:
                called_functions = [fc['name'] for fc in function_calls]
                if test_case['expected_function'] in called_functions:
                    print(f"✓ Expected function '{test_case['expected_function']}' was called")
                else:
                    print(f"⚠️  Expected '{test_case['expected_function']}' but got: {called_functions}")
            else:
                if not function_calls:
                    print("✓ Correctly handled without function calls")
            
            # Show AI response
            response_preview = ai_response[:150] + "..." if len(ai_response) > 150 else ai_response
            print(f"AI: {response_preview}")
            
            # Update history
            history.append({'role': 'user', 'content': test_case['message']})
            history.append({'role': 'assistant', 'content': ai_response})
            
            # Small delay between tests
            time.sleep(1)
            
        except requests.exceptions.Timeout:
            print("✗ Request timed out (AI might be processing)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("Automatic function calling test completed!")
    print("\nKey Points:")
    print("✓ AI should automatically detect when to call functions")
    print("✓ Functions are executed and results sent back to AI")
    print("✓ AI incorporates function results into its response")
    print("✓ No manual intervention needed")

def main():
    print("Waiting for backend to be ready...")
    if not wait_for_backend():
        print("✗ Backend not available. Make sure it's running:")
        print("  cd /workspaces/Mindplex-Hyperon/experiments && ./start_all.sh")
        return 1
    
    time.sleep(2)  # Give it extra time to fully initialize
    
    test_function_calling()
    return 0

if __name__ == "__main__":
    exit(main())
