#!/usr/bin/env python3
"""
Test script for getChainerResult function with STV verification
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments')))

from experiments.mining_api import getChainerResult

def test_getChainerResult():
    """Test getChainerResult function"""
    print("🧪 Testing getChainerResult function...")
    
    test_queries = [
        '(engagement 1 "high")',
        '(length 1 "low")', 
        '(tone 1 "Analytical")'
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: {query}")
        
        try:
            result = getChainerResult(query, depth=3)
            print(f"✅ getChainerResult completed successfully!")
            print(f"📊 Status: {result.get('status', 'unknown')}")
            print(f"📊 Proof count: {result.get('proof_count', 0)}")
            print(f"📊 Justification length: {len(result.get('justification', ''))}")
            
            if result.get('status') == 'success':
                print("✅ SUCCESS: Found proofs and generated justification")
            elif result.get('status') == 'no_proof':
                print("⚠️  WARNING: No proofs found")
            else:
                print(f"❌ ERROR: Unexpected status - {result.get('status')}")
                
        except Exception as e:
            print(f"❌ Error testing getChainerResult: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_getChainerResult()
