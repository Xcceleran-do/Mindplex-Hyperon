#!/usr/bin/env python3
"""
End-to-end integration test for Chainer + STV + getChainerResult
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments')))

from experiments.mining_api import getChainerResult

def real_integration_test():
    print("🚀 Starting Real Integration Test for Chainer + STV")
    print("="*60)

    queries = [
        '(engagement 1 "high")',
        '(length 1 "low")',
        '(tone 1 "Analytical")',
        '(engagement 9 "high")',
        '(length 5 "medium")'
    ]

    for query in queries:
        print(f"\n📝 Running query: {query}")
        try:
            result = getChainerResult(query, depth=3)

            proofs = result.get('proof_count', 0)
            justification = result.get('justification', '')
            status = result.get('status', 'unknown')

            print(f"📊 Status: {status}")
            print(f"📊 Proof count: {proofs}")
            print(f"📊 Justification length: {len(justification)}")

            # Basic verification
            if status != 'success':
                print(f"❌ ERROR: Query did not succeed")
            elif proofs == 0:
                print(f"⚠️ WARNING: No proofs found")
            else:
                print(f"✅ SUCCESS: Proofs found and justification generated")

            # Optional: print short preview of justification
            preview = justification[:300].replace('\n', ' ')
            print(f"📄 Justification preview: {preview}...")

        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            import traceback
            traceback.print_exc()

    print("\n🏁 Real Integration Test Complete")

if __name__ == "__main__":
    real_integration_test()
