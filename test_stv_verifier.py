#!/usr/bin/env python3
"""
Quick test script for STV verifier
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments')))

from experiments.stv.stv_verifier import verify_proofs_stv

# Test with sample proof data
sample_proof = [
    [":", 
     [["rule:-", ["->", ["length", "1", "low"], ["engagement", "1", "high"]]], 
      ["fact:-", ["length", "1", "low"]]], 
     ["engagement", "1", "high", ["stv", "0.8", "0.7"]]
    ]
]

print("Testing STV Verifier...")
print("Input:", sample_proof)

try:
    result = verify_proofs_stv(sample_proof)
    print("✅ STV Verifier working!")
    print("Result:", result)
except Exception as e:
    print("❌ Error:", e)
    import traceback
    traceback.print_exc()
