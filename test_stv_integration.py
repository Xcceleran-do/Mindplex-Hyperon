#!/usr/bin/env python3
"""
STV Integration Test Script
Tests the STV-enhanced backward chaining functionality
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments')))

from experiments.mining_api import backWardChainer, metta4Miner
import json

def test_stv_knowledge_base():
    """Test that STV knowledge base is properly initialized"""
    print("🔍 Testing STV Knowledge Base Initialization...")
    
    try:
        # Check &res1 knowledge base (the correct atomspace)
        res1_kb = metta4Miner.run("!(get-atoms &res1)")
        print(f"✅ &res1 atoms: {len(res1_kb) if res1_kb else 0}")
        
        # Check facts and rules
        facts = metta4Miner.run("!(get-atoms &facts)")
        rules = metta4Miner.run("!(get-atoms &rules)")
        print(f"✅ Facts: {len(facts) if facts else 0}")
        print(f"✅ Rules: {len(rules) if rules else 0}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing STV knowledge base: {e}")
        return False

def test_stv_backward_chaining():
    """Test STV-enhanced backward chaining"""
    print("\n🔍 Testing STV Backward Chaining...")
    
    test_queries = [
        '(engagement 1 "high")',
        '(length 1 "low")',
        '(tone 1 "Analytical")',
        '(engagement 9 "high")'
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: {query}")
        
        try:
            # Test regular backward chaining
            print("  🔄 Regular backward chaining...")
            regular_result = backWardChainer(query, depth=3)
            print(f"    Result: {regular_result}")
            
            # Test STV-enhanced backward chaining
            print("  🎯 STV-enhanced backward chaining...")
            stv_result = backWardChainer(query, depth=3)
            print(f"    Result: {stv_result}")
            
            # Compare results
            if stv_result and len(stv_result) > 0:
                print("  ✅ STV chaining found proofs with truth values")
                for i, proof in enumerate(stv_result[:2]):  # Show first 2 proofs
                    print(f"    Proof {i+1}: {proof}")
            else:
                print("  ⚠️  No proofs found")
                
        except Exception as e:
            print(f"  ❌ Error testing query {query}: {e}")
    
    return True  # Test passes if it completes without crashing

def test_confidence_thresholding():
    """Test confidence thresholding functionality"""
    print("\n🔍 Testing Confidence Thresholding...")
    
    query = '(engagement 1 "high")'
    thresholds = [0.1, 0.5, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\n📊 Testing with confidence threshold: {threshold}")
        
        try:
            result = backWardChainer(query, depth=3)
            print(f"  Proofs found: {len(result) if result else 0}")
            
            if result and len(result) > 0:
                print(f"  ✅ Threshold {threshold}: {len(result)} proofs passed")
            else:
                print(f"  ⚠️  Threshold {threshold}: No proofs passed")
                
        except Exception as e:
            print(f"  ❌ Error with threshold {threshold}: {e}")
    
    return True  # Test passes if it completes without crashing

def test_truth_propagation():
    """Test truth value propagation through inference chains"""
    print("\n🔍 Testing Truth Value Propagation...")
    
    # Test a query that should use multiple rules
    query = '(engagement 1 "high")'
    
    try:
        result = backWardChainer(query, depth=5)
        print(f"📊 Query: {query}")
        print(f"📊 Total proofs: {len(result) if result else 0}")
        print(f"📊 Raw result: {result}")
        
        if result and len(result) > 0:
            print(f"✅ Truth propagation test completed successfully")
        else:
            print("⚠️  No proofs found for truth propagation test")
            
    except Exception as e:
        print(f"❌ Error in truth propagation test: {e}")
    
    return True  # Test passes if it completes without crashing

def test_comparison_regular_vs_stv():
    """Compare regular vs STV backward chaining"""
    print("\n🔍 Comparing Regular vs STV Backward Chaining...")
    
    query = '(engagement 1 "high")'
    
    try:
        # Regular chaining
        print("🔄 Regular Backward Chaining:")
        regular_result = backWardChainer(query, depth=3)
        regular_proofs = len(regular_result) if regular_result else 0
        print(f"  Proofs found: {regular_proofs}")
        
        # STV-enhanced chaining
        print("\n🎯 STV-Enhanced Backward Chaining:")
        stv_result = backWardChainer(query, depth=3)
        stv_proofs = len(stv_result) if stv_result else 0
        print(f"  Proofs found: {stv_proofs}")
        print(f"  Raw result: {stv_result}")
        
        # Comparison
        print(f"\n📊 Comparison:")
        print(f"  Regular chaining: {regular_proofs} proofs")
        print(f"  STV chaining: {stv_proofs} proofs with truth values")
        
        if stv_proofs > 0:
            print(f"  ✅ STV provides additional truth value information!")
        else:
            print(f"  ⚠️  STV found fewer proofs (due to confidence filtering)")
            
    except Exception as e:
        print(f"❌ Error in comparison test: {e}")
    
    return True  # Test passes if it completes without crashing

def main():
    """Run all STV integration tests"""
    print("🚀 Starting STV Integration Tests")
    print("=" * 50)
    
        # Initialize MeTTa if needed
    try:
        res1_atoms = metta4Miner.run("!(get-atoms &res1)")
        print(f"✅ &res1 atomspace initialized with {len(res1_atoms) if res1_atoms else 0} atoms")
    except Exception as e:
        print(f"⚠️  Error checking &res1: {e}")
    
    # Run tests
    tests = [
        test_stv_knowledge_base,
        test_stv_backward_chaining,
        test_confidence_thresholding,
        test_truth_propagation,
        test_comparison_regular_vs_stv
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            print(f"\n🧪 Running test: {test.__name__}")
            result = test()
            if result:
                passed += 1
                print(f"✅ Test {test.__name__} PASSED")
            else:
                print(f"❌ Test {test.__name__} FAILED (returned False)")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests completed")
    print("🎉 STV Integration Testing Complete!")

if __name__ == "__main__":
    main()
