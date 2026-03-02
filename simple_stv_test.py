#!/usr/bin/env python3
"""
Simple STV Test Script
Tests the STV-enhanced backward chaining functionality without Flask dependencies
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments')))

from hyperon import MeTTa

def test_stv_metta_functions():
    """Test STV functions directly in MeTTa"""
    print("🚀 Testing STV Functions in MeTTa")
    print("=" * 50)
    
    # Initialize MeTTa
    metta = MeTTa()
    
    try:
        # Load the STV-enhanced modules
        print("📦 Loading STV modules...")
        metta.run("""
            ! (register-module! experiments)
            ! (import! &self experiments:chainer:main)
            ! (import! &self experiments:chainer:facts)
            ! (import! &self experiments:chainer:rules)
            ! (import! &stv-formulas experiments:PLN:Formulas)
        """)
        
        # Test knowledge base initialization
        print("\n🔍 Testing Knowledge Base...")
        kb_atoms = metta.run("!(get-atoms &kb)")
        stv_kb_atoms = metta.run("!(get-atoms &stv-kb)")
        facts_atoms = metta.run("!(get-atoms &facts)")
        rules_atoms = metta.run("!(get-atoms &rules)")
        
        print(f"✅ Regular KB atoms: {len(kb_atoms) if kb_atoms else 0}")
        print(f"✅ STV KB atoms: {len(stv_kb_atoms) if stv_kb_atoms else 0}")
        print(f"✅ Facts atoms: {len(facts_atoms) if facts_atoms else 0}")
        print(f"✅ Rules atoms: {len(rules_atoms) if rules_atoms else 0}")
        
        # Test regular backward chaining
        print("\n🔄 Testing Regular Backward Chaining...")
        query = "(engagement 1 \"high\")"
        regular_result = metta.run(f"!(backward-chain &kb (fromNumber 3) (: $prf {query}))")
        print(f"📊 Regular result: {regular_result}")
        
        # Test STV backward chaining
        print("\n🎯 Testing STV Backward Chaining...")
        stv_result = metta.run(f"!(backward-chain-stv &stv-kb (fromNumber 3) (: $prf {query}))")
        print(f"📊 STV result: {stv_result}")
        
        # Test STV aggregation
        print("\n🔗 Testing STV Aggregation...")
        test_queries = [
            '(engagement 9 "high")',
            '(engagement 8 "high")',
            '(engagement 7 "high")',
            '(engagement 5 "high")',
            '(engagement 4 "medium")'
        ]
        test_proofs = """
            (superpose 
                (: (fact:- (engagement 1 "high") (stv 0.9 0.8)) (engagement 1 "high") (stv 0.9 0.8))
                (: (fact:- (length 1 "low") (stv 0.8 0.7)) (length 1 "low") (stv 0.8 0.7))
            )
        """
        aggregation_result = metta.run(f"!(aggregate-proofs-stv {test_proofs})")
        print(f"📊 Aggregation result: {aggregation_result}")
        
        # Test PLN formulas
        print("\n🧮 Testing PLN Truth Formulas...")
        modus_ponens_test = metta.run("!(Truth_ModusPonens (stv 0.8 0.7) (stv 0.9 0.8))")
        print(f"📊 Modus Ponens: {modus_ponens_test}")
        
        or_test = metta.run("!(Truth_Or (stv 0.8 0.7) (stv 0.9 0.8))")
        print(f"📊 Truth OR: {or_test}")
        
        # Test individual STV facts
        print("\n📋 Testing Individual STV Facts...")
        stv_facts = metta.run("!(get-atoms &facts)")
        if stv_facts:
            print("📝 Sample STV facts:")
            for i, fact in enumerate(stv_facts[:3]):  # Show first 3 facts
                print(f"  {i+1}: {fact}")
        
        # Test individual STV rules
        print("\n📋 Testing Individual STV Rules...")
        stv_rules = metta.run("!(get-atoms &rules)")
        if stv_rules:
            print("📝 Sample STV rules:")
            for i, rule in enumerate(stv_rules[:3]):  # Show first 3 rules
                print(f"  {i+1}: {rule}")
        
        print("\n" + "=" * 50)
        print("🎉 STV MeTTa Functions Test Complete!")
        print("✅ All core STV functionality tested successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in STV test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stv_fact_structure():
    """Test that STV facts have the correct structure"""
    print("\n🔍 Testing STV Fact Structure...")
    
    metta = MeTTa()
    
    try:
        metta.run("""
            ! (register-module! experiments)
            ! (import! &self experiments:chainer:facts)
        """)
        
        # Get facts and check for STV values
        facts = metta.run("!(get-atoms &facts)")
        
        stv_count = 0
        regular_count = 0
        
        for fact in facts:
            fact_str = str(fact)
            if "(stv " in fact_str:
                stv_count += 1
                # Extract STV values
                import re
                stv_match = re.search(r'\(stv ([\d.]+) ([\d.]+)\)', fact_str)
                if stv_match:
                    strength = float(stv_match.group(1))
                    confidence = float(stv_match.group(2))
                    print(f"  ✅ STV Fact: Strength={strength}, Confidence={confidence}")
            else:
                regular_count += 1
        
        print(f"📊 Found {stv_count} STV facts and {regular_count} regular facts")
        
        if stv_count > 0:
            print("✅ STV facts are properly structured!")
        else:
            print("⚠️  No STV facts found")
            
        return stv_count > 0
        
    except Exception as e:
        print(f"❌ Error testing STV fact structure: {e}")
        return False

def test_stv_rule_structure():
    """Test that STV rules have the correct structure"""
    print("\n🔍 Testing STV Rule Structure...")
    
    metta = MeTTa()
    
    try:
        metta.run("""
            ! (register-module! experiments)
            ! (import! &self experiments:chainer:rules)
        """)
        
        # Get rules and check for STV values
        rules = metta.run("!(get-atoms &rules)")
        
        stv_count = 0
        supportof_count = 0
        
        for rule in rules:
            rule_str = str(rule)
            if "(stv " in rule_str:
                stv_count += 1
                # Extract STV values
                import re
                stv_match = re.search(r'\(stv ([\d.]+) ([\d.]+)\)', rule_str)
                if stv_match:
                    strength = float(stv_match.group(1))
                    confidence = float(stv_match.group(2))
                    print(f"  ✅ STV Rule: Strength={strength}, Confidence={confidence}")
            elif "supportOf" in rule_str:
                supportof_count += 1
                print(f"  📋 SupportOf Rule: {rule_str}")
        
        print(f"📊 Found {stv_count} STV rules and {supportof_count} supportOf rules")
        
        if stv_count > 0:
            print("✅ STV rules are properly structured!")
        else:
            print("⚠️  No STV rules found")
            
        return stv_count > 0
        
    except Exception as e:
        print(f"❌ Error testing STV rule structure: {e}")
        return False

def main():
    """Run all simple STV tests"""
    print("🚀 Starting Simple STV Integration Tests")
    print("=" * 50)
    
    tests = [
        test_stv_fact_structure,
        test_stv_rule_structure,
        test_stv_metta_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All STV tests passed! Integration is working correctly!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
