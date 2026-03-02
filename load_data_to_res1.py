#!/usr/bin/env python3
"""
Load data.metta into &res1 atomspace for STV backward chaining
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments')))

from experiments.mining_api import metta4Miner

def load_data_to_res1():
    """Load data.metta into &res1 atomspace"""
    print("🚀 Loading data.metta into &res1 atomspace...")
    
    try:
        # Read the data.metta file
        data_file = "/home/henok/Desktop/projects/Mindplex-Hyperon/experiments/atomspace_visualizer/public/data.metta"
        
        with open(data_file, 'r') as f:
            data_content = f.read()
        
        print(f"📁 Read data.metta file ({len(data_content)} characters)")
        
        # Parse and add each fact to &res1
        facts = data_content.strip().split('\n')
        loaded_count = 0
        
        for fact in facts:
            fact = fact.strip()
            if fact and not fact.startswith(';'):  # Skip empty lines and comments
                try:
                    # Parse the fact
                    parsed_fact = metta4Miner.parse_single(fact)
                    # Add to &res1 atomspace
                    metta4Miner.run(f"!(add-atom &res1 {parsed_fact})")
                    loaded_count += 1
                    print(f"✅ Loaded: {fact[:50]}...")
                except Exception as e:
                    print(f"❌ Failed to load: {fact[:50]}... - {e}")
        
        # Check final atomspace contents
        res1_atoms = metta4Miner.run("!(get-atoms &res1)")
        print(f"\n📊 Final &res1 atomspace contains {len(res1_atoms) if res1_atoms else 0} atoms")
        print(f"✅ Successfully loaded {loaded_count} facts into &res1")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data.metta: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_chaining_with_data():
    """Test backward chaining with loaded data"""
    print("\n🧪 Testing backward chaining with loaded data...")
    
    test_queries = [
        '(engagement A_14219 "Low")',
        '(length A_14219 "Short")',
        '(tone A_13014 "Casual")',
        '(reading_time A_13014 "Long")'
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: {query}")
        try:
            result = metta4Miner.run(f"""!(backward-chain &res1 (S (S (S Z))) (: $prf {query}))""")
            print(f"📊 Result: {result}")
        except Exception as e:
            print(f"❌ Error with query {query}: {e}")

if __name__ == "__main__":
    # Load data
    success = load_data_to_res1()
    
    if success:
        # Test backward chaining
        test_backward_chaining_with_data()
        print("\n🎉 Data loading and testing complete!")
    else:
        print("\n❌ Data loading failed!")
