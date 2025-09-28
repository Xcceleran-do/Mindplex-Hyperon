from hyperon import MeTTa

metta4Miner = MeTTa()

metta4Miner.run("""
    ! (register-module! ../experiments)

    ! (import! &self experiments:pattern-miner:pattern-miner)
    ! (import! &self experiments:utils:common-utils)
    ! (import! &self experiments:frequent-pattern-miner:frequent-pattern-miner)
    ! (import! &db experiments:data:small-ugly)

    !(bind! &dbb (new-space)) ;; create the database
    
    !(bind! &res1 (new-space)) ;; space to hold the miner result
""")

def mine_pattern(numberOfConjunction):
    """this function will mine patterns with the given number of conjunction"""
    answer = metta4Miner.run(""" !(pattern-miner &res1 &db 3 0) """)
    return answer