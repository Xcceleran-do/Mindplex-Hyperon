
def parse_sexpr_list(s):
    """
    Parses a string representation of a list of S-expressions into a python list of strings.
    Example: "((A 1) (B 2))" -> ["(A 1)", "(B 2)"]
    """
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    
    results = []
    balance = 0
    current = []
    for char in s:
        if char == '(':
            balance += 1
            current.append(char)
        elif char == ')':
            balance -= 1
            current.append(char)
        elif char.isspace() and balance == 0:
            if current:
                results.append("".join(current))
                current = []
        else:
            current.append(char)
            
    if current:
        results.append("".join(current))
        
    return results

input_str = "((topic var AI) (topic var AI))"
print(f"Input: {input_str}")
parsed = parse_sexpr_list(input_str)
print(f"Parsed: {parsed}")
print(f"Type of elements: {[type(x) for x in parsed]}")
