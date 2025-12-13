from hyperon import MeTTa
import re

metta = MeTTa()

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

def _normalize_metta_vars(expr_str: str) -> str:
    """Normalize variables like $x#12345 to $x."""
    return re.sub(r"\$([A-Za-z_][\w-]*)(?:#[0-9]+)", r"$\1", expr_str)

def _extract_vars_from_expr(expr_str: str) -> set[str]:
    """Extract normalized variable names from a MeTTa expression string."""
    vars_found = re.findall(r"\$([A-Za-z_][\w-]*)(?:#[0-9]+)?", expr_str)
    return set(f"${v}" for v in vars_found)

def _expr_functor(expr: str) -> str:
    """Extract the leading functor symbol from an expression string."""
    s = expr.strip()
    if s.startswith('('):
        s = s[1:]
    match = re.match(r"([^\s()]+)", s)
    return match.group(1) if match else ""

def _generate_star_join_combos(exprs: list[str], k: int) -> list[tuple[str, ...]]:
    """
    Generate size-k combos where there exists exactly one hub var present in all clauses,
    and all other vars are local.
    """
    if k <= 0 or k > len(exprs):
        return []

    # 1. Index clauses by variable
    var_to_indices = {}
    expr_vars_list = []
    
    for i, expr in enumerate(exprs):
        vs = _extract_vars_from_expr(expr)
        expr_vars_list.append(vs)
        for v in vs:
            var_to_indices.setdefault(v, []).append(i)

    results = []
    seen = set()

    # 2. Iterate over each potential hub variable
    for hub, indices in var_to_indices.items():
        # Filter clauses that contain the hub
        pool = indices
        if len(pool) < k:
            continue
            
        functors = [_expr_functor(exprs[i]) for i in pool]
        
        def is_compatible(idx_list, new_idx):
            new_vars = expr_vars_list[new_idx]
            for existing_idx in idx_list:
                existing_vars = expr_vars_list[existing_idx]
                # Intersection should be exactly {hub}
                intersection = new_vars.intersection(existing_vars)
                if len(intersection) > 1: # hub is already in both
                    return False
            return True

        def backtrack(start_index, current_indices, current_functors):
            if len(current_indices) == k:
                combo = tuple(sorted(exprs[i] for i in current_indices))
                if combo not in seen:
                    seen.add(combo)
                    results.append(combo)
                return

            for i in range(start_index, len(pool)):
                idx = pool[i]
                functor = functors[i]
                
                if functor and functor in current_functors:
                    continue
                
                if not is_compatible(current_indices, idx):
                    continue
                
                new_functors = current_functors | {functor} if functor else current_functors
                backtrack(i + 1, current_indices + [idx], new_functors)

        backtrack(0, [], set())

    return results

def unique_combinations_star(list_expr_atom, size_atom):
    """
    Star-join combinations: enforce a single hub variable during generation.
    """
    try:
        k = int(str(size_atom))
    except Exception:
        k = 0

    item_strs = len(list_expr_atom)

    # Normalize and dedup
    # seen = set()
    # items = []
    # for it in item_strs:
    #     norm = _normalize_metta_vars(it)
    #     if norm not in seen:
    #         seen.add(norm)
    #         items.append(norm)

    # if k <= 0 or k > len(items):
    #     return "()"
    
    combos = _generate_star_join_combos(list_expr_atom, k)
    
    # conj_items = []
    # for combo in combos:
    #     joined = " ".join(combo)
    #     conj_items.append(f"(conjunct (, {joined}))")
        
    # combined = "(" + " ".join(conj_items) + ")"
    return item_strs
