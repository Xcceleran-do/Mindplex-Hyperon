from hyperon import *
from hyperon.ext import register_atoms
import random
import string
import time
from hyperon.atoms import OperationAtom, V
from hyperon.ext import register_atoms
import itertools
from itertools import combinations
import re


def combine_lists_op(metta, var1, var2):
    input_str1 = str(var1)
    input_str2 = str(var2)

    list1 = parse_metta_structure(input_str1)
    list2 = parse_metta_structure(input_str2)

    combinations = generate_replacement_combinations(list1, list2)
    combined_pattern = " ".join(
        ["({})".format(" ".join(combo)) for combo in combinations]
    )

    combined_pattern_atoms = "(" + combined_pattern + ")"
    atoms = metta.parse_all(combined_pattern_atoms)
    return atoms


def generate_replacement_combinations(list1, list2):
    """Generate combinations by replacing elements in list2 with elements from list1"""
    result = []
    
    # Try replacing each position in list2 with each element from list1
    for element_from_list1 in list1:
        for pos in range(len(list2)):
            new_combo = list2.copy()
            new_combo[pos] = element_from_list1
            # Only add if no duplicates exist in the combination
            if len(set(new_combo)) == len(new_combo):
                result.append(new_combo)
    
    # Try combinations with multiple elements from list1
    for r in range(2, min(len(list1) + 1, len(list2) + 1)):
        for selected_elements in combinations(list1, r):
            for positions in combinations(range(len(list2)), r):
                for perm in itertools.permutations(selected_elements):
                    new_combo = list2.copy()
                    for pos, element in zip(positions, perm):
                        new_combo[pos] = element
                    # Only add if no duplicates exist in the combination
                    if len(set(new_combo)) == len(new_combo):
                        result.append(new_combo)
    
    # Remove duplicates
    seen = set()
    unique_result = []
    for combo in result:
        combo_tuple = tuple(combo)
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            unique_result.append(combo)
    
    return unique_result


def parse_metta_structure(input_str):
    """Convert a string like ($A $B $C) into a flat list ['$A', '$B', '$C']"""
    elements = []
    current = ""
    in_word = False

    for char in input_str:
        if char == "(":
            continue
        elif char == ")":
            if in_word:
                elements.append(current.strip())
                current = ""
                in_word = False
        elif char.isspace():
            if in_word:
                elements.append(current.strip())
                current = ""
                in_word = False
        else:
            current += char
            in_word = True

    if in_word:
        elements.append(current.strip())

    return elements


def generate_random_string(length=1):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def generate_random_var():
    base_name = "R-" + generate_random_string() + str(int(time.time()))
    new_var = V(base_name)
    return [new_var]


# =========================
# Unique combinations (MeTTa)
# =========================
def _extract_top_level_expressions(list_atom_str: str) -> list[str]:
    """Extract top-level S-expressions from a MeTTa list string.

    Example: "((A 1) (B 2) (C 3))" -> ["(A 1)", "(B 2)", "(C 3)"]
    Assumes the input is a single list expression wrapping items.
    """
    s = list_atom_str.strip()
    if not s or s[0] != '(' or s[-1] != ')':
        # Not a list; treat whole as a single item
        return [s]

    items: list[str] = []
    depth = 0
    buf: list[str] | None = None
    for ch in s:
        if ch == '(':
            depth += 1
            if depth == 2:
                buf = ['(']
            elif depth > 2 and buf is not None:
                buf.append('(')
        elif ch == ')':
            if depth == 2 and buf is not None:
                buf.append(')')
                items.append(''.join(buf).strip())
                buf = None
            elif depth > 2 and buf is not None:
                buf.append(')')
            depth -= 1
        else:
            if buf is not None:
                buf.append(ch)
    return items


def _normalize_metta_vars(expr_str: str) -> str:
    """Normalize variables like $x#12345 to $x for display/uniqueness.

    This preserves the base variable name and strips the instance suffix.
    """
    return re.sub(r"\$([A-Za-z_][\w-]*)(?:#[0-9]+)", r"$\1", expr_str)


# =========================
# Combo variable filters (helpers)
# =========================

_VAR_TOKEN_RE = re.compile(r"\$[A-Za-z_][\w-]*(?:#[0-9]+)?")

def _base_var_name(var_token: str) -> str:
    """Strip MeTTa instance suffix from a variable token: $x#123 -> $x.

    If no suffix, returns the token unchanged.
    """
    return re.sub(r"(#\d+)$", "", var_token)


def _extract_vars_from_expr(expr_str: str) -> set[str]:
    """Extract normalized variable names from a MeTTa expression string.

    Example: "(INHERITANCE_LINK $X#1 $Y)" -> {"$X", "$Y"}
    """
    vars_found = _VAR_TOKEN_RE.findall(expr_str)
    return { _base_var_name(v) for v in vars_found }


def _combo_unique_vars(combo: tuple[str, ...]) -> set[str]:
    """Collect the set of normalized variable names across a combo of expr strings."""
    uniq: set[str] = set()
    for expr in combo:
        uniq.update(_extract_vars_from_expr(expr))
    return uniq


def _filter_combos_single_var(combos: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Keep only combos that contain exactly one unique variable across all expressions.

    - Reject combos with 0 variables (no join).
    - Reject combos with >1 variables (multiple joins).
    """
    kept: list[tuple[str, ...]] = []
    for combo in combos:
        vars_in_combo = _combo_unique_vars(combo)
        if len(vars_in_combo) == 1:
            kept.append(combo)
    return kept


# =========================
# Star-join generator (single hub enforced at generation)
# =========================

def _expr_vars(expr: str) -> set[str]:
    return _extract_vars_from_expr(expr)


def _group_clauses_by_var(exprs: list[str]) -> dict[str, list[int]]:
    """Build inverted index: var -> list of indices of exprs containing it."""
    inv: dict[str, list[int]] = {}
    for i, e in enumerate(exprs):
        for v in _expr_vars(e):
            inv.setdefault(v, []).append(i)
    return inv


def _nonhub_mask_setup(exprs: list[str]) -> tuple[dict[str, int], list[set[str]]]:
    """Map variables to dense ids and precompute var sets per expr."""
    var_ids: dict[str, int] = {}
    next_id = 0
    expr_vars: list[set[str]] = []
    for e in exprs:
        vs = _expr_vars(e)
        expr_vars.append(vs)
        for v in vs:
            if v not in var_ids:
                var_ids[v] = next_id
                next_id += 1
    return var_ids, expr_vars


def _build_nonhub_masks_for_hub(hub: str, expr_vars: list[set[str]], var_ids: dict[str, int]) -> list[int]:
    """For each expression, the bitmask of its variables excluding the hub."""
    masks: list[int] = []
    for vs in expr_vars:
        m = 0
        for v in vs:
            if v == hub:
                continue
            m |= 1 << var_ids[v]
        masks.append(m)
    return masks


def _generate_star_join_combos(exprs: list[str], k: int) -> list[tuple[str, ...]]:
    """Generate size-k combos where there exists exactly one hub var present in all clauses,
    and all other vars are local (no second shared var)."""
    if k <= 0 or k > len(exprs):
        return []

    inv = _group_clauses_by_var(exprs)
    var_ids, expr_vars = _nonhub_mask_setup(exprs)

    results: list[tuple[str, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for hub, indices in inv.items():
        # candidate pool: only clauses containing the hub
        pool = [i for i in indices if hub in expr_vars[i]]
        if len(pool) < k:
            continue
        # precompute masks for this hub
        masks = _build_nonhub_masks_for_hub(hub, expr_vars, var_ids)

        # sort pool by nonhub var count (ascending) to improve pruning
        pool.sort(key=lambda i: bin(masks[i]).count("1"))

        choose: list[int] = []

        def backtrack(start: int, used_mask: int):
            if len(choose) == k:
                key = tuple(sorted(choose))
                if key not in seen:
                    seen.add(key)
                    results.append(tuple(exprs[i] for i in choose))
                return
            for idx in range(start, len(pool)):
                i = pool[idx]
                m = masks[i]
                if (m & used_mask) != 0:
                    continue  # would create a second shared variable
                choose.append(i)
                backtrack(idx + 1, used_mask | m)
                choose.pop()

        backtrack(0, 0)

    return results


def unique_combinations_star_metta_op(metta, list_expr_atom, size_atom):
    """Star-join combinations: enforce a single hub variable during generation.

    Inputs:
    - list_expr_atom: MeTTa list of expressions
    - size_atom: Number atom for k

    Output: Expression list of (conjunct (, ...)) items.
    """
    try:
        k = int(str(size_atom))
    except Exception:
        try:
            k = int(size_atom)
        except Exception:
            k = 0

    raw_list_str = str(list_expr_atom)
    item_strs = _extract_top_level_expressions(raw_list_str)

    # Normalize and dedup
    seen: set[str] = set()
    items: list[str] = []
    for it in item_strs:
        norm = _normalize_metta_vars(it)
        if norm not in seen:
            seen.add(norm)
            items.append(norm)

    if k <= 0 or k > len(items):
        return metta.parse_all("()")
    
    print("combination making started in python")
    combos = _generate_star_join_combos(items, k)
    print("combination making ended in python")
    conj_items = ["(conjunct (, {}) )".format(" ".join(combo)) for combo in combos]
    print("formatting making ended in python")
    combined = "(" + " ".join(conj_items) + ")"
    print("join , making ended in python")
    x = metta.parse_all(combined)
    print("parsing making ended in python")
    return x

@register_atoms(pass_metta=True)
def cnj_exp(metta):
    combineLists = OperationAtom(
        "combine_lists",
        lambda var1, var2: combine_lists_op(metta, var1, var2),
        ["Atom", "Atom", "Expression"],
        unwrap=False,
    )
    generateRandomVar = OperationAtom(
        "generateRandomVar", lambda: generate_random_var(), ["Expression"], unwrap=False
    )
    uniqueCombinationsStar = OperationAtom(
        "unique_combinations_star",
        lambda lst, size: unique_combinations_star_metta_op(metta, lst, size),
        ["Atom", "Atom", "Expression"],
        unwrap=False,
    )
    return {
        r"combine_lists": combineLists,
        r"generateRandomVar": generateRandomVar,
        r"unique_combinations_star": uniqueCombinationsStar,
    }
