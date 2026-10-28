"""Lightweight mutation testing for the p3 benchmark.

Generates simple AST-level mutants of the source-under-test (numeric/logical
operator swaps, boundary flips, boolean flips) and reports what fraction of
them the LLM-generated test suite kills.

Why not `mutmut`: mutmut 3.x reads config from cwd and is awkward to embed
inline. This module is ~150 lines, deterministic, and produces per-mutant
metadata we can drop into the bench row.

Public API:
    mutation_score(source_code, test_code, source_module_name, ...) -> dict

Returned dict:
    {
        "n_mutants": int,
        "n_killed": int,
        "n_survived": int,
        "n_timeout": int,
        "n_incompetent": int,  # mutants that produced invalid Python
        "score": float | None,  # killed / (killed + survived), None if no mutants
        "per_op_kill_rate": {"BinOp:Add->Sub": 0.6, ...},
    }

A mutant is "killed" if pytest returns a non-zero exit code (any test failed
or errored) when run against the mutated source.
"""
from __future__ import annotations

import ast
import copy
import subprocess
import sys
import tempfile
from pathlib import Path


# ---- mutation operators -------------------------------------------------------

BINOP_SWAPS: dict[type, list[type]] = {
    ast.Add:  [ast.Sub, ast.Mult],
    ast.Sub:  [ast.Add],
    ast.Mult: [ast.Add, ast.Div],
    ast.Div:  [ast.Mult],
    ast.Mod:  [ast.Mult],
    ast.FloorDiv: [ast.Div],
}
COMPARE_SWAPS: dict[type, list[type]] = {
    ast.Lt:    [ast.LtE, ast.Gt],
    ast.LtE:   [ast.Lt, ast.GtE],
    ast.Gt:    [ast.GtE, ast.Lt],
    ast.GtE:   [ast.Gt, ast.LtE],
    ast.Eq:    [ast.NotEq],
    ast.NotEq: [ast.Eq],
}
BOOLOP_SWAPS: dict[type, list[type]] = {
    ast.And: [ast.Or],
    ast.Or:  [ast.And],
}


class MutantCollector(ast.NodeVisitor):
    """Walk the AST and emit (path, new_op_cls) tuples for every swap point."""
    def __init__(self):
        self.mutations: list[tuple[list[int], type, str]] = []

    def visit(self, node):
        # We use a two-pass approach: walk, and record locations relative to root.
        # Simpler: collect (node_id, alt_cls) and rewrite in a separate pass.
        return super().visit(node)


def _find_mutation_sites(tree: ast.AST) -> list[tuple[ast.AST, type, str]]:
    """Return (parent_op_node, replacement_cls, op_label) for each mutation point."""
    sites: list[tuple[ast.AST, type, str]] = []
    for node in ast.walk(tree):
        # BinOp: node.op is a BinOp op instance (Add(), Sub(), ...)
        if isinstance(node, ast.BinOp):
            for alt in BINOP_SWAPS.get(type(node.op), []):
                sites.append((node, alt, f"BinOp:{type(node.op).__name__}->{alt.__name__}"))
        elif isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                for alt in COMPARE_SWAPS.get(type(op), []):
                    sites.append((node, (i, alt),
                                  f"Compare:{type(op).__name__}->{alt.__name__}"))
        elif isinstance(node, ast.BoolOp):
            for alt in BOOLOP_SWAPS.get(type(node.op), []):
                sites.append((node, alt, f"BoolOp:{type(node.op).__name__}->{alt.__name__}"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            sites.append((node, "drop_not", "UnaryOp:Not->drop"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            sites.append((node, not node.value, f"Const:Bool->{not node.value}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            # Numeric constant flip: 0 -> 1, else -> value + 1 (keeps type)
            new_val = 1 if node.value == 0 else node.value + 1
            sites.append((node, ("num", new_val), f"Const:Num->{new_val}"))
    return sites


def _apply_mutation(tree: ast.AST, site_index: int) -> ast.AST | None:
    """Deep-copy `tree` and apply the site_index-th mutation. Returns None
    if the resulting tree can't be unparsed."""
    mutant_tree = copy.deepcopy(tree)
    sites = _find_mutation_sites(mutant_tree)
    if site_index >= len(sites):
        return None
    node, replacement, _ = sites[site_index]
    if isinstance(node, ast.BinOp):
        node.op = replacement()  # type: ignore[operator]
    elif isinstance(node, ast.Compare):
        i, alt_cls = replacement  # type: ignore[misc]
        node.ops[i] = alt_cls()
    elif isinstance(node, ast.BoolOp):
        node.op = replacement()  # type: ignore[operator]
    elif isinstance(node, ast.UnaryOp) and replacement == "drop_not":
        # Replace UnaryOp(Not, x) with x. Handled by returning modified tree
        # via a NodeTransformer pass.
        return _drop_specific_not(mutant_tree, node)
    elif isinstance(node, ast.Constant):
        if isinstance(replacement, tuple) and replacement[0] == "num":
            node.value = replacement[1]
        else:
            node.value = replacement
    return mutant_tree


def _drop_specific_not(tree: ast.AST, target: ast.UnaryOp) -> ast.AST:
    class Dropper(ast.NodeTransformer):
        def visit_UnaryOp(self, n: ast.UnaryOp):
            if n is target:
                return n.operand
            return self.generic_visit(n) or n
    return Dropper().visit(tree)


# ---- runner -------------------------------------------------------------------

def mutation_score(
    source_code: str,
    test_code: str,
    source_module_name: str,
    *,
    max_mutants: int = 30,
    per_mutant_timeout_sec: int = 15,
) -> dict:
    """Generate up to `max_mutants` mutants of source_code, run test_code
    against each. Return kill statistics.

    Note: this is O(max_mutants) pytest subprocess invocations, so it's slow.
    Cap `max_mutants` low for the sweep; raise it for the final numbers.
    """
    try:
        original_tree = ast.parse(source_code)
    except SyntaxError:
        return _empty_result(reason="source_parse_error")

    sites = _find_mutation_sites(original_tree)
    if not sites:
        return _empty_result(reason="no_mutation_sites")

    if len(sites) > max_mutants:
        # Sample deterministically: every k-th site to keep coverage across operators.
        step = len(sites) / max_mutants
        indices = [int(i * step) for i in range(max_mutants)]
    else:
        indices = list(range(len(sites)))

    killed = survived = timeout = incompetent = 0
    per_op: dict[str, list[int]] = {}  # op_label -> [killed, total]

    for idx in indices:
        _, _, op_label = sites[idx]
        per_op.setdefault(op_label, [0, 0])[1] += 1

        mutant_tree = _apply_mutation(original_tree, idx)
        if mutant_tree is None:
            incompetent += 1
            continue
        try:
            mutant_code = ast.unparse(mutant_tree)
        except Exception:
            incompetent += 1
            continue

        outcome = _run_tests_against(
            mutant_code, test_code, source_module_name,
            timeout_sec=per_mutant_timeout_sec,
        )
        if outcome == "killed":
            killed += 1
            per_op[op_label][0] += 1
        elif outcome == "survived":
            survived += 1
        elif outcome == "timeout":
            timeout += 1
        else:
            incompetent += 1

    total_scored = killed + survived
    return {
        "n_mutants": len(indices),
        "n_killed": killed,
        "n_survived": survived,
        "n_timeout": timeout,
        "n_incompetent": incompetent,
        "score": killed / total_scored if total_scored > 0 else None,
        "per_op_kill_rate": {
            k: (v[0] / v[1] if v[1] else None) for k, v in per_op.items()
        },
    }


def _run_tests_against(
    mutant_source: str, test_code: str, source_module_name: str,
    *, timeout_sec: int,
) -> str:
    """Return 'killed' (tests failed), 'survived' (all passed),
    'timeout', or 'incompetent'."""
    with tempfile.TemporaryDirectory(prefix="p3mut_") as tmp:
        wd = Path(tmp)
        (wd / f"{source_module_name}.py").write_text(mutant_source)
        (wd / "test_generated.py").write_text(test_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "test_generated.py",
                 "-q", "--tb=no", "--no-header", "--disable-warnings"],
                cwd=wd, capture_output=True, text=True, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return "timeout"

        if proc.returncode == 0:
            return "survived"
        if proc.returncode == 5:  # no tests collected — the test file itself broke
            return "incompetent"
        return "killed"


def _empty_result(reason: str) -> dict:
    return {
        "n_mutants": 0, "n_killed": 0, "n_survived": 0,
        "n_timeout": 0, "n_incompetent": 0,
        "score": None, "per_op_kill_rate": {}, "reason": reason,
    }
