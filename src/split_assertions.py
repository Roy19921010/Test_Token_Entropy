"""AST-based splitter: rewrite each pytest test function so every assertion
lives in its own test.

Rationale: pytest short-circuits on the first failing assertion within a test,
so multi-assertion tests undercount actual repair success. By splitting each
multi-assertion test into N single-assertion tests (with prior setup lines
duplicated into each), we get per-assertion pass/fail counts from pytest.

Splitting policy:
  - top-level asserts in a test body        → split (duplicate prior setup)
  - asserts inside a for/while loop         → NOT split (semantics depend on iteration)
  - asserts inside an if-block              → NOT split (semantics depend on branch)
  - asserts inside try/except/with          → NOT split (control flow matters)
  - class-based test methods (unittest)     → split; setUp is re-invoked by pytest per test

A test is either fully split (all its asserts are top-level and get separated) OR
kept intact (any assertion is inside a control-flow block). This bimodal policy
is honest: we log per-suite the fraction of asserts that got split vs kept.

Public API:
    split_code(code: str) -> (new_code: str, stats: dict)

stats: {
    "n_tests_input": int,     # tests in original code
    "n_tests_output": int,    # tests after splitting
    "n_asserts_total": int,   # total asserts across all tests
    "n_asserts_split": int,   # asserts that landed in their own test
    "n_asserts_kept": int,    # asserts that stayed grouped (control-flow-embedded)
    "n_tests_unchanged": int, # tests with 0 or 1 top-level asserts (no split needed)
    "n_tests_split": int,     # tests that got broken into >1 new test
    "n_tests_kept_intact": int, # tests with control-flow asserts left as-is
    "parse_error": bool,      # if we couldn't parse the input
}
"""
from __future__ import annotations

import ast
import copy
from typing import Any


def _has_topless_assert(body: list[ast.stmt]) -> bool:
    """Any assert directly in this body (not nested inside for/while/if/try/with)?"""
    return any(isinstance(s, ast.Assert) for s in body)


def _has_embedded_assert(body: list[ast.stmt]) -> bool:
    """Any assert inside a for/while/if/try/with block?"""
    for s in body:
        if isinstance(s, (ast.For, ast.While, ast.If, ast.Try, ast.With,
                          ast.AsyncFor, ast.AsyncWith)):
            for sub in ast.walk(s):
                if isinstance(sub, ast.Assert):
                    return True
    return False


def _split_one_function(fn: ast.FunctionDef | ast.AsyncFunctionDef,
                        stats: dict) -> list[ast.stmt]:
    """Return the replacement statement(s) for one test function.

    If the function is not a test, or if it has embedded asserts (in control
    flow), return [fn] unchanged. If it has top-level asserts only, split into
    N functions, one per top-level assert, with prior setup lines duplicated
    into each split.
    """
    name = fn.name
    if not name.startswith("test"):
        return [fn]

    body = fn.body
    top_asserts = [i for i, s in enumerate(body) if isinstance(s, ast.Assert)]

    # Test with 0 or 1 top-level asserts → no split needed.
    # But still count it and check for embedded asserts.
    total_asserts = sum(1 for s in ast.walk(fn) if isinstance(s, ast.Assert))
    stats["n_asserts_total"] += total_asserts

    if _has_embedded_assert(body):
        # Control-flow embedded — leave as-is; count all its asserts as "kept"
        stats["n_tests_kept_intact"] += 1
        stats["n_asserts_kept"] += total_asserts
        return [fn]

    if len(top_asserts) <= 1:
        stats["n_tests_unchanged"] += 1
        stats["n_asserts_split"] += total_asserts  # trivially "split" — one assert per test already
        return [fn]

    # Multiple top-level asserts → split
    stats["n_tests_split"] += 1
    stats["n_asserts_split"] += len(top_asserts)

    fn_type = type(fn)  # FunctionDef or AsyncFunctionDef
    new_fns: list[ast.stmt] = []
    for split_idx, assert_pos in enumerate(top_asserts):
        # Body for this split: all statements from 0..assert_pos, but with
        # any assert prior to assert_pos removed (to prevent early failure).
        new_body = []
        for i, stmt in enumerate(body):
            if i > assert_pos:
                break
            if i == assert_pos:
                new_body.append(copy.deepcopy(stmt))
            elif isinstance(stmt, ast.Assert):
                # Prior assertion in setup — skip it. This is the whole point:
                # we don't want assertion #1 failing to short-circuit split #2.
                # Downside: if #1 was doing input validation that #2 relies on,
                # #2's behavior may differ. Acceptable per our splitting policy.
                continue
            else:
                new_body.append(copy.deepcopy(stmt))

        new_fn = fn_type(
            name=f"{name}__split{split_idx}",
            args=copy.deepcopy(fn.args),
            body=new_body,
            decorator_list=copy.deepcopy(fn.decorator_list),
            returns=copy.deepcopy(fn.returns) if fn.returns else None,
            type_comment=fn.type_comment,
        )
        # Preserve position for readable error messages
        ast.copy_location(new_fn, fn)
        ast.fix_missing_locations(new_fn)
        new_fns.append(new_fn)
    return new_fns


def _walk_and_split(nodes: list[ast.stmt], stats: dict) -> list[ast.stmt]:
    """Recursively transform a list of top-level or class-body statements."""
    out: list[ast.stmt] = []
    for stmt in nodes:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            replacements = _split_one_function(stmt, stats)
            out.extend(replacements)
        elif isinstance(stmt, ast.ClassDef):
            # Recurse into class body to split test methods
            stmt.body = _walk_and_split(stmt.body, stats)
            out.append(stmt)
        else:
            out.append(stmt)
    return out


def split_code(code: str) -> tuple[str, dict]:
    """Split multi-assertion pytest tests into single-assertion tests.

    Returns (new_code, stats). If the input doesn't parse, returns
    (original_code, {"parse_error": True, ...zeros}).
    """
    stats = {
        "n_tests_input": 0,
        "n_tests_output": 0,
        "n_asserts_total": 0,
        "n_asserts_split": 0,
        "n_asserts_kept": 0,
        "n_tests_unchanged": 0,
        "n_tests_split": 0,
        "n_tests_kept_intact": 0,
        "parse_error": False,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        stats["parse_error"] = True
        return code, stats

    # Count input tests
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("test"):
            stats["n_tests_input"] += 1

    tree.body = _walk_and_split(tree.body, stats)

    # Count output tests
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("test"):
            stats["n_tests_output"] += 1

    new_code = ast.unparse(tree)
    return new_code, stats


if __name__ == "__main__":
    # Smoke test
    sample = '''
import pytest
from solution import add

def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
    assert add(-1, 1) == 0

def test_loop():
    for i in range(3):
        assert add(i, 1) > 0

def test_single():
    assert add(5, 5) == 10

class TestClass:
    def test_method(self):
        x = 1
        assert x == 1
        y = x + 1
        assert y == 2
'''
    new_code, stats = split_code(sample)
    print("=== ORIGINAL ===")
    print(sample)
    print("=== SPLIT ===")
    print(new_code)
    print("=== STATS ===")
    import json as _j
    print(_j.dumps(stats, indent=2))
