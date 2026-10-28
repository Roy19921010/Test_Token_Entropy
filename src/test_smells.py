"""Static analysis of LLM-generated pytest suites.

Two functions:
    test_structure(code)  → shape metrics (LOC, n_tests, asserts_per_test, ...)
    test_smells(code)     → counts of known test smells from testsmells.org

Both are pure static analysis — never runs the code. Failure to parse returns
zeros for count-based smells and None for ratio-based structure fields, so
consumers don't need to special-case parse failures.

Smell coverage (subset of testsmells.org):
    - empty_test          — body is `pass` or single trivial assert
    - assertion_roulette  — multiple asserts in one test with no messages
    - magic_number        — assertion contains bare numeric literal
    - sleepy_test         — contains time.sleep(...)
    - conditional_test    — test contains if/for/while (should be data-driven instead)
    - duplicate_test      — two tests with identical bodies
    - sensitive_equality  — `assert float == float` without pytest.approx
    - no_assertion        — test function without any assert / pytest.raises call
"""
from __future__ import annotations

import ast
import hashlib
import re
from typing import Any


def _safe_parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_")]


def _asserts_in(func: ast.FunctionDef) -> list[ast.Assert]:
    return [n for n in ast.walk(func) if isinstance(n, ast.Assert)]


def _pytest_raises_in(func: ast.FunctionDef) -> list[ast.With]:
    """Detect `with pytest.raises(...)` blocks (also counts as an assertion)."""
    out = []
    for n in ast.walk(func):
        if not isinstance(n, ast.With):
            continue
        for item in n.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and _is_pytest_raises(call.func):
                out.append(n)
                break
    return out


def _is_pytest_raises(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "raises" and (
            isinstance(node.value, ast.Name) and node.value.id == "pytest"
        )
    if isinstance(node, ast.Name):
        return node.id == "raises"
    return False


def test_structure(code: str) -> dict[str, Any]:
    """Shape metrics — LOC, number of test functions, asserts per test.

    Used both for its own value and as a control variable (longer tests are
    likely to have higher coverage / more chances to fail / etc.)."""
    non_empty_lines = [l for l in code.splitlines() if l.strip()]
    total_lines = code.count("\n") + 1 if code else 0

    tree = _safe_parse(code)
    if tree is None:
        return {
            "generated_loc": len(non_empty_lines),
            "generated_lines_total": total_lines,
            "n_test_functions": 0,
            "n_asserts_total": 0,
            "n_pytest_raises_total": 0,
            "asserts_per_test_mean": None,
            "asserts_per_test_max": None,
            "tests_with_zero_asserts": 0,
            "n_imports": 0,
        }

    tests = _test_functions(tree)
    imports = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom))]

    n_asserts_per_test = [len(_asserts_in(t)) + len(_pytest_raises_in(t))
                          for t in tests]
    return {
        "generated_loc": len(non_empty_lines),
        "generated_lines_total": total_lines,
        "n_test_functions": len(tests),
        "n_asserts_total": sum(len(_asserts_in(t)) for t in tests),
        "n_pytest_raises_total": sum(len(_pytest_raises_in(t)) for t in tests),
        "asserts_per_test_mean": (sum(n_asserts_per_test) / len(n_asserts_per_test)
                                  if n_asserts_per_test else None),
        "asserts_per_test_max": max(n_asserts_per_test) if n_asserts_per_test else None,
        "tests_with_zero_asserts": sum(1 for n in n_asserts_per_test if n == 0),
        "n_imports": len(imports),
    }


# ---- smell detectors ----------------------------------------------------------

def _is_trivial_assert(node: ast.Assert) -> bool:
    """`assert True`, `assert 1`, `assert 'x'` — pass-through asserts."""
    t = node.test
    if isinstance(t, ast.Constant):
        return bool(t.value)  # `assert True` etc.
    return False


def _has_only_pass(func: ast.FunctionDef) -> bool:
    body = [n for n in func.body if not isinstance(n, ast.Expr) or
            not (isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def _contains_control_flow(func: ast.FunctionDef) -> bool:
    """Test contains `if`, `for`, `while`, `try` — usually a code smell."""
    for n in ast.walk(func):
        # Skip the outer function node itself
        if n is func:
            continue
        if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.AsyncFor)):
            return True
    return False


_FLOAT_LITERAL_RE = re.compile(r"^-?\d+\.\d")


def _is_float_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return True
    # crude: `1.0 + x`, `math.sqrt(...)` etc. we can't be sure without types.
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("sqrt", "log", "exp", "sin",
                                                        "cos", "tan", "pow"):
            return True
    return False


def _sensitive_equality(node: ast.Assert) -> bool:
    """`assert x == y` where either side is clearly float."""
    if not isinstance(node.test, ast.Compare):
        return False
    if not any(isinstance(op, ast.Eq) for op in node.test.ops):
        return False
    left = node.test.left
    rights = node.test.comparators
    exprs = [left] + list(rights)
    return any(_is_float_expr(e) for e in exprs)


def _contains_magic_number(node: ast.Assert) -> bool:
    """Bare numeric literal (not 0, 1, -1) inside the assert expression."""
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            if isinstance(n.value, bool):
                continue
            if n.value in (0, 1, -1, 0.0, 1.0, -1.0):
                continue
            return True
    return False


def _has_sleep(func: ast.FunctionDef) -> bool:
    for n in ast.walk(func):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == "sleep":
                return True
            if isinstance(f, ast.Name) and f.id == "sleep":
                return True
    return False


def _body_hash(func: ast.FunctionDef) -> str:
    """Stable hash of the function body, ignoring the name and decorators."""
    try:
        src = ast.unparse(ast.Module(body=func.body, type_ignores=[]))
    except Exception:
        src = repr(func.body)
    return hashlib.sha1(src.encode()).hexdigest()


def test_smells(code: str) -> dict[str, int]:
    """Return counts of test smells across all test_* functions in `code`.

    All counts default to 0 on parse failure. Higher = smellier."""
    smells = {
        "empty_test": 0,
        "no_assertion": 0,
        "assertion_roulette": 0,
        "magic_number": 0,
        "sleepy_test": 0,
        "conditional_test": 0,
        "duplicate_test": 0,
        "sensitive_equality": 0,
        "total": 0,
    }

    tree = _safe_parse(code)
    if tree is None:
        return smells

    tests = _test_functions(tree)
    body_hashes: dict[str, int] = {}

    for t in tests:
        asserts = _asserts_in(t)
        raises = _pytest_raises_in(t)

        # empty_test
        if _has_only_pass(t) or (
            len(asserts) == 1 and not raises and _is_trivial_assert(asserts[0])
        ):
            smells["empty_test"] += 1

        # no_assertion
        if not asserts and not raises:
            smells["no_assertion"] += 1

        # assertion_roulette: >1 assert, none with messages
        if len(asserts) > 1 and all(a.msg is None for a in asserts):
            smells["assertion_roulette"] += 1

        # magic_number: any assert containing a non-trivial numeric literal
        if any(_contains_magic_number(a) for a in asserts):
            smells["magic_number"] += 1

        # sleepy_test
        if _has_sleep(t):
            smells["sleepy_test"] += 1

        # conditional_test
        if _contains_control_flow(t):
            smells["conditional_test"] += 1

        # sensitive_equality
        if any(_sensitive_equality(a) for a in asserts):
            smells["sensitive_equality"] += 1

        h = _body_hash(t)
        body_hashes[h] = body_hashes.get(h, 0) + 1

    # duplicate_test: count each duplicate beyond the first
    smells["duplicate_test"] = sum(c - 1 for c in body_hashes.values() if c > 1)

    smells["total"] = sum(v for k, v in smells.items() if k != "total")
    return smells
