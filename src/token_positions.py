"""Token → line/col + AST-region mapping for the generated test suites.

Provides positional tags on top of the raw per-token logprobs so the RQ2
Analyzer Agent can answer:
  - Which specific `test_*` function was this token in?
  - Which assertion within that function?
  - Which line and column of the code?

Public API:
    position_map(entries)              per-token tags (line, col, region, in_test, assertion_idx)
    per_assertion_entropy(entries)     entropy aggregated per (test, assertion_idx)
    high_entropy_locations(entries)    top-K uncertain tokens with tags
    position_summary(entries)          scalar rollups suitable for CSV correlation

The generation text can contain prose before/after the fenced ```python``` block.
Tokens whose char-offset falls outside the fence are tagged `region="outside_code"`.
If the fenced code fails to parse, per-token tags fall back to `region="unknown"`.
"""
from __future__ import annotations

import ast
import math
import re
from collections import defaultdict

from metrics import token_entropies

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


CODE_FENCE_OPEN_RE = re.compile(r"```(?:python)?\s*\n")


def _char_offsets(entries: list[dict]) -> tuple[str, list[int]]:
    """Reconstruct full generation text and per-token start offsets."""
    parts: list[str] = []
    starts: list[int] = []
    pos = 0
    for e in entries:
        starts.append(pos)
        tok = e.get("token", "")
        parts.append(tok)
        pos += len(tok)
    return "".join(parts), starts


def _line_starts(text: str) -> list[int]:
    """`starts[i]` = char offset of the start of 1-indexed line i+1."""
    starts = [0]
    for i, c in enumerate(text):
        if c == "\n":
            starts.append(i + 1)
    return starts


def _linecol_to_offset(line_starts: list[int], line: int, col: int) -> int:
    """AST lineno is 1-indexed, col_offset is 0-indexed."""
    return line_starts[line - 1] + col


def _find_code_fence(text: str) -> tuple[int | None, int | None]:
    """Return (start, end) char offsets of the fenced code region in `text`.
    If no fence is present, returns (0, len(text)) so parsing is still attempted.
    """
    m = CODE_FENCE_OPEN_RE.search(text)
    if m is None:
        return 0, len(text)
    start = m.end()
    close = text.find("\n```", start)
    end = close if close != -1 else len(text)
    return start, end


def _line_col_at(text: str, offset: int) -> tuple[int, int]:
    """1-indexed line, 0-indexed col of `text[offset]`."""
    line = text.count("\n", 0, offset) + 1
    prev_nl = text.rfind("\n", 0, offset)
    col = offset - (prev_nl + 1) if prev_nl != -1 else offset
    return line, col


# ---- core: per-token position map ---------------------------------------------

def _parametrize_rhs_spans(code: str, line_starts: list[int], tree) -> dict:
    """For each `test_*` function, inspect its decorators to find
    ``@pytest.mark.parametrize(argnames, argvalues)`` and return, per
    (function_name, parameter_name), the list of ``(char_start, char_end)``
    spans within the argvalues list that correspond to that parameter.

    Returns dict ``{(func_name, param_name): [(s, e), ...]}``. Char offsets
    are in the extracted-code string (not the raw-text string).

    Falls back to empty dict on syntax error or non-standard decorator shape.
    """
    out: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    if tree is None:
        return dict(out)

    def _node_span(n) -> tuple[int, int] | None:
        if getattr(n, "end_lineno", None) is None:
            return None
        s = _linecol_to_offset(line_starts, n.lineno, n.col_offset)
        e = _linecol_to_offset(line_starts, n.end_lineno, n.end_col_offset or 0)
        return (s, e)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            # Resolve decorator dotted-name
            parts: list[str] = []
            cur = dec.func
            while isinstance(cur, ast.Attribute):
                parts.insert(0, cur.attr); cur = cur.value
            if isinstance(cur, ast.Name):
                parts.insert(0, cur.id)
            dname = ".".join(parts)
            if "parametrize" not in dname:
                continue

            # argnames: first positional or `argnames=`
            argnames_node = None
            argvalues_node = None
            if len(dec.args) >= 1:
                argnames_node = dec.args[0]
            if len(dec.args) >= 2:
                argvalues_node = dec.args[1]
            for kw in dec.keywords:
                if kw.arg == "argnames":
                    argnames_node = kw.value
                elif kw.arg == "argvalues":
                    argvalues_node = kw.value

            # Decode argnames (string literal) into a list of parameter names
            param_names: list[str] = []
            if isinstance(argnames_node, ast.Constant) and isinstance(argnames_node.value, str):
                raw = argnames_node.value
                # pytest accepts "a,b" or "a, b" or ["a","b"]
                param_names = [p.strip() for p in raw.replace(" ", "").split(",") if p.strip()]
            elif isinstance(argnames_node, (ast.List, ast.Tuple)):
                for elt in argnames_node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        param_names.append(elt.value)
            if not param_names or argvalues_node is None:
                continue

            # argvalues: expect list/tuple of per-case tuples (or scalars if 1 param)
            if not isinstance(argvalues_node, (ast.List, ast.Tuple)):
                continue
            for case in argvalues_node.elts:
                if len(param_names) == 1:
                    span = _node_span(case)
                    if span is not None:
                        out[(node.name, param_names[0])].append(span)
                else:
                    if not isinstance(case, (ast.List, ast.Tuple)):
                        continue
                    for pname, pval in zip(param_names, case.elts):
                        span = _node_span(pval)
                        if span is not None:
                            out[(node.name, pname)].append(span)

    return dict(out)


def position_map(entries: list[dict]) -> list[dict]:
    """Return one tag-dict per token in `entries`.

    Each dict has:
        token_idx, token, char_start, char_end,
        text_line, text_col       (in the raw generation)
        code_line, code_col       (in the extracted fenced code; None if outside)
        region                    "import" | "assert" | "testbody" | "other"
                                  | "outside_code" | "unknown"
        in_test                   name of enclosing `test_*` function, or None
        assertion_idx             0-indexed assertion within `in_test`, or None
    """
    if not entries:
        return []

    text, tok_starts = _char_offsets(entries)
    code_start, code_end = _find_code_fence(text)
    code = text[code_start:code_end]

    line_starts = _line_starts(code)

    tests: list[tuple[int, int, str]] = []
    asserts: list[tuple[int, int, int, str]] = []
    imports: list[tuple[int, int]] = []

    parse_ok = True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        parse_ok = False
        tree = None

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test_"):
                if node.end_lineno is None:
                    continue
                s = _linecol_to_offset(line_starts, node.lineno, node.col_offset)
                e = _linecol_to_offset(line_starts, node.end_lineno,
                                       node.end_col_offset or 0)
                tests.append((s, e, node.name))
                aidx = 0
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assert) and sub.end_lineno is not None:
                        s2 = _linecol_to_offset(line_starts, sub.lineno, sub.col_offset)
                        e2 = _linecol_to_offset(line_starts, sub.end_lineno,
                                                sub.end_col_offset or 0)
                        asserts.append((s2, e2, aidx, node.name))
                        aidx += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if node.end_lineno is None:
                    continue
                s = _linecol_to_offset(line_starts, node.lineno, node.col_offset)
                e = _linecol_to_offset(line_starts, node.end_lineno,
                                       node.end_col_offset or 0)
                imports.append((s, e))

    result: list[dict] = []
    for i, entry in enumerate(entries):
        tok_start = tok_starts[i]
        tok = entry.get("token", "")
        tok_end = tok_start + len(tok)
        text_line, text_col = _line_col_at(text, tok_start)

        if tok_start < code_start or tok_start >= code_end:
            result.append({
                "token_idx": i, "token": tok,
                "char_start": tok_start, "char_end": tok_end,
                "text_line": text_line, "text_col": text_col,
                "code_line": None, "code_col": None,
                "region": "outside_code", "in_test": None, "assertion_idx": None,
            })
            continue

        offset_in_code = tok_start - code_start
        code_line, code_col = _line_col_at(code, offset_in_code)

        if not parse_ok:
            region = "unknown"
            in_test = None
            assertion_idx = None
        else:
            region = "other"
            in_test = None
            assertion_idx = None
            # tests first (broader), then asserts (narrower) can override
            for s, e, tname in tests:
                if s <= offset_in_code < e:
                    in_test = tname
                    region = "testbody"
                    break
            for s, e, aidx, tname in asserts:
                if s <= offset_in_code < e:
                    region = "assert"
                    assertion_idx = aidx
                    in_test = tname
                    break
            if region == "other":
                for s, e in imports:
                    if s <= offset_in_code < e:
                        region = "import"
                        break

        result.append({
            "token_idx": i, "token": tok,
            "char_start": tok_start, "char_end": tok_end,
            "text_line": text_line, "text_col": text_col,
            "code_line": code_line, "code_col": code_col,
            "region": region, "in_test": in_test, "assertion_idx": assertion_idx,
        })
    return result


# ---- aggregate helpers --------------------------------------------------------

def per_assertion_entropy(entries: list[dict]) -> list[dict]:
    """For each assertion in the code, entropy aggregated over its tokens.

    Returns list sorted by mean_entropy descending (most-uncertain assertion first):
        {test_name, assertion_idx, n_tokens, mean_entropy, max_entropy,
         span_start_line, span_end_line, decoded_text}
    """
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)

    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] == "assert" and p["in_test"] and p["assertion_idx"] is not None:
            buckets[(p["in_test"], p["assertion_idx"])].append(i)

    out: list[dict] = []
    for (tname, aidx), idxs in buckets.items():
        es = [ents[i] for i in idxs]
        first_idx, last_idx = min(idxs), max(idxs)
        decoded = "".join(entries[i].get("token", "") for i in range(first_idx, last_idx + 1))
        out.append({
            "test_name": tname,
            "assertion_idx": aidx,
            "n_tokens": len(idxs),
            "mean_entropy": sum(es) / len(es),
            "max_entropy": max(es),
            "span_start_line": positions[first_idx]["code_line"],
            "span_end_line": positions[last_idx]["code_line"],
            "decoded_text": decoded,
        })
    out.sort(key=lambda x: x["mean_entropy"], reverse=True)
    return out


def per_testbody_entropy(entries: list[dict]) -> list[dict]:
    """For each `test_*` function, entropy aggregated over all tokens inside
    its body (assertions plus any other testbody tokens).

    Returns list sorted by mean_entropy descending. Each record has the same
    shape as `per_assertion_entropy` but with `assertion_idx = None` and
    `decoded_text` being the entire test function's body (may be truncated
    if the caller displays it).
    """
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)

    buckets: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] in ("testbody", "assert") and p["in_test"]:
            buckets[p["in_test"]].append(i)

    out: list[dict] = []
    for tname, idxs in buckets.items():
        es = [ents[i] for i in idxs]
        first_idx, last_idx = min(idxs), max(idxs)
        decoded = "".join(entries[i].get("token", "") for i in range(first_idx, last_idx + 1))
        # Truncate the decoded body to avoid ballooning the prompt.
        if len(decoded) > 400:
            decoded = decoded[:400] + " ..."
        out.append({
            "test_name": tname,
            "assertion_idx": None,
            "n_tokens": len(idxs),
            "mean_entropy": sum(es) / len(es),
            "max_entropy": max(es),
            "span_start_line": positions[first_idx]["code_line"],
            "span_end_line": positions[last_idx]["code_line"],
            "decoded_text": decoded,
        })
    out.sort(key=lambda x: x["mean_entropy"], reverse=True)
    return out


def per_testbody_entropy_right_of_eq(entries: list[dict]) -> list[dict]:
    """For each `test_*` function, entropy aggregated over tokens on the
    RIGHT-HAND SIDE of each assertion's `==`. Motivation: left-of-`==`
    tokens (the function call and inputs the LLM chose to test) carry
    design-choice uncertainty; right-of-`==` tokens (the expected output)
    carry correctness-check uncertainty, which correlates more strongly
    with per-testbody pass/fail.

    For non-equality assertions (`pytest.raises`, bare `assert expr`,
    etc.), we fall back to using all assertion tokens for that assertion.

    Parametrize fix: when an assertion is of the form ``<lhs> == <Name>`` and
    ``<Name>`` is a parameter bound by ``@pytest.mark.parametrize`` on the
    enclosing function, the right-of-``==`` token set is replaced by the
    tokens inside the decorator's argvalues list for that parameter. This
    recovers the correctness-commitment signal for parametrized tests, where
    the expected value literal lives in the decorator, not the assertion.

    Returns list sorted by mean_entropy descending (same shape as
    `per_testbody_entropy`). If a testbody has no assertions or no
    right-of-`==` tokens anywhere, it is skipped.
    """
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)

    # Precompute parametrize argvalue spans and per-assertion RHS-Name
    # bindings. Both operate on the fenced code string, not raw text.
    text, tok_starts = _char_offsets(entries)
    code_start, code_end = _find_code_fence(text)
    code = text[code_start:code_end]
    line_starts = _line_starts(code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    param_spans = _parametrize_rhs_spans(code, line_starts, tree) if tree is not None else {}
    # Per-assertion RHS Name binding: (func, aidx) -> name_id
    rhs_name: dict[tuple[str, int], str] = {}
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            aidx = 0
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assert):
                    continue
                test = sub.test
                if (isinstance(test, ast.Compare)
                        and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq)
                        and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Name)):
                    rhs_name[(node.name, aidx)] = test.comparators[0].id
                aidx += 1

    def _tokens_in_code_spans(spans):
        """Return raw-entries token indices whose char-start falls inside any
        (code_s, code_e) span in the CODE string (offsets are relative to
        the code string, converted to raw-text offsets)."""
        raw_spans = [(code_start + s, code_start + e) for s, e in spans]
        out_idx: list[int] = []
        for i, start in enumerate(tok_starts):
            for s, e in raw_spans:
                if s <= start < e:
                    out_idx.append(i); break
        return out_idx

    # First pass: for each testbody, collect ordered list of (token_idx, region, token_text)
    tb_items: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] in ("testbody", "assert") and p["in_test"]:
            tb_items[p["in_test"]].append((i, p["region"], entries[i].get("token", "")))

    def partition_right_indices(items, fname):
        """Given ordered items for a testbody, return the set of token indices
        that fall right-of-`==` within each assertion span. If the assertion's
        RHS is a parametrize-bound Name, substitute the decorator's argvalues
        tokens for that parameter instead."""
        # Group consecutive 'assert'-region tokens into separate assertions.
        assertions: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        for i, reg, tok in items:
            if reg == "assert":
                cur.append((i, tok))
            else:
                if cur:
                    assertions.append(cur)
                    cur = []
        if cur:
            assertions.append(cur)

        right_idx: list[int] = []
        for aidx, assertion in enumerate(assertions):
            # Check for parametrize substitution first.
            name_id = rhs_name.get((fname, aidx))
            if name_id is not None:
                spans = param_spans.get((fname, name_id))
                if spans:
                    right_idx.extend(_tokens_in_code_spans(spans))
                    continue
            # Find '==' token position within this assertion
            eq_pos = None
            for j, (_, tok) in enumerate(assertion):
                if "==" in tok:
                    eq_pos = j
                    break
            if eq_pos is not None:
                right_idx.extend(i for i, _ in assertion[eq_pos + 1:])
            else:
                # Non-equality assertion: fall back to all its tokens
                right_idx.extend(i for i, _ in assertion)
        return right_idx

    out: list[dict] = []
    for tname, items in tb_items.items():
        kept = partition_right_indices(items, tname)
        if not kept:
            continue
        es = [ents[i] for i in kept]
        first_idx, last_idx = min(kept), max(kept)
        decoded = "".join(entries[i].get("token", "") for i in range(first_idx, last_idx + 1))
        if len(decoded) > 400:
            decoded = decoded[:400] + " ..."
        out.append({
            "test_name": tname,
            "assertion_idx": None,
            "n_tokens": len(kept),
            "mean_entropy": sum(es) / len(es),
            "max_entropy": max(es),
            "span_start_line": positions[first_idx]["code_line"],
            "span_end_line": positions[last_idx]["code_line"],
            "decoded_text": decoded,
        })
    out.sort(key=lambda x: x["mean_entropy"], reverse=True)
    return out


def per_testbody_metrics_right_of_eq(entries: list[dict]) -> list[dict]:
    """Per-testbody full metric vector restricted to right-of-``==`` tokens.

    Like :func:`per_testbody_entropy_right_of_eq` but reports
    ``mean_entropy``, ``max_entropy``, ``perplexity`` (PPL), and
    ``mean_top1_gap`` (g_top1) so RQ2 tables can span the same four
    metrics as the paper's full-testbody table.
    """
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)

    # Collect per-testbody ordered (idx, region, token).
    tb_items: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] in ("testbody", "assert") and p["in_test"]:
            tb_items[p["in_test"]].append((i, p["region"], entries[i].get("token", "")))

    def _partition_right_indices(items):
        assertions: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        for i, reg, tok in items:
            if reg == "assert":
                cur.append((i, tok))
            else:
                if cur:
                    assertions.append(cur)
                    cur = []
        if cur:
            assertions.append(cur)
        right_idx: list[int] = []
        for assertion in assertions:
            eq_pos = None
            for j, (_, tok) in enumerate(assertion):
                if "==" in tok:
                    eq_pos = j
                    break
            if eq_pos is not None:
                right_idx.extend(i for i, _ in assertion[eq_pos + 1:])
            else:
                right_idx.extend(i for i, _ in assertion)
        return right_idx

    out: list[dict] = []
    for tname, items in tb_items.items():
        kept = _partition_right_indices(items)
        if not kept:
            continue
        es = [ents[i] for i in kept]
        chosen_logps: list[float] = []
        top1_gaps: list[float] = []
        for i in kept:
            e = entries[i]
            lp = e.get("logprob")
            if lp is not None:
                chosen_logps.append(float(lp))
            top = e.get("top_logprobs") or []
            if len(top) >= 2:
                lp1 = top[0].get("logprob")
                lp2 = top[1].get("logprob")
                if lp1 is not None and lp2 is not None:
                    top1_gaps.append(math.exp(float(lp1)) - math.exp(float(lp2)))
            elif len(top) == 1:
                lp1 = top[0].get("logprob")
                if lp1 is not None:
                    top1_gaps.append(math.exp(float(lp1)))
        mean_h = sum(es) / len(es)
        max_h = max(es)
        ppl = math.exp(-sum(chosen_logps) / len(chosen_logps)) if chosen_logps else None
        mean_gap = sum(top1_gaps) / len(top1_gaps) if top1_gaps else None
        out.append({
            "test_name": tname,
            "assertion_idx": None,
            "n_tokens": len(kept),
            "mean_entropy": mean_h,
            "max_entropy": max_h,
            "perplexity": ppl,
            "mean_top1_gap": mean_gap,
        })
    return out


def per_testbody_metrics_full(entries: list[dict]) -> list[dict]:
    """Per-testbody full-token (region=testbody/assert) analogue of
    :func:`per_testbody_metrics_right_of_eq` --- all four metrics computed
    over every token inside each ``test_*`` function body (matches the
    denominator used by :func:`per_testbody_entropy`)."""
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] in ("testbody", "assert") and p["in_test"]:
            buckets[p["in_test"]].append(i)
    out: list[dict] = []
    for tname, idxs in buckets.items():
        es = [ents[i] for i in idxs]
        chosen_logps: list[float] = []
        top1_gaps: list[float] = []
        for i in idxs:
            e = entries[i]
            lp = e.get("logprob")
            if lp is not None:
                chosen_logps.append(float(lp))
            top = e.get("top_logprobs") or []
            if len(top) >= 2:
                lp1 = top[0].get("logprob")
                lp2 = top[1].get("logprob")
                if lp1 is not None and lp2 is not None:
                    top1_gaps.append(math.exp(float(lp1)) - math.exp(float(lp2)))
            elif len(top) == 1:
                lp1 = top[0].get("logprob")
                if lp1 is not None:
                    top1_gaps.append(math.exp(float(lp1)))
        out.append({
            "test_name": tname,
            "assertion_idx": None,
            "n_tokens": len(idxs),
            "mean_entropy": sum(es) / len(es),
            "max_entropy": max(es),
            "perplexity": math.exp(-sum(chosen_logps) / len(chosen_logps)) if chosen_logps else None,
            "mean_top1_gap": sum(top1_gaps) / len(top1_gaps) if top1_gaps else None,
        })
    return out


def per_assertion_entropy_right_of_eq(entries: list[dict]) -> list[dict]:
    """Per-assertion entropy restricted to tokens on the RHS of `==`.

    Same denoising principle as :func:`per_testbody_entropy_right_of_eq`
    but one record per assertion instead of one per testbody. Non-equality
    assertions (``pytest.raises``, bare ``assert expr``, ``assert not x``)
    fall back to using all their assert-region tokens.

    Assertions whose RHS slice is empty (e.g., malformed, or ``==`` appears
    as the very last token) are skipped.
    """
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)

    # Group assert-region tokens by (test_name, assertion_idx), preserving
    # order of appearance in the token stream.
    buckets: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] == "assert" and p["in_test"] and p["assertion_idx"] is not None:
            buckets[(p["in_test"], p["assertion_idx"])].append((i, entries[i].get("token", "")))

    out: list[dict] = []
    for (tname, aidx), items in buckets.items():
        eq_pos = None
        for j, (_, tok) in enumerate(items):
            if "==" in tok:
                eq_pos = j
                break
        if eq_pos is not None:
            kept = [i for i, _ in items[eq_pos + 1:]]
        else:
            kept = [i for i, _ in items]
        if not kept:
            continue
        es = [ents[i] for i in kept]
        first_idx, last_idx = min(kept), max(kept)
        decoded = "".join(entries[i].get("token", "") for i in range(first_idx, last_idx + 1))
        out.append({
            "test_name": tname,
            "assertion_idx": aidx,
            "n_tokens": len(kept),
            "mean_entropy": sum(es) / len(es),
            "max_entropy": max(es),
            "span_start_line": positions[first_idx]["code_line"],
            "span_end_line": positions[last_idx]["code_line"],
            "decoded_text": decoded,
        })
    out.sort(key=lambda x: x["mean_entropy"], reverse=True)
    return out


def per_assertion_metrics_right_of_eq(entries: list[dict]) -> list[dict]:
    """Per-assertion 4-metric vector restricted to right-of-``==`` tokens.

    Reports ``mean_entropy``, ``max_entropy``, ``perplexity``, and
    ``mean_top1_gap``. Falls back to all assertion tokens for non-equality
    assertions.
    """
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)
    buckets: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] == "assert" and p["in_test"] and p["assertion_idx"] is not None:
            buckets[(p["in_test"], p["assertion_idx"])].append((i, entries[i].get("token", "")))

    out: list[dict] = []
    for (tname, aidx), items in buckets.items():
        eq_pos = None
        for j, (_, tok) in enumerate(items):
            if "==" in tok:
                eq_pos = j
                break
        if eq_pos is not None:
            kept = [i for i, _ in items[eq_pos + 1:]]
        else:
            kept = [i for i, _ in items]
        if not kept:
            continue
        es = [ents[i] for i in kept]
        chosen_logps: list[float] = []
        top1_gaps: list[float] = []
        for i in kept:
            e = entries[i]
            lp = e.get("logprob")
            if lp is not None:
                chosen_logps.append(float(lp))
            top = e.get("top_logprobs") or []
            if len(top) >= 2:
                lp1 = top[0].get("logprob")
                lp2 = top[1].get("logprob")
                if lp1 is not None and lp2 is not None:
                    top1_gaps.append(math.exp(float(lp1)) - math.exp(float(lp2)))
            elif len(top) == 1:
                lp1 = top[0].get("logprob")
                if lp1 is not None:
                    top1_gaps.append(math.exp(float(lp1)))
        out.append({
            "test_name": tname,
            "assertion_idx": aidx,
            "n_tokens": len(kept),
            "mean_entropy": sum(es) / len(es),
            "max_entropy": max(es),
            "perplexity": math.exp(-sum(chosen_logps) / len(chosen_logps)) if chosen_logps else None,
            "mean_top1_gap": sum(top1_gaps) / len(top1_gaps) if top1_gaps else None,
        })
    return out


def per_assertion_metrics_full(entries: list[dict]) -> list[dict]:
    """Per-assertion 4-metric vector over ALL assert-region tokens (matches
    denominator of :func:`per_assertion_entropy`)."""
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] == "assert" and p["in_test"] and p["assertion_idx"] is not None:
            buckets[(p["in_test"], p["assertion_idx"])].append(i)
    out: list[dict] = []
    for (tname, aidx), idxs in buckets.items():
        es = [ents[i] for i in idxs]
        chosen_logps: list[float] = []
        top1_gaps: list[float] = []
        for i in idxs:
            e = entries[i]
            lp = e.get("logprob")
            if lp is not None:
                chosen_logps.append(float(lp))
            top = e.get("top_logprobs") or []
            if len(top) >= 2:
                lp1 = top[0].get("logprob")
                lp2 = top[1].get("logprob")
                if lp1 is not None and lp2 is not None:
                    top1_gaps.append(math.exp(float(lp1)) - math.exp(float(lp2)))
            elif len(top) == 1:
                lp1 = top[0].get("logprob")
                if lp1 is not None:
                    top1_gaps.append(math.exp(float(lp1)))
        out.append({
            "test_name": tname,
            "assertion_idx": aidx,
            "n_tokens": len(idxs),
            "mean_entropy": sum(es) / len(es),
            "max_entropy": max(es),
            "perplexity": math.exp(-sum(chosen_logps) / len(chosen_logps)) if chosen_logps else None,
            "mean_top1_gap": sum(top1_gaps) / len(top1_gaps) if top1_gaps else None,
        })
    return out


def suite_entropy_right_of_eq(entries: list[dict]) -> dict:
    """Suite-level H_mean / H_max / PPL / g_top1 over tokens on the RHS of
    each assertion's ``==`` (falling back to all assertion tokens for
    non-equality assertions). The denoised analog of the paper's
    ``logit__mean_entropy``/``logit__max_entropy``/etc. applied to the
    whole suite.

    Returns ``None`` for any statistic that cannot be defined (no kept
    tokens, or empty top-logprob list for PPL/g_top1).
    """
    if not entries:
        return {"mean_entropy": None, "max_entropy": None,
                "perplexity": None, "mean_top1_gap": None, "n_tokens": 0}
    ents = token_entropies(entries)
    positions = position_map(entries)

    # Group assert-region tokens by (test_name, assertion_idx), preserving order.
    buckets: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] == "assert" and p["in_test"] and p["assertion_idx"] is not None:
            buckets[(p["in_test"], p["assertion_idx"])].append((i, entries[i].get("token", "")))

    kept: list[int] = []
    for items in buckets.values():
        eq_pos = None
        for j, (_, tok) in enumerate(items):
            if "==" in tok:
                eq_pos = j
                break
        if eq_pos is not None:
            kept.extend(i for i, _ in items[eq_pos + 1:])
        else:
            kept.extend(i for i, _ in items)
    if not kept:
        return {"mean_entropy": None, "max_entropy": None,
                "perplexity": None, "mean_top1_gap": None, "n_tokens": 0}
    es = [ents[i] for i in kept]
    # PPL and top1-gap use chosen logprob + top1 logprob directly.
    chosen_logps: list[float] = []
    top1_gaps: list[float] = []
    for i in kept:
        e = entries[i]
        lp = e.get("logprob")
        if lp is not None:
            chosen_logps.append(float(lp))
        top = e.get("top_logprobs") or []
        if len(top) >= 1 and lp is not None:
            top1 = top[0]
            top1_lp = top1.get("logprob")
            if top1_lp is not None:
                top1_gaps.append(float(lp) - float(top1_lp))
    mean_h = sum(es) / len(es)
    max_h = max(es)
    ppl = None
    if chosen_logps:
        mean_lp = sum(chosen_logps) / len(chosen_logps)
        ppl = math.exp(-mean_lp)
    mean_gap = sum(top1_gaps) / len(top1_gaps) if top1_gaps else None
    return {
        "mean_entropy": mean_h,
        "max_entropy": max_h,
        "perplexity": ppl,
        "mean_top1_gap": mean_gap,
        "n_tokens": len(kept),
    }


def high_entropy_locations(entries: list[dict], top_k: int = 10,
                           threshold: float | None = None) -> list[dict]:
    """Top-K highest-entropy tokens with position tags.

    Small enough to store in the JSONL row for direct RQ2 use. Sorted by
    token order (not entropy) so callers see them in code sequence."""
    if not entries:
        return []
    ents = token_entropies(entries)
    positions = position_map(entries)

    candidates = list(range(len(ents)))
    if threshold is not None:
        candidates = [i for i in candidates if ents[i] >= threshold]
    candidates.sort(key=lambda i: ents[i], reverse=True)
    top = sorted(candidates[:top_k])

    return [
        {
            "token_idx": i,
            "token": positions[i]["token"],
            "entropy": ents[i],
            "logprob": entries[i]["logprob"],
            "code_line": positions[i]["code_line"],
            "code_col": positions[i]["code_col"],
            "region": positions[i]["region"],
            "in_test": positions[i]["in_test"],
            "assertion_idx": positions[i]["assertion_idx"],
        }
        for i in top
    ]


# ---- prominence-based peak detection ------------------------------------------

def prominent_entropy_peaks(entries: list[dict], prominence: float = 0.8,
                            distance: int = 1) -> list[dict]:
    """Find local per-token entropy peaks using scipy's prominence-based peak
    detection. Each returned peak is a single token whose entropy stands out
    from its neighbors by at least `prominence` nats.

    Based on: Peak Detection as described in prior work on perplexity-guided
    code review, adapted here from token perplexity to token entropy (in nats).

    Returns one dict per peak, sorted by prominence descending. Each dict has
    the same fields as `high_entropy_locations` plus:
        "prominence": prominence value from scipy (nats)
        "peak_idx":   the same as token_idx, for convenience
    """
    if not _HAS_SCIPY:
        raise RuntimeError(
            "scipy is required for prominence-based peak detection. "
            "Install with: uv add scipy"
        )
    if not entries:
        return []

    ents = token_entropies(entries)
    if len(ents) < 3:
        return []  # need >=3 points to have neighbors on both sides

    positions = position_map(entries)
    idxs, props = _scipy_find_peaks(
        ents, prominence=prominence, distance=distance,
    )
    prominences = props.get("prominences", [])

    peaks = []
    for peak_idx, prom in zip(idxs, prominences):
        i = int(peak_idx)
        peaks.append({
            "token_idx": i,
            "peak_idx": i,
            "token": positions[i]["token"],
            "entropy": ents[i],
            "logprob": entries[i]["logprob"],
            "prominence": float(prom),
            "code_line": positions[i]["code_line"],
            "code_col": positions[i]["code_col"],
            "region": positions[i]["region"],
            "in_test": positions[i]["in_test"],
            "assertion_idx": positions[i]["assertion_idx"],
        })
    # Sort by prominence descending — hottest peaks first
    peaks.sort(key=lambda p: p["prominence"], reverse=True)
    return peaks


def peaks_grouped_by_assertion(entries: list[dict], prominence: float = 0.8,
                               distance: int = 1) -> list[dict]:
    """Group entropy peaks by containing (test_name, assertion_idx), then
    return one entry per assertion sorted by max-prominence of contained peaks.

    Assertions with no peaks are omitted. Assertions with multiple peaks list
    all of them; ranking key is the max prominence within the assertion.

    Returns list-of-dicts, sorted by max_prominence descending, each with:
        test_name, assertion_idx, span_start_line, span_end_line, decoded_text,
        peaks (list of peak dicts inside this assertion),
        max_prominence, sum_prominence, n_peaks
    """
    peaks = prominent_entropy_peaks(entries, prominence, distance)
    if not peaks:
        return []

    # Build a fresh map from (test_name, assertion_idx) → per_assertion record
    per_assert = per_assertion_entropy(entries)
    key_to_assertion = {(a["test_name"], a["assertion_idx"]): a for a in per_assert}

    # Group peaks by their containing assertion
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for p in peaks:
        if p["in_test"] is None or p["assertion_idx"] is None:
            continue
        buckets[(p["in_test"], p["assertion_idx"])].append(p)

    grouped = []
    for key, peak_list in buckets.items():
        base = key_to_assertion.get(key)
        if base is None:
            # Peak is in an assertion we couldn't tag — skip
            continue
        peak_list_sorted = sorted(peak_list, key=lambda p: p["prominence"],
                                  reverse=True)
        grouped.append({
            "test_name": base["test_name"],
            "assertion_idx": base["assertion_idx"],
            "span_start_line": base["span_start_line"],
            "span_end_line": base["span_end_line"],
            "decoded_text": base["decoded_text"],
            "mean_entropy": base["mean_entropy"],
            "max_entropy": base["max_entropy"],
            "peaks": peak_list_sorted,
            "n_peaks": len(peak_list_sorted),
            "max_prominence": max(p["prominence"] for p in peak_list_sorted),
            "sum_prominence": sum(p["prominence"] for p in peak_list_sorted),
        })
    grouped.sort(key=lambda a: a["max_prominence"], reverse=True)
    return grouped


# ---- scalar summaries (correlation-friendly) ----------------------------------

def position_summary(entries: list[dict], high_thresh: float = 2.0) -> dict:
    """Row-level scalars derived from per-assertion aggregation.

    New signals suitable for the correlation table:
        max_assertion_mean_entropy       hottest assertion's mean entropy
        max_assertion_max_entropy        hottest assertion's peak entropy
        n_assertions_tagged              how many assertions we could position-tag
        n_high_entropy_assertions        assertions whose mean-entropy > high_thresh
        frac_high_entropy_assertions     same as above, normalized
    """
    per_assert = per_assertion_entropy(entries)
    if not per_assert:
        return {
            "max_assertion_mean_entropy": None,
            "max_assertion_max_entropy": None,
            "n_assertions_tagged": 0,
            "n_high_entropy_assertions": 0,
            "frac_high_entropy_assertions": None,
        }
    means = [a["mean_entropy"] for a in per_assert]
    maxes = [a["max_entropy"] for a in per_assert]
    n_hot = sum(1 for m in means if m > high_thresh)
    return {
        "max_assertion_mean_entropy": max(means),
        "max_assertion_max_entropy": max(maxes),
        "n_assertions_tagged": len(per_assert),
        "n_high_entropy_assertions": n_hot,
        "frac_high_entropy_assertions": n_hot / len(per_assert),
    }
