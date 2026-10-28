"""Authoritative recompute of RQ1 suite-level, RQ2 per-testbody, and RQ2
per-assertion correlation tables + classification tables + iter-1 hit-rate
figure, for both the paper's full-testbody baseline and the right-of-==
denoised variant.

Conventions:
  * Suite-level entropy averages over tokens INSIDE the fenced ```python```
    block only (region != outside_code). This strips chain-of-thought
    preambles emitted by reasoning models such as gpt-oss.
  * Per-testbody granularity = one row per AST ``test_*`` function (matches
    what the Analyzer flags). A function with @pytest.mark.parametrize is
    still ONE testbody row, labeled "fail" iff any of its parametrize cases
    failed.
  * Right-of-``==`` substitution: when an assertion's RHS is a parameter name
    bound by @pytest.mark.parametrize, the RHS token set is replaced by the
    tokens inside the decorator's argvalues list for that parameter. This
    recovers the correctness-commitment signal on parametrized tests.

Outputs:
  paper/rq2_righteq_tables.tex       — all 5 tables + 4-bar hit-rate figure
  paper/logs/recompute_audit.log     — every number traceable to cell/metric/N
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ast as _ast
from metrics import token_entropies
from token_positions import (
    position_map,
    _char_offsets,
    _find_code_fence,
    _line_starts,
    _parametrize_rhs_spans,
)

# Edit this CELLS list for your (LLM, benchmark) matrix. The 4 LLM ids
# below are the specific model names used in the original paper.
CELLS = [
    ("Qwen3-Coder",  "HE", "data/qwen/humaneval",         3),
    ("Qwen3-Coder",  "MP", "data/qwen/mbpp_plus",         2),
    ("Qwen3-Coder",  "CE", "data/qwen/classeval",         3),
    ("gpt-oss",      "HE", "data/gpt_oss/humaneval",      3),
    ("gpt-oss",      "MP", "data/gpt_oss/mbpp_plus",      2),
    ("gpt-oss",      "CE", "data/gpt_oss/classeval",      3),
    ("GLM-5.2",      "HE", "data/glm52/humaneval",        3),
    ("GLM-5.2",      "MP", "data/glm52/mbpp_plus",        2),
    ("GLM-5.2",      "CE", "data/glm52/classeval",        3),
    ("DeepSeek-V4",  "HE", "data/deepseek_v4/humaneval",  2),
    ("DeepSeek-V4",  "MP", "data/deepseek_v4/mbpp_plus",  2),
    ("DeepSeek-V4",  "CE", "data/deepseek_v4/classeval",  4),
]
CELLS_ORDER = [(llm, b) for llm in ("Qwen3-Coder", "gpt-oss", "GLM-5.2", "DeepSeek-V4")
                        for b in ("HE", "MP", "CE")]
METRICS = [
    ("mean_entropy",  r"$H_{\text{mean}}$"),
    ("max_entropy",   r"$H_{\text{max}}$"),
    ("perplexity",    r"$\mathrm{PPL}$"),
    ("mean_top1_gap", r"$g_{\text{top1}}$"),
]

# Match pytest short-summary lines, keeping any [params] suffix.
_FAIL_RE_FULL = re.compile(
    r"^(?:FAILED|ERROR)\s+test_generated\.py::([^\s]+)", re.M
)
# Pattern for extracting the AST function name from a pytest test id.
_PYTEST_ID_PAT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def _iter_jsonl(path: Path):
    text = path.read_text()
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        yield obj
        i = end


# -----------------------------------------------------------------------------
# Pytest short-summary parsers
# -----------------------------------------------------------------------------
def parse_failing_cases(tail: str) -> set[str]:
    """Full pytest test ids that failed, keeping `[params]`.

    Returns e.g. {'test_foo[a-b]', 'test_bar'}.
    """
    out = set()
    for m in _FAIL_RE_FULL.finditer(tail):
        name = m.group(1)
        if "__split" in name:
            # Split-pytest naming: test_foo__split3[params] -> we don't collapse here
            pass
        out.add(name)
    return out


def parse_failing_ast_functions(tail: str) -> set[str]:
    """AST-level failing function names. Collapses `[params]` and `__split`
    suffixes so e.g. 'test_foo[a-b]' and 'test_foo__split0' both map to
    'test_foo'."""
    out = set()
    for m in _FAIL_RE_FULL.finditer(tail):
        name = m.group(1)
        if "__split" in name:
            name = name.split("__split", 1)[0]
        else:
            # Strip [params]
            bracket = name.find("[")
            if bracket > 0:
                name = name[:bracket]
        out.add(name)
    return out


def parse_failing_assertions(tail: str) -> set[tuple[str, int]]:
    """From a SPLIT-pytest tail, return {(base_name, split_idx)} for every
    failing split test. ``test_foo__split3[params]`` -> ('test_foo', 3)."""
    out = set()
    for m in _FAIL_RE_FULL.finditer(tail):
        name = m.group(1)
        if "__split" not in name:
            continue
        base, suffix = name.split("__split", 1)
        suffix = suffix.split("[")[0]  # strip [params]
        try:
            idx = int(suffix)
        except ValueError:
            continue
        out.add((base, idx))
    return out


# -----------------------------------------------------------------------------
# Pytest collected-test id enumeration
# -----------------------------------------------------------------------------
# The pytest tail contains "N passed, M failed" summary and FAILED/ERROR lines,
# but it doesn't list the PASSING test ids by default. To enumerate all
# collected test ids (including parametrize cases) we re-parse the generated
# code: for each AST test_* function with parametrize, we expand its cases.
def enumerate_test_cases(code: str) -> dict[str, list[str]]:
    """Return {ast_function_name: [full_pytest_id, ...]}.

    For a plain test_foo(): [{'test_foo': ['test_foo']}].
    For a parametrize'd function with K cases: list of K ids. We don't need
    the exact [params] string that pytest would synthesize; we just need N
    counts matching pytest's n_collected. So we return
    [f'{name}[case{i}]' for i in range(K)] as a stable alias.
    """
    import ast as _ast
    out = {}
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return out
    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        # Count parametrize cases
        n_cases = 1
        for dec in node.decorator_list:
            # Expect: @pytest.mark.parametrize(..., [list of tuples])
            if not isinstance(dec, _ast.Call):
                continue
            # Resolve decorator name
            name = ""
            if isinstance(dec.func, _ast.Attribute):
                parts = []
                cur = dec.func
                while isinstance(cur, _ast.Attribute):
                    parts.insert(0, cur.attr); cur = cur.value
                if isinstance(cur, _ast.Name):
                    parts.insert(0, cur.id)
                name = ".".join(parts)
            if "parametrize" not in name:
                continue
            # The second positional arg (or 'argvalues' kwarg) is the list of
            # cases. If multiple parametrize decorators are stacked, pytest
            # multiplies the case counts.
            cases_arg = None
            if len(dec.args) >= 2:
                cases_arg = dec.args[1]
            else:
                for kw in dec.keywords:
                    if kw.arg == "argvalues":
                        cases_arg = kw.value
                        break
            k = 1
            if isinstance(cases_arg, (_ast.List, _ast.Tuple)):
                k = max(1, len(cases_arg.elts))
            n_cases *= k
        out[node.name] = [f"{node.name}[case{i}]" for i in range(n_cases)]
    return out


# -----------------------------------------------------------------------------
# Entropy aggregates: full / right-of-== at suite, testbody, assertion
# -----------------------------------------------------------------------------
def _four_metrics(ents, logps, top0s, top1s):
    """Compute 4-metric tuple over provided arrays, length-aligned."""
    if not ents:
        return (None, None, None, None)
    h_mean = sum(ents) / len(ents)
    h_max = max(ents)
    ppl = math.exp(-sum(logps) / len(logps)) if logps else None
    if top0s and top1s and len(top0s) == len(top1s):
        gaps = [math.exp(a) - math.exp(b) for a, b in zip(top0s, top1s)]
        g = sum(gaps) / len(gaps) if gaps else None
    elif top0s:
        gaps = [math.exp(a) for a in top0s]
        g = sum(gaps) / len(gaps) if gaps else None
    else:
        g = None
    return (h_mean, h_max, ppl, g)


def _collect_token_data(entries, keep_mask):
    """For token indices where keep_mask[i]==True, collect (ents, logps, top0s, top1s)."""
    ents = token_entropies(entries)
    kept_ents = []
    kept_logps = []
    kept_top0 = []
    kept_top1 = []
    for i, keep in enumerate(keep_mask):
        if not keep:
            continue
        kept_ents.append(ents[i])
        e = entries[i]
        lp = e.get("logprob")
        if lp is not None:
            kept_logps.append(float(lp))
        top = e.get("top_logprobs") or []
        if len(top) >= 2:
            a = top[0].get("logprob")
            b = top[1].get("logprob")
            if a is not None and b is not None:
                kept_top0.append(float(a))
                kept_top1.append(float(b))
        elif len(top) == 1:
            a = top[0].get("logprob")
            if a is not None:
                kept_top0.append(float(a))
    return kept_ents, kept_logps, kept_top0, kept_top1


def _assertion_rhs_name_bindings(code: str) -> dict:
    """For each (func_name, assertion_idx) whose assertion is of the form
    ``<lhs> == <single Name>``, return the Name identifier. Used to detect
    assertions whose RHS resolves to a @parametrize parameter.

    Returns dict ``{(func_name, aidx): rhs_name}``.
    """
    out: dict[tuple[str, int], str] = {}
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return out
    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        aidx = 0
        for sub in _ast.walk(node):
            if not isinstance(sub, _ast.Assert):
                continue
            test = sub.test
            # Pattern: Compare(left=..., ops=[Eq()], comparators=[Name])
            if (isinstance(test, _ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], _ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], _ast.Name)):
                out[(node.name, aidx)] = test.comparators[0].id
            aidx += 1
    return out


def _parametrize_context(entries):
    """Compute per-row parametrize substitution data.

    Returns a tuple ``(code_start, line_starts, param_spans, assertion_name_bindings)``:
      - code_start:          char offset of the fenced code block within the raw text
      - line_starts:         for the code string, offsets of each code line
      - param_spans:         {(func_name, param_name): [(char_start, char_end), ...]}
                              with char offsets in the CODE string (not raw text)
      - assertion_name_bindings: {(func_name, aidx): rhs_name_identifier}
                              for assertions of the form ``<lhs> == <Name>``
    """
    text, _ = _char_offsets(entries)
    cs, ce = _find_code_fence(text)
    code = text[cs:ce]
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        tree = None
    line_starts = _line_starts(code)
    param_spans = _parametrize_rhs_spans(code, line_starts, tree) if tree is not None else {}
    name_bindings = _assertion_rhs_name_bindings(code)
    return cs, line_starts, param_spans, name_bindings


def _tokens_in_code_spans(entries, code_start, spans) -> list[int]:
    """Return token indices whose char-start falls inside any (s, e) span
    in the CODE string. Converts code-string offsets to raw-text offsets.
    """
    if not spans:
        return []
    # Raw-text spans
    raw_spans = [(code_start + s, code_start + e) for s, e in spans]
    out = []
    _, tok_starts = _char_offsets(entries)
    for i, start in enumerate(tok_starts):
        for s, e in raw_spans:
            if s <= start < e:
                out.append(i); break
    return out


def _right_of_eq_token_mask(entries, positions, scope: str, in_test_filter=None,
                             assertion_filter=None, param_ctx=None):
    """Build a boolean mask over `entries` keeping tokens that are RHS of == in
    each assertion. `scope` is 'suite' | 'testbody' | 'assertion'.

    - For 'suite': keep RHS tokens across every assertion in the suite.
    - For 'testbody' with in_test_filter=NAME: keep RHS tokens of assertions
      inside that AST function.
    - For 'assertion' with (in_test, aidx) filter: keep RHS tokens of exactly
      that assertion. Non-equality assertions fall back to all their tokens.

    When `param_ctx` is provided (output of :func:`_parametrize_context`) and
    an assertion's RHS is a single Name identifier bound by
    ``@pytest.mark.parametrize``, the RHS tokens are REPLACED by the tokens
    inside the decorator's argvalues list for that parameter. This recovers
    the correctness-commitment signal for parametrized tests, where the
    expected value lives in the decorator, not the assertion.
    """
    mask = [False] * len(entries)
    # Group assert-region tokens by (in_test, assertion_idx), preserving order.
    buckets: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for i, p in enumerate(positions):
        if p["region"] == "assert" and p["in_test"] and p["assertion_idx"] is not None:
            buckets[(p["in_test"], p["assertion_idx"])].append((i, entries[i].get("token", "")))

    code_start, _ls, param_spans, name_bindings = (None, None, {}, {})
    if param_ctx is not None:
        code_start, _ls, param_spans, name_bindings = param_ctx

    for key, items in buckets.items():
        in_test, aidx = key
        if scope == "testbody" and in_test_filter is not None and in_test != in_test_filter:
            continue
        if scope == "assertion" and assertion_filter is not None and key != assertion_filter:
            continue
        # Find '==' position
        eq_pos = None
        for j, (_, tok) in enumerate(items):
            if "==" in tok:
                eq_pos = j
                break
        if eq_pos is not None:
            # Check if the RHS is a parametrize-bound Name whose argvalues
            # live in a decorator span.
            rhs_name = name_bindings.get((in_test, aidx))
            substituted = False
            if rhs_name is not None:
                spans = param_spans.get((in_test, rhs_name))
                if spans:
                    for ti in _tokens_in_code_spans(entries, code_start, spans):
                        mask[ti] = True
                    substituted = True
            if not substituted:
                for i, _ in items[eq_pos + 1:]:
                    mask[i] = True
        else:
            # Non-equality: keep all assert tokens for this assertion.
            for i, _ in items:
                mask[i] = True
    return mask


def _full_token_mask(entries, positions, scope: str, in_test_filter=None,
                     assertion_filter=None):
    """Boolean mask for the 'full' variant.

    - 'suite': keep tokens inside the fenced ```python ... ``` block only
      (region != 'outside_code' and != 'unknown'). This strips reasoning
      preamble emitted by chain-of-thought models like gpt-oss. For
      non-reasoning models (Qwen, GLM, DeepSeek) this is a no-op because they
      emit code immediately.
    - 'testbody' with in_test_filter: keep tokens where region in (testbody,
      assert) and in_test == NAME.
    - 'assertion' with filter: keep tokens whose (in_test, assertion_idx)
      matches.
    """
    mask = [False] * len(entries)
    for i, p in enumerate(positions):
        if scope == "suite":
            if p["region"] not in ("outside_code", "unknown"):
                mask[i] = True
        elif scope == "testbody":
            if p["region"] in ("testbody", "assert") and p["in_test"] == in_test_filter:
                mask[i] = True
        elif scope == "assertion":
            key = (p["in_test"], p["assertion_idx"])
            if p["region"] == "assert" and key == assertion_filter:
                mask[i] = True
    return mask


# -----------------------------------------------------------------------------
# Per-suite computation
# -----------------------------------------------------------------------------
def suite_metrics(row, variant: str):
    """Return 4-tuple of (H_mean, H_max, PPL, g_top1) at suite level."""
    entries = row.get("entries") or []
    if not entries:
        return (None, None, None, None)
    positions = position_map(entries)
    if variant == "full":
        mask = _full_token_mask(entries, positions, scope="suite")
    else:
        pctx = _parametrize_context(entries)
        mask = _right_of_eq_token_mask(entries, positions, scope="suite", param_ctx=pctx)
    data = _collect_token_data(entries, mask)
    return _four_metrics(*data)


def testbody_metrics_per_case(row, variant: str) -> list[dict]:
    """Return one dict per pytest-collected test id (including parametrize
    cases). Each dict has:
      {'ast_name': ..., 'full_id': ..., 'metrics': (H_mean, H_max, PPL, g_top1)}
    All cases of an AST function share the same metric values (we don't try
    to attribute entropy per parametrize case because the AST span is one
    function body).
    """
    entries = row.get("entries") or []
    code = row.get("generated_code") or ""
    if not entries or not code:
        return []
    positions = position_map(entries)
    cases = enumerate_test_cases(code)  # ast_name -> [full_ids]
    pctx = None
    out = []
    for ast_name, full_ids in cases.items():
        if variant == "full":
            mask = _full_token_mask(entries, positions, scope="testbody",
                                     in_test_filter=ast_name)
        else:
            if pctx is None:
                pctx = _parametrize_context(entries)
            mask = _right_of_eq_token_mask(entries, positions, scope="testbody",
                                            in_test_filter=ast_name, param_ctx=pctx)
        data = _collect_token_data(entries, mask)
        metrics = _four_metrics(*data)
        if any(m is None for m in metrics[:2]):  # need at least H_mean/H_max
            continue
        # ONE row per AST test_* function (the Analyzer's view). Parametrize
        # cases are NOT expanded; the function is labeled "fail" iff any of
        # its pytest cases failed. ``full_ids`` is kept only to count pytest
        # cases downstream, but the row carries the AST function name.
        out.append({
            "ast_name": ast_name,
            "n_cases": len(full_ids),
            "metrics": metrics,
        })
    return out


def assertion_metrics(row, variant: str) -> list[dict]:
    """One dict per (ast_name, assertion_idx)."""
    entries = row.get("entries") or []
    if not entries:
        return []
    positions = position_map(entries)
    keys = set()
    for p in positions:
        if p["region"] == "assert" and p["in_test"] and p["assertion_idx"] is not None:
            keys.add((p["in_test"], p["assertion_idx"]))
    pctx = None
    out = []
    for key in keys:
        in_test, aidx = key
        if variant == "full":
            mask = _full_token_mask(entries, positions, scope="assertion",
                                     assertion_filter=key)
        else:
            if pctx is None:
                pctx = _parametrize_context(entries)
            mask = _right_of_eq_token_mask(entries, positions, scope="assertion",
                                            assertion_filter=key, param_ctx=pctx)
        data = _collect_token_data(entries, mask)
        metrics = _four_metrics(*data)
        if any(m is None for m in metrics[:2]):
            continue
        out.append({"ast_name": in_test, "assertion_idx": aidx, "metrics": metrics})
    return out


# -----------------------------------------------------------------------------
# Cell-level aggregates
# -----------------------------------------------------------------------------
def _rho(x, y):
    if len(x) < 3 or len(set(y)) < 2:
        return None
    r = spearmanr(x, y).statistic
    return None if r != r else r


def suite_level(dirname: str, audit):
    base_path = ROOT / dirname / "results.jsonl"
    xs_full = {k: [] for k, _ in METRICS}
    xs_re = {k: [] for k, _ in METRICS}
    ys = []
    for row in _iter_jsonl(base_path):
        if not row.get("exec__parses"):
            continue
        nc = row.get("exec__n_collected", 0) or 0
        if nc <= 0:
            continue
        nf = row.get("exec__n_failed", 0) or 0
        y = nf / nc
        mf = suite_metrics(row, "full")
        mr = suite_metrics(row, "re")
        if any(v is None for v in mf) or any(v is None for v in mr):
            continue
        ys.append(y)
        for (mkey, _), vf, vr in zip(METRICS, mf, mr):
            xs_full[mkey].append(vf)
            xs_re[mkey].append(vr)
    out = {"N": len(ys)}
    for mkey, _ in METRICS:
        out[f"full_{mkey}"] = _rho(xs_full[mkey], ys)
        out[f"re_{mkey}"]   = _rho(xs_re[mkey], ys)
    audit(f"[suite] {dirname} N={out['N']}  "
          f"full H_mean={out['full_mean_entropy']}  re H_mean={out['re_mean_entropy']}")
    return out


def per_testbody_level(dirname: str, audit):
    base_path = ROOT / dirname / "results.jsonl"
    collected = {m: {"full": [], "re": []} for m, _ in METRICS}
    cls = {"full": {"auroc": [], "prauc": [], "base": []},
           "re":   {"auroc": [], "prauc": [], "base": []}}
    n_rows_scored = 0
    for row in _iter_jsonl(base_path):
        if not row.get("exec__parses") or row.get("exec__n_collected", 0) <= 0:
            continue
        failing_ast_names = parse_failing_ast_functions(row.get("exec__pytest_stdout_tail", ""))
        for variant in ("full", "re"):
            recs = testbody_metrics_per_case(row, variant)
            if not recs:
                continue
            # ONE row per AST test_* function (matches the Analyzer's view).
            # Label = 1 iff ANY pytest case of this function failed.
            labels = []
            preds_by_metric = {m: [] for m, _ in METRICS}
            for rec in recs:
                label = 1 if rec["ast_name"] in failing_ast_names else 0
                labels.append(label)
                for (mkey, _), v in zip(METRICS, rec["metrics"]):
                    preds_by_metric[mkey].append(v)
            pos = sum(labels)
            if pos == 0 or pos == len(labels) or len(labels) < 2:
                continue
            # Within-suite ρ for each metric
            for mkey, _ in METRICS:
                rho = _rho(preds_by_metric[mkey], labels)
                if rho is not None:
                    collected[mkey][variant].append(rho)
            # AUROC / PR-AUC on H_mean
            preds = preds_by_metric["mean_entropy"]
            try:
                cls[variant]["auroc"].append(roc_auc_score(labels, preds))
            except Exception:
                pass
            try:
                cls[variant]["prauc"].append(average_precision_score(labels, preds))
            except Exception:
                pass
            cls[variant]["base"].append(pos / len(labels))
            if variant == "full":
                n_rows_scored += 1

    out = {
        "n_rows_scored": n_rows_scored,
        "rho":  {m: {k: (mean(v) if v else None) for k, v in d.items()}
                 for m, d in collected.items()},
        "cls":  {v: {k: (mean(vals) if vals else None) for k, vals in d.items()}
                 for v, d in cls.items()},
    }
    audit(f"[per-testbody] {dirname} n_rows_scored={n_rows_scored}  "
          f"full H_mean rho={out['rho']['mean_entropy']['full']}  "
          f"re H_mean rho={out['rho']['mean_entropy']['re']}")
    return out


def per_assertion_level(dirname: str, audit):
    base_path = ROOT / dirname / "results.jsonl"
    split_path = ROOT / dirname / "results_split.jsonl"
    if not split_path.exists():
        return None
    split_rows = {r["task_id"]: r for r in _iter_jsonl(split_path)}
    collected = {m: {"full": [], "re": []} for m, _ in METRICS}
    cls = {"full": {"auroc": [], "prauc": [], "base": []},
           "re":   {"auroc": [], "prauc": [], "base": []}}
    n_scored = 0
    for row in _iter_jsonl(base_path):
        if not row.get("exec__parses"):
            continue
        split = split_rows.get(row["task_id"])
        if split is None:
            continue
        failing = parse_failing_assertions(split.get("exec__pytest_stdout_tail", ""))
        for variant in ("full", "re"):
            recs = assertion_metrics(row, variant)
            if not recs:
                continue
            labels = []
            preds_by_metric = {m: [] for m, _ in METRICS}
            for rec in recs:
                key = (rec["ast_name"], rec["assertion_idx"])
                labels.append(1 if key in failing else 0)
                for (mkey, _), v in zip(METRICS, rec["metrics"]):
                    preds_by_metric[mkey].append(v)
            pos = sum(labels)
            if pos == 0 or pos == len(labels) or len(labels) < 2:
                continue
            for mkey, _ in METRICS:
                rho = _rho(preds_by_metric[mkey], labels)
                if rho is not None:
                    collected[mkey][variant].append(rho)
            preds = preds_by_metric["mean_entropy"]
            try:
                cls[variant]["auroc"].append(roc_auc_score(labels, preds))
            except Exception:
                pass
            try:
                cls[variant]["prauc"].append(average_precision_score(labels, preds))
            except Exception:
                pass
            cls[variant]["base"].append(pos / len(labels))
            if variant == "full":
                n_scored += 1
    out = {
        "n_scored": n_scored,
        "rho":  {m: {k: (mean(v) if v else None) for k, v in d.items()}
                 for m, d in collected.items()},
        "cls":  {v: {k: (mean(vals) if vals else None) for k, vals in d.items()}
                 for v, d in cls.items()},
    }
    audit(f"[per-assertion] {dirname} n_scored={n_scored}  "
          f"full H_mean rho={out['rho']['mean_entropy']['full']}  "
          f"re H_mean rho={out['rho']['mean_entropy']['re']}")
    return out


# -----------------------------------------------------------------------------
# iter-1 hit-rate: full-testbody vs right-of-== (reads analyzer traces)
# -----------------------------------------------------------------------------
def iter1_hit_rate(dirname: str, K: int, variant: str, audit):
    """Compute iter-1 hit rate from the no-guard analyzer traces, on the
    complete failing-baseline cohort.

    variant='full' reads traces_full_K{K}_optout_5iter_nostopos.jsonl
      (full-testbody rank, no-guard, opt-out prompt, no-stop-on-success).
    variant='righteq' reads traces_righteq_K{K}_optout_5iter.jsonl.
    """
    base_path = ROOT / dirname / "results.jsonl"
    if variant == "full":
        trace_path = ROOT / dirname / "traces_noguard_hr" / f"traces_full_K{K}_optout_5iter_nostopos.jsonl"
    else:
        trace_path = ROOT / dirname / "traces_noguard_hr" / f"traces_righteq_K{K}_optout_5iter.jsonl"
    if not trace_path.exists():
        trace_path = None
    if trace_path is None:
        audit(f"[hit-rate] MISSING trace for {dirname} K={K} variant={variant}")
        return None

    # Baseline failing testbodies per task (AST-level names).
    failing_by_task = {}
    collected_by_task = {}
    for r in _iter_jsonl(base_path):
        if not r.get("exec__parses"):
            continue
        if r.get("exec__n_collected", 0) <= 0:
            continue
        failing_by_task[r["task_id"]] = parse_failing_ast_functions(
            r.get("exec__pytest_stdout_tail", ""))
        collected_by_task[r["task_id"]] = r["exec__n_collected"]

    obs = []; nulls = []; ceils = []
    for row in _iter_jsonl(trace_path):
        tid = row["task_id"]
        failing = failing_by_task.get(tid)
        if failing is None:
            continue
        n_fail = len(failing)
        n_tot = collected_by_task[tid]
        if n_fail <= 0 or n_tot <= 0:
            continue
        iter1 = next((it for it in row["iterations"] if it.get("iter") == 1), None)
        if iter1 is None:
            continue
        hot = iter1.get("hot_assertions") or []
        flagged = {h.get("test_name") for h in hot if h.get("test_name")}
        if not flagged:
            continue
        k_used = iter1.get("top_k_used") or len(flagged) or K
        hits = len(flagged & failing)
        obs.append(hits / k_used)
        nulls.append(n_fail / n_tot)
        ceils.append(min(k_used, n_fail) / k_used)
    n = len(obs)
    stats = {
        "n": n,
        "obs":  (mean(obs) * 100) if obs else None,
        "null": (mean(nulls) * 100) if nulls else None,
        "ceil": (mean(ceils) * 100) if ceils else None,
    }
    audit(f"[hit-rate {variant}] {dirname} K={K} N={n}  obs={stats['obs']}  "
          f"null={stats['null']}  ceil={stats['ceil']}")
    return stats


# -----------------------------------------------------------------------------
# LaTeX emission helpers
# -----------------------------------------------------------------------------
def fmt_rho(x, bold=False):
    if x is None:
        return "---"
    s = f"{x:+.3f}"
    return r"$\mathbf{" + s + r"}$" if bold else f"${s}$"


def fmt_auc(x):
    return "---" if x is None else f"{x:.3f}"


def fmt_lift(x, base):
    if x is None or base in (None, 0):
        return "---"
    return f"{x:.3f} ($\\times${x/base:.1f})"


def bold_pair(a, b):
    if a is None or b is None:
        return (fmt_rho(a), fmt_rho(b))
    if abs(a) >= abs(b):
        return (fmt_rho(a, True), fmt_rho(b))
    return (fmt_rho(a), fmt_rho(b, True))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    audit_lines = []
    def audit(msg):
        audit_lines.append(msg)
        print(msg, file=sys.stderr)

    print("Computing suite level ...", file=sys.stderr)
    suite = {(l, b): suite_level(d, audit) for l, b, d, _K in CELLS}

    print("Computing per-testbody ...", file=sys.stderr)
    pt = {(l, b): per_testbody_level(d, audit) for l, b, d, _K in CELLS}

    print("Computing per-assertion ...", file=sys.stderr)
    pa = {(l, b): per_assertion_level(d, audit) for l, b, d, _K in CELLS}

    print("Computing iter-1 hit rate (full and right-eq) ...", file=sys.stderr)
    hit = {}
    for l, b, d, K in CELLS:
        hit[(l, b, "full")] = iter1_hit_rate(d, K, "full", audit)
        hit[(l, b, "re")]   = iter1_hit_rate(d, K, "re", audit)

    out = []
    out.append(r"% Generated by paper/recompute_tables.py — do not edit by hand.")
    out.append(r"% Parametrize correction: each pytest-collected test case (including")
    out.append(r"% @pytest.mark.parametrize cases) contributes its own labeled row")
    out.append(r"% to per-testbody within-suite rho. Entropy values are shared across")
    out.append(r"% all cases of an AST function (because the token span is one body).")
    out.append("")

    # ---------- 1. Suite-level 4-metric table ----------
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering\footnotesize")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\caption{Suite-level cross-suite Spearman $\rho$ between four")
    out.append(r"whole-suite entropy aggregates (two token-selection variants) and")
    out.append(r"the per-suite failure rate. One point per generated suite in each")
    out.append(r"cell. \emph{Full}: aggregate over every token inside the fenced code.")
    out.append(r"\emph{Right-of-\texttt{==}}: aggregate over tokens on the right-hand")
    out.append(r"side of each assertion's \texttt{==}. Bold: the stronger $|\rho|$")
    out.append(r"within each (model, benchmark, metric) triple.}")
    out.append(r"\label{tab:rq1-suite-righteq}")
    out.append(r"\begin{tabular}{llccc|ccc|ccc|ccc}")
    out.append(r"\toprule")
    out.append(r"              & & \multicolumn{3}{c|}{Qwen3-Coder-480B}")
    out.append(r"              & \multicolumn{3}{c|}{gpt-oss-120b}")
    out.append(r"              & \multicolumn{3}{c|}{GLM-5.2}")
    out.append(r"              & \multicolumn{3}{c}{DeepSeek-V4-Flash} \\")
    out.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(l){12-14}")
    out.append(r"Metric & Variant & HE & MP & CE & HE & MP & CE & HE & MP & CE & HE & MP & CE \\")
    out.append(r"\midrule")
    for mi, (mkey, mlabel) in enumerate(METRICS):
        fc, rc = [], []
        for (llm, b) in CELLS_ORDER:
            f = suite[(llm, b)].get(f"full_{mkey}")
            r = suite[(llm, b)].get(f"re_{mkey}")
            a, bb = bold_pair(f, r)
            fc.append(a); rc.append(bb)
        out.append(mlabel + r" & full & " + " & ".join(fc) + r" \\")
        out.append(r" & right-of-\texttt{==} & " + " & ".join(rc) + r" \\")
        if mi < len(METRICS) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    out.append(r"\end{table*}")
    out.append("")

    # ---------- 2. Per-testbody 4-metric table ----------
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering\footnotesize")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\caption{Per-test-body within-suite Spearman $\rho$ between four")
    out.append(r"uncertainty metrics (two token-selection variants) and per-testbody")
    out.append(r"pass/fail. $\rho$ is computed inside each suite over its pytest-collected")
    out.append(r"test cases (parametrize cases of a function each count as separate")
    out.append(r"rows) and then averaged across suites with at least one passing and")
    out.append(r"one failing case. \emph{Full}: tokens inside the AST function body.")
    out.append(r"\emph{Right-of-\texttt{==}}: assertion-RHS tokens only.}")
    out.append(r"\label{tab:rq2-pertest-righteq}")
    out.append(r"\begin{tabular}{llccc|ccc|ccc|ccc}")
    out.append(r"\toprule")
    out.append(r"              & & \multicolumn{3}{c|}{Qwen3-Coder-480B}")
    out.append(r"              & \multicolumn{3}{c|}{gpt-oss-120b}")
    out.append(r"              & \multicolumn{3}{c|}{GLM-5.2}")
    out.append(r"              & \multicolumn{3}{c}{DeepSeek-V4-Flash} \\")
    out.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(l){12-14}")
    out.append(r"Metric & Variant & HE & MP & CE & HE & MP & CE & HE & MP & CE & HE & MP & CE \\")
    out.append(r"\midrule")
    for mi, (mkey, mlabel) in enumerate(METRICS):
        fc, rc = [], []
        for (llm, b) in CELLS_ORDER:
            f = pt[(llm, b)]["rho"][mkey]["full"]
            r = pt[(llm, b)]["rho"][mkey]["re"]
            a, bb = bold_pair(f, r)
            fc.append(a); rc.append(bb)
        out.append(mlabel + r" & full & " + " & ".join(fc) + r" \\")
        out.append(r" & right-of-\texttt{==} & " + " & ".join(rc) + r" \\")
        if mi < len(METRICS) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    out.append(r"\end{table*}")
    out.append("")

    # ---------- 3. Per-testbody AUROC/PR-AUC ----------
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering\footnotesize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\caption{Within-suite classification performance of per-testbody")
    out.append(r"$H_{\text{mean}}$ as a failing-testbody detector.}")
    out.append(r"\label{tab:rq2-pertest-cls-righteq}")
    out.append(r"\begin{tabular}{llrrrr}")
    out.append(r"\toprule")
    out.append(r" & & \multicolumn{2}{c}{Full-testbody} & \multicolumn{2}{c}{Right-of-\texttt{==}} \\")
    out.append(r"\cmidrule(lr){3-4} \cmidrule(l){5-6}")
    out.append(r"Model & Bench & AUROC & PR-AUC ($\times$null) & AUROC & PR-AUC ($\times$null) \\")
    out.append(r"\midrule")
    for i, (llm, b) in enumerate(CELLS_ORDER):
        d = pt[(llm, b)]["cls"]
        au_f = fmt_auc(d["full"]["auroc"]); au_r = fmt_auc(d["re"]["auroc"])
        pr_f = fmt_lift(d["full"]["prauc"], d["full"]["base"])
        pr_r = fmt_lift(d["re"]["prauc"],   d["re"]["base"])
        model = llm if b == "HE" else ""
        out.append(f"{model:12s} & {b} & {au_f} & {pr_f} & {au_r} & {pr_r} \\\\")
        if b == "CE" and i < len(CELLS_ORDER) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    out.append(r"\end{table*}")
    out.append("")

    # ---------- 4. Per-assertion 4-metric table ----------
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering\footnotesize")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\caption{Per-assertion within-suite Spearman $\rho$. $\rho$ is")
    out.append(r"computed inside each suite over its assertions and then averaged")
    out.append(r"across suites with at least one passing and one failing assertion.}")
    out.append(r"\label{tab:rq2-perassert-righteq}")
    out.append(r"\begin{tabular}{llccc|ccc|ccc|ccc}")
    out.append(r"\toprule")
    out.append(r"              & & \multicolumn{3}{c|}{Qwen3-Coder-480B}")
    out.append(r"              & \multicolumn{3}{c|}{gpt-oss-120b}")
    out.append(r"              & \multicolumn{3}{c|}{GLM-5.2}")
    out.append(r"              & \multicolumn{3}{c}{DeepSeek-V4-Flash} \\")
    out.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(l){12-14}")
    out.append(r"Metric & Variant & HE & MP & CE & HE & MP & CE & HE & MP & CE & HE & MP & CE \\")
    out.append(r"\midrule")
    for mi, (mkey, mlabel) in enumerate(METRICS):
        fc, rc = [], []
        for (llm, b) in CELLS_ORDER:
            d = pa[(llm, b)]
            if d is None:
                fc.append("---"); rc.append("---"); continue
            f = d["rho"][mkey]["full"]; r = d["rho"][mkey]["re"]
            a, bb = bold_pair(f, r)
            fc.append(a); rc.append(bb)
        out.append(mlabel + r" & full & " + " & ".join(fc) + r" \\")
        out.append(r" & right-of-\texttt{==} & " + " & ".join(rc) + r" \\")
        if mi < len(METRICS) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    out.append(r"\end{table*}")
    out.append("")

    # ---------- 5. Per-assertion AUROC/PR-AUC ----------
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering\footnotesize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\caption{Within-suite classification performance of per-assertion")
    out.append(r"$H_{\text{mean}}$.}")
    out.append(r"\label{tab:rq2-perassert-cls-righteq}")
    out.append(r"\begin{tabular}{llrrrr}")
    out.append(r"\toprule")
    out.append(r" & & \multicolumn{2}{c}{Full assertion} & \multicolumn{2}{c}{Right-of-\texttt{==}} \\")
    out.append(r"\cmidrule(lr){3-4} \cmidrule(l){5-6}")
    out.append(r"Model & Bench & AUROC & PR-AUC ($\times$null) & AUROC & PR-AUC ($\times$null) \\")
    out.append(r"\midrule")
    for i, (llm, b) in enumerate(CELLS_ORDER):
        d = pa[(llm, b)]
        if d is None:
            au_f = au_r = pr_f = pr_r = "---"
        else:
            c = d["cls"]
            au_f = fmt_auc(c["full"]["auroc"]); au_r = fmt_auc(c["re"]["auroc"])
            pr_f = fmt_lift(c["full"]["prauc"], c["full"]["base"])
            pr_r = fmt_lift(c["re"]["prauc"],   c["re"]["base"])
        model = llm if b == "HE" else ""
        out.append(f"{model:12s} & {b} & {au_f} & {pr_f} & {au_r} & {pr_r} \\\\")
        if b == "CE" and i < len(CELLS_ORDER) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    out.append(r"\end{table*}")
    out.append("")

    # ---------- 6. Hit-rate figure (4 bars per cell: full-obs, re-obs, null, ceil) ----------
    def _coords(variant, key):
        parts = []
        for i, c in enumerate(CELLS_ORDER, start=1):
            h = hit[c + (variant,)]
            v = (h or {}).get(key)
            parts.append(f"({i},{v:.1f})" if v is not None else f"({i},0)")
        return " ".join(parts)
    # Null and ceiling come from the baseline failing cohort, so they're the
    # same under both variants (up to small flakiness) — use the right-eq
    # values as the single plotted series.
    obs_full = _coords("full", "obs")
    obs_re   = _coords("re",   "obs")
    null_c   = _coords("re",   "null")
    ceil_c   = _coords("re",   "ceil")

    out.append(r"% ----- iter-1 hit-rate figure (4-bar: full-obs, re-obs, null, ceiling) -----")
    out.append(r"\begin{figure*}[!tbp]")
    out.append(r"\centering")
    out.append(r"\begin{tikzpicture}")
    out.append(r"\begin{axis}[")
    out.append(r"  width=17.5cm, height=6.0cm,")
    out.append(r"  ybar=1pt,")
    out.append(r"  bar width=4.5pt,")
    out.append(r"  xmin=0.5, xmax=12.5,")
    out.append(r"  ymin=0, ymax=95,")
    out.append(r"  ylabel={iter-1 hit rate (\%)},")
    out.append(r"  xtick={1,2,3,4,5,6,7,8,9,10,11,12},")
    out.append(r"  xticklabels={")
    out.append(r"    Qwen HE, Qwen MP, Qwen CE,")
    out.append(r"    gpt-oss HE, gpt-oss MP, gpt-oss CE,")
    out.append(r"    GLM HE, GLM MP, GLM CE,")
    out.append(r"    DeepSeek HE, DeepSeek MP, DeepSeek CE},")
    out.append(r"  x tick label style={rotate=35, anchor=east, font=\scriptsize},")
    out.append(r"  ymajorgrids, grid style={line width=.2pt, draw=gray!30},")
    out.append(r"  tick label style={font=\scriptsize},")
    out.append(r"  label style={font=\small},")
    out.append(r"  legend style={font=\scriptsize, at={(0.5,1.12)}, anchor=south, legend columns=4,")
    out.append(r"                /tikz/every even column/.append style={column sep=6pt}},")
    out.append(r"  enlarge x limits=0.04,")
    out.append(r"  nodes near coords={\pgfmathprintnumber[fixed,precision=0]{\pgfplotspointmeta}},")
    out.append(r"  every node near coord/.append style={font=\tiny, rotate=90, anchor=west, yshift=1pt, xshift=-2pt},")
    out.append(r"]")
    # Bar order per column (left to right): null, full-testbody obs,
    # right-of-== obs, ceiling. Keep the emit order aligned with the
    # visual order so re-generation preserves the layout the user wants.
    out.append(r"\addplot[fill=none, draw=black, pattern=north east lines, pattern color=gray!60!black]")
    out.append(r"  coordinates {" + null_c + r"};")
    out.append(r"\addlegendentry{hypergeom.\ null}")
    out.append(r"\addplot[fill=blue!35!white, draw=blue!60!black]")
    out.append(r"  coordinates {" + obs_full + r"};")
    out.append(r"\addlegendentry{full-testbody obs}")
    out.append(r"\addplot[fill=blue!70!black, draw=blue!80!black]")
    out.append(r"  coordinates {" + obs_re + r"};")
    out.append(r"\addlegendentry{right-of-\texttt{==} obs}")
    out.append(r"\addplot[fill=green!40!white, draw=green!55!black, dashed]")
    out.append(r"  coordinates {" + ceil_c + r"};")
    out.append(r"\addlegendentry{ceiling $\min(K,n_{\text{fail}})/K$}")
    out.append(r"\end{axis}")
    out.append(r"\end{tikzpicture}")
    out.append(r"\caption{Iteration-1 hit rate per (model, benchmark) cell under full-testbody")
    out.append(r"ranking (light blue) vs.\ right-of-\texttt{==} ranking (dark blue), compared")
    out.append(r"to the hypergeometric null (hatched) and the theoretical ceiling")
    out.append(r"$\overline{\min(K,n_{\text{fail}})/K}$ (dashed green). Both variants are")
    out.append(r"computed on the full no-guard failing-baseline cohort. Null and ceiling")
    out.append(r"depend only on the cohort's failure counts and are shown once per cell")
    out.append(r"(right-eq values).}")
    out.append(r"\label{fig:rq3-signal-quality-righteq}")
    out.append(r"\end{figure*}")
    out.append("")

    # Write tables file
    tex_path = ROOT / "paper" / "rq2_righteq_tables.tex"
    tex_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {tex_path}", file=sys.stderr)

    # Audit log
    audit_path = ROOT / "paper" / "logs" / "recompute_audit.log"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(audit_lines) + "\n")
    print(f"Wrote {audit_path}", file=sys.stderr)

    # Terminal summary
    print("\n=== SUITE SUMMARY (|rho|) ===")
    for mkey, mlabel in METRICS:
        fv = [abs(suite[c][f"full_{mkey}"]) for c in CELLS_ORDER if suite[c][f"full_{mkey}"] is not None]
        rv = [abs(suite[c][f"re_{mkey}"])   for c in CELLS_ORDER if suite[c][f"re_{mkey}"]   is not None]
        wins = sum(1 for c in CELLS_ORDER
                   if suite[c][f"full_{mkey}"] is not None and suite[c][f"re_{mkey}"] is not None
                   and abs(suite[c][f"re_{mkey}"]) > abs(suite[c][f"full_{mkey}"]))
        print(f"  {mlabel:>18s}: full={mean(fv):.3f}  re={mean(rv):.3f}  Δ={mean(rv)-mean(fv):+.3f}  re-wins={wins}/12")

    print("\n=== PER-TESTBODY SUMMARY (|rho|) ===")
    for mkey, mlabel in METRICS:
        fv = [abs(pt[c]["rho"][mkey]["full"]) for c in CELLS_ORDER if pt[c]["rho"][mkey]["full"] is not None]
        rv = [abs(pt[c]["rho"][mkey]["re"])   for c in CELLS_ORDER if pt[c]["rho"][mkey]["re"]   is not None]
        wins = sum(1 for c in CELLS_ORDER
                   if pt[c]["rho"][mkey]["full"] is not None and pt[c]["rho"][mkey]["re"] is not None
                   and abs(pt[c]["rho"][mkey]["re"]) > abs(pt[c]["rho"][mkey]["full"]))
        print(f"  {mlabel:>18s}: full={mean(fv):.3f}  re={mean(rv):.3f}  Δ={mean(rv)-mean(fv):+.3f}  re-wins={wins}/12")

    print("\n=== PER-ASSERTION SUMMARY (|rho|) ===")
    for mkey, mlabel in METRICS:
        cells_ok = [c for c in CELLS_ORDER if pa[c] is not None]
        fv = [abs(pa[c]["rho"][mkey]["full"]) for c in cells_ok if pa[c]["rho"][mkey]["full"] is not None]
        rv = [abs(pa[c]["rho"][mkey]["re"])   for c in cells_ok if pa[c]["rho"][mkey]["re"]   is not None]
        wins = sum(1 for c in cells_ok
                   if pa[c]["rho"][mkey]["full"] is not None and pa[c]["rho"][mkey]["re"] is not None
                   and abs(pa[c]["rho"][mkey]["re"]) > abs(pa[c]["rho"][mkey]["full"]))
        print(f"  {mlabel:>18s}: full={mean(fv):.3f}  re={mean(rv):.3f}  Δ={mean(rv)-mean(fv):+.3f}  re-wins={wins}/12")


if __name__ == "__main__":
    main()
