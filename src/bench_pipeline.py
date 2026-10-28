"""One (task, model, seed, temperature) → one row of the benchmark table.

Steps:
    1. Stream a test-suite completion for the task, capturing per-token logprobs.
    2. Extract Python code from the completion (strips ```python fences).
    3. Write source + tests into a tmp dir; run pytest --cov against the source.
    4. Return a dict with logprob aggregates + execution metrics.

The task object is dict-shaped and benchmark-agnostic:
    {
        "task_id": "HumanEval/0",
        "prompt": "...instructions for the LLM...",   # what we send
        "source_code": "def solve(...): ...",         # code under test
        "source_module_name": "solution",             # import name in the tmp dir
    }
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from metrics import aggregate_metrics
from mutation import mutation_score
from test_smells import test_structure, test_smells
from token_positions import (
    high_entropy_locations, per_assertion_entropy, position_summary,
)


CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


@dataclass
class GenResult:
    text: str
    entries: list[dict] = field(default_factory=list)
    finish_reason: str | None = None


def stream_generate(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    temperature: float,
    top_logprobs: int,
    seed: int | None,
    max_tokens: int | None = 4096,
    stream_timeout_sec: float = 180.0,
    disable_thinking: bool = False,
) -> GenResult:
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
        stream=True,
        timeout=stream_timeout_sec,
    )
    if seed is not None:
        kwargs["seed"] = seed
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if disable_thinking:
        # Turns off reasoning-mode output on models that support it
        # (\eg GLM-5.2, Qwen3-32B). No-op on non-reasoning models.
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    stream = client.chat.completions.create(**kwargs)
    parts: list[str] = []
    entries: list[dict] = []
    finish_reason = None
    started = time.monotonic()
    for chunk in stream:
        if time.monotonic() - started > stream_timeout_sec:
            finish_reason = "wallclock_timeout"
            try:
                stream.close()
            except Exception:
                pass
            break
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None and getattr(delta, "content", None):
            parts.append(delta.content)
        lp = getattr(choice, "logprobs", None)
        if lp is not None and getattr(lp, "content", None):
            for tok in lp.content:
                entries.append({
                    "token": tok.token,
                    "logprob": tok.logprob,
                    "top_logprobs": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in (tok.top_logprobs or [])
                    ],
                })
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
    return GenResult(text="".join(parts), entries=entries, finish_reason=finish_reason)


def extract_python(text: str) -> str:
    """Prefer the first fenced block; fall back to the full text."""
    m = CODE_FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def run_pytest_with_coverage(
    workdir: Path,
    source_module_name: str,
    test_filename: str = "test_generated.py",
    timeout_sec: int = 60,
) -> dict:
    """Runs `pytest --cov=<module>` in workdir and parses coverage.json."""
    cov_json = workdir / "coverage.json"
    cmd = [
        sys.executable, "-m", "pytest",
        test_filename,
        f"--cov={source_module_name}",
        "--cov-report=json:coverage.json",
        "--cov-branch",
        "-q", "--tb=line", "--no-header",
        "-rfE",  # per-test summary for failures & errors — feeds _classify_errors
        "--disable-warnings",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True,
            timeout=timeout_sec,
        )
        stdout, stderr, returncode, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        returncode, timed_out = -1, True

    passed, failed, errored, collected = _parse_pytest_summary(stdout)
    err_breakdown = _classify_errors(stdout)
    pass_rate = passed / collected if collected > 0 else None

    stmt_cov = branch_cov = None
    if cov_json.exists():
        try:
            cov = json.loads(cov_json.read_text())
            totals = cov.get("totals", {})
            stmt_cov = totals.get("percent_covered")
            covered_br = totals.get("covered_branches")
            total_br = totals.get("num_branches")
            if total_br:
                branch_cov = 100.0 * covered_br / total_br
        except json.JSONDecodeError:
            pass

    return {
        "pytest_returncode": returncode,
        "pytest_timed_out": timed_out,
        "n_collected": collected,
        "n_passed": passed,
        "n_failed": failed,
        "n_errored": errored,
        "pass_rate": pass_rate,
        "statement_coverage": stmt_cov,
        "branch_coverage": branch_cov,
        "n_assertion_failures": err_breakdown["assertion"],
        "n_import_errors": err_breakdown["import"],
        "n_type_errors": err_breakdown["type"],
        "n_name_errors": err_breakdown["name"],
        "n_attribute_errors": err_breakdown["attribute"],
        "n_other_errors": err_breakdown["other"],
        "pytest_stdout_tail": stdout[-2000:],
        "pytest_stderr_tail": stderr[-2000:],
    }


_SUMMARY_TOKEN_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors)\b")


_FAILURE_LINE_RE = re.compile(r"^(FAILED|ERROR)\s+.*?::.*?\s+-\s+(\w+):", re.MULTILINE)


def _classify_errors(stdout: str) -> dict:
    """Bucket pytest FAILED / ERROR lines by exception class.

    Requires `-rfE` in the pytest invocation, which prints one line per failure/error
    like `FAILED test_generated.py::test_foo - AssertionError: ...`.
    """
    buckets = {"assertion": 0, "import": 0, "type": 0, "name": 0,
               "attribute": 0, "other": 0}
    for kind_marker, exc in _FAILURE_LINE_RE.findall(stdout):
        exc_low = exc.lower()
        if "assertion" in exc_low:
            buckets["assertion"] += 1
        elif exc in ("ImportError", "ModuleNotFoundError"):
            buckets["import"] += 1
        elif exc == "TypeError":
            buckets["type"] += 1
        elif exc == "NameError":
            buckets["name"] += 1
        elif exc == "AttributeError":
            buckets["attribute"] += 1
        else:
            buckets["other"] += 1
    return buckets


def _parse_pytest_summary(stdout: str) -> tuple[int, int, int, int]:
    """Scan bottom-up for the first line containing '<N> passed/failed/error(s)'.
    Handles both the '=== 14 passed in 0.63s ===' form and the bare
    '14 passed in 0.63s' form."""
    passed = failed = errored = 0
    for line in reversed(stdout.splitlines()):
        matches = _SUMMARY_TOKEN_RE.findall(line)
        if not matches:
            continue
        for num, kind in matches:
            if kind == "passed":
                passed = int(num)
            elif kind == "failed":
                failed = int(num)
            elif kind.startswith("error"):
                errored = int(num)
        break
    return passed, failed, errored, passed + failed + errored


def evaluate_generation(
    task: dict,
    generated_code: str,
    *,
    timeout_sec: int = 60,
) -> dict:
    """Write source + generated tests to a tmp dir, run pytest+coverage."""
    if not parses(generated_code):
        return {
            "parses": False, "pytest_returncode": None, "pytest_timed_out": False,
            "n_collected": 0, "n_passed": 0, "n_failed": 0, "n_errored": 0,
            "pass_rate": None,
            "statement_coverage": None, "branch_coverage": None,
            "n_assertion_failures": 0, "n_import_errors": 0, "n_type_errors": 0,
            "n_name_errors": 0, "n_attribute_errors": 0, "n_other_errors": 0,
            "pytest_stdout_tail": "", "pytest_stderr_tail": "",
        }

    source_module = task["source_module_name"]
    with tempfile.TemporaryDirectory(prefix="p3bench_") as tmp:
        wd = Path(tmp)
        (wd / f"{source_module}.py").write_text(task["source_code"])
        (wd / "test_generated.py").write_text(generated_code)
        result = run_pytest_with_coverage(wd, source_module, timeout_sec=timeout_sec)
        result["parses"] = True
        return result


def run_one(
    task: dict,
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    seed: int | None,
    top_logprobs: int,
    timeout_sec: int = 60,
    run_mutation: bool = False,
    mutation_max_mutants: int = 30,
    disable_thinking: bool = False,
) -> dict:
    gen = stream_generate(
        client, model, task["prompt"],
        temperature=temperature, top_logprobs=top_logprobs, seed=seed,
        disable_thinking=disable_thinking,
    )
    code = extract_python(gen.text)
    logit_metrics = aggregate_metrics(gen.entries)
    structure_metrics = test_structure(code)
    smell_metrics = test_smells(code)
    pos_summary = position_summary(gen.entries)
    per_assert = per_assertion_entropy(gen.entries)
    hot_tokens = high_entropy_locations(gen.entries, top_k=10)
    exec_metrics = evaluate_generation(task, code, timeout_sec=timeout_sec)

    mut_metrics: dict = {}
    if run_mutation and exec_metrics.get("parses") and exec_metrics.get("n_collected", 0) > 0:
        mut = mutation_score(
            task["source_code"], code, task["source_module_name"],
            max_mutants=mutation_max_mutants,
        )
        mut_metrics = {
            "n_mutants": mut["n_mutants"],
            "n_killed": mut["n_killed"],
            "n_survived": mut["n_survived"],
            "n_timeout": mut["n_timeout"],
            "n_incompetent": mut["n_incompetent"],
            "score": mut["score"],
            "per_op_kill_rate": mut["per_op_kill_rate"],
        }

    return {
        "task_id": task["task_id"],
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "finish_reason": gen.finish_reason,
        **{f"logit__{k}": v for k, v in logit_metrics.items()},
        **{f"struct__{k}": v for k, v in structure_metrics.items()},
        **{f"smell__{k}": v for k, v in smell_metrics.items()},
        **{f"pos__{k}": v for k, v in pos_summary.items()},
        **{f"exec__{k}": v for k, v in exec_metrics.items()},
        **{f"mut__{k}": v for k, v in mut_metrics.items()},
        # RQ2 payload — dict/list, kept in JSONL but excluded from CSV
        "per_assertion_entropy": per_assert,
        "high_entropy_tokens": hot_tokens,
        "generated_text": gen.text,
        "generated_code": code,
        "entries": gen.entries,
    }