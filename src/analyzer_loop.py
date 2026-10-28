"""RQ2: Entropy-aware Analyzer Agent that repairs failing test suites.

Given a completed benchmark row (from `outputs_bench/results.jsonl`) whose
test suite had failures, this module regenerates the suite with a targeted
prompt that surfaces the most-uncertain assertion identified during the
original generation. Optionally iterates.

Two prompt modes (for ablation — the paper will report both):
    entropy    the analyzer prompt names the specific uncertain assertion
               and asks the model to re-derive it
    naive      the analyzer prompt shows the failure log only, no entropy info

Both modes see the pytest failure log for the current suite. The difference
is only whether the entropy-flagged assertion is highlighted.

Public API:
    repair_row(row, task, ...) -> RepairTrace

Where RepairTrace is a dict:
    {
      "task_id": ...,
      "mode": "entropy" | "naive",
      "n_iterations": int,
      "iterations": [
         {"iter": 0, "code": ..., "n_passed": ..., "n_failed": ...,
          "pass_rate": ..., "entries": [...], "hot_targets": [...]},
         ...
      ],
      "baseline_pass_rate": ...,   # iteration 0 = the original row
      "final_pass_rate": ...,      # last iteration
      "improved": bool,
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from bench_pipeline import (
    stream_generate, extract_python, evaluate_generation,
)
from llm_loader import TOP_LOGPROBS, MODELS, client
from token_positions import (
    per_assertion_entropy, per_testbody_entropy,
    per_testbody_entropy_right_of_eq,
    high_entropy_locations, peaks_grouped_by_assertion,
)


DEFAULT_MODEL = MODELS[0]


# ---- prompt builders ---------------------------------------------------------

ANALYZER_ENTROPY_PROMPT = """\
You previously generated a pytest test suite for the module `{module}`, but some
tests failed against the correct reference implementation. During generation, the
model was measurably UNCERTAIN about the following assertion(s):

{hot_callouts}

The reference module (source of truth) is:
```python
{source}
```

The generated test file was:
```python
{previous_code}
```

The pytest run reported these failures:
```
{failure_log}
```

Re-derive the flagged assertion from first principles against the reference module.
Fix any other failing assertions the same way. Return the complete corrected pytest
file inside a single ```python fenced block. Do not add or remove tests unless a
test is fundamentally malformed.
"""


ANALYZER_ENTROPY_ONLY_PROMPT = """\
You previously generated a pytest test suite for the module `{module}`. During
generation, the model was measurably UNCERTAIN about the following assertion(s),
which suggests they may be incorrect:

{hot_callouts}

The reference module (source of truth) is:
```python
{source}
```

The generated test file was:
```python
{previous_code}
```

No test execution feedback is available. Re-derive the flagged assertion from
first principles against the reference module. If you spot other clearly-wrong
assertions, fix them too. Return the complete corrected pytest file inside a
single ```python fenced block. Do not add or remove tests unless a test is
fundamentally malformed.
"""


# Strict variant of entropy_only used in the pointer-vs-signal ablation
# (Section 5.2 random-K). Removes the "fix other wrong assertions too" clause
# so the Analyzer is instructed to modify ONLY the flagged assertions. This
# isolates the entropy-targeting contribution from the free-clause auditing
# effect. Prompt otherwise identical to ANALYZER_ENTROPY_ONLY_PROMPT.
ANALYZER_ENTROPY_ONLY_STRICT_PROMPT = """\
You previously generated a pytest test suite for the module `{module}`. During
generation, the model was measurably UNCERTAIN about the following assertion(s),
which suggests they may be incorrect:

{hot_callouts}

The reference module (source of truth) is:
```python
{source}
```

The generated test file was:
```python
{previous_code}
```

No test execution feedback is available. Re-derive ONLY the flagged
assertion(s) from first principles against the reference module. Do NOT
modify any other assertion in the file, and do not add or remove tests.
Return the complete pytest file inside a single ```python fenced block, with
the flagged assertion(s) corrected and every other assertion preserved
verbatim.
"""


# Strict-mode variant for TESTBODY-level ranking: the entropy pointer names
# whole test functions rather than individual assertions. The Analyzer is
# instructed to modify only the assertions inside the flagged test(s), leaving
# all other tests' assertions untouched. Used to compare test-body pointer
# targeting against assertion-level pointer targeting under the same strict
# no-free-clause discipline.
ANALYZER_ENTROPY_ONLY_STRICT_TESTBODY_PROMPT = """\
You previously generated a pytest test suite for the module `{module}`. During
generation, the model was measurably UNCERTAIN about the following test
function(s), which suggests one or more of their assertions may be incorrect:

{hot_callouts}

The reference module (source of truth) is:
```python
{source}
```

The generated test file was:
```python
{previous_code}
```

No test execution feedback is available. Re-derive the assertions inside the
flagged test(s) from first principles against the reference module. Do NOT
modify any assertion in any OTHER test function in the file, and do not add or
remove tests. Return the complete pytest file inside a single ```python fenced
block, with the flagged tests' assertions corrected and every unflagged test
preserved verbatim.
"""


# Opt-out ablation variants: the entropy pointer is a suggestion, not a
# directive. The Analyzer inspects each flagged assertion/test and only
# re-derives it if it actively believes the current form is wrong. Compared to
# the strict variants above, this lets the LLM's own judgment gate whether
# re-derivation happens at all — closer to how a real deployment would use the
# framework, but complicates the isolation of the ranker's targeting signal
# (the effective K becomes "K minus whatever the LLM chose to skip").
ANALYZER_ENTROPY_ONLY_STRICT_OPTOUT_PROMPT = """\
You previously generated a pytest test suite for the module `{module}`. During
generation, the model was measurably UNCERTAIN about the following assertion(s),
which suggests they may be incorrect:

{hot_callouts}

The reference module (source of truth) is:
```python
{source}
```

The generated test file was:
```python
{previous_code}
```

No test execution feedback is available. Inspect each flagged assertion
against the reference module. If you believe a flagged assertion is already
correct as written, leave it unchanged. Only re-derive assertions you believe
are incorrect. Do NOT modify any other assertion in the file, and do not add
or remove tests. Return the complete pytest file inside a single ```python
fenced block, with any incorrect flagged assertion(s) corrected and every
other assertion preserved verbatim.
"""


ANALYZER_ENTROPY_ONLY_STRICT_TESTBODY_OPTOUT_PROMPT = """\
You previously generated a pytest test suite for the module `{module}`. During
generation, the model was measurably UNCERTAIN about the following test
function(s), which suggests one or more of their assertions may be incorrect:

{hot_callouts}

The reference module (source of truth) is:
```python
{source}
```

The generated test file was:
```python
{previous_code}
```

No test execution feedback is available. Inspect the assertions inside the
flagged test(s) against the reference module. If you believe an assertion is
already correct as written, leave it unchanged. Only re-derive assertions you
believe are incorrect. Do NOT modify any assertion in any OTHER test function
in the file, and do not add or remove tests. Return the complete pytest file
inside a single ```python fenced block, with any incorrect assertions in
flagged tests corrected and every unflagged test preserved verbatim.
"""


ANALYZER_NAIVE_PROMPT = """\
You previously generated a pytest test suite for the module `{module}`, but some
tests failed against the correct reference implementation.

The reference module (source of truth) is:
```python
{source}
```

The generated test file was:
```python
{previous_code}
```

The pytest run reported these failures:
```
{failure_log}
```

Fix the failing assertions by re-deriving their expected values from the reference
module. Return the complete corrected pytest file inside a single ```python fenced
block. Do not add or remove tests unless a test is fundamentally malformed.
"""


def _format_hot_callouts(hot_regions: list[dict]) -> str:
    """Render a list of hot-region callouts for the entropy prompt.

    Each region is either an assertion (assertion_idx is set) or a whole
    test body (assertion_idx is None). The callout labels adapt accordingly.
    """
    blocks = []
    for i, ha in enumerate(hot_regions, start=1):
        if ha.get("assertion_idx") is not None:
            locator = (
                f"     Assertion:   #{ha['assertion_idx']} at line "
                f"{ha['span_start_line']}"
            )
        else:
            locator = (
                f"     Test body:   lines "
                f"{ha['span_start_line']}--{ha['span_end_line']}"
            )
        blocks.append(
            f"  {i}. Test:        {ha['test_name']}\n"
            f"{locator}\n"
            f"     Excerpt:     {ha['decoded_text'].strip()}\n"
            f"     Uncertainty: mean token entropy = {ha['mean_entropy']:.2f}, "
            f"peak = {ha['max_entropy']:.2f}"
        )
    return "\n\n".join(blocks)


def _build_prompt(mode: str, task: dict, previous_code: str,
                  failure_log: str, hot_assertions: list[dict]) -> str:
    """Build the Analyzer prompt for one of four modes:

    - "naive": failure log only, no entropy pointer.
    - "entropy": failure log + entropy pointer (both). Superset of naive.
    - "entropy_only": entropy pointer only, no failure log. Tests whether
      entropy can drive repair without execution feedback (i.e. pre-emptive).
    - "entropy_only_strict": same as entropy_only but with the "fix any
      other clearly-wrong assertions" clause removed, so the Analyzer is
      told to modify only the flagged assertions. Used to isolate the
      entropy-targeting contribution from the free-clause auditing effect.
    - "entropy_only_strict_testbody": strict variant for testbody-level
      ranking. The Analyzer is told to modify assertions only inside the
      flagged test function(s), not inside any other test. Used to compare
      test-body pointer targeting against assertion-level under the same
      strict discipline.

    `hot_assertions` is a list of the top-K hottest assertions (may be empty
    if row has no tagged assertions).
    """
    module = task["source_module_name"]
    source = task["source_code"]

    if mode == "naive" or not hot_assertions:
        return ANALYZER_NAIVE_PROMPT.format(
            module=module, source=source,
            previous_code=previous_code, failure_log=failure_log,
        )

    hot_callouts = _format_hot_callouts(hot_assertions)

    if mode == "entropy_only":
        return ANALYZER_ENTROPY_ONLY_PROMPT.format(
            module=module, source=source, previous_code=previous_code,
            hot_callouts=hot_callouts,
        )

    if mode == "entropy_only_strict":
        return ANALYZER_ENTROPY_ONLY_STRICT_PROMPT.format(
            module=module, source=source, previous_code=previous_code,
            hot_callouts=hot_callouts,
        )

    if mode == "entropy_only_strict_testbody":
        return ANALYZER_ENTROPY_ONLY_STRICT_TESTBODY_PROMPT.format(
            module=module, source=source, previous_code=previous_code,
            hot_callouts=hot_callouts,
        )

    if mode == "entropy_only_strict_optout":
        return ANALYZER_ENTROPY_ONLY_STRICT_OPTOUT_PROMPT.format(
            module=module, source=source, previous_code=previous_code,
            hot_callouts=hot_callouts,
        )

    if mode == "entropy_only_strict_testbody_optout":
        return ANALYZER_ENTROPY_ONLY_STRICT_TESTBODY_OPTOUT_PROMPT.format(
            module=module, source=source, previous_code=previous_code,
            hot_callouts=hot_callouts,
        )

    # default: "entropy" — everything
    return ANALYZER_ENTROPY_PROMPT.format(
        module=module, source=source, previous_code=previous_code,
        failure_log=failure_log, hot_callouts=hot_callouts,
    )


# ---- one repair iteration ----------------------------------------------------

def _current_state(row: dict, task: dict) -> dict:
    """Bundle the fields the analyzer needs from a benchmark row."""
    return {
        "code": row.get("generated_code", ""),
        "entries": row.get("entries", []),
        "n_passed": row.get("exec__n_passed", 0),
        "n_failed": row.get("exec__n_failed", 0),
        "n_collected": row.get("exec__n_collected", 0),
        "pass_rate": row.get("exec__pass_rate"),
        "failure_log": row.get("exec__pytest_stdout_tail", "")
                       + "\n"
                       + row.get("exec__pytest_stderr_tail", ""),
    }


def _hot_assertions_from(entries: list[dict], top_k: int = 1,
                         rank: str = "max_entropy",
                         prominence: float = 0.8,
                         task_id: str | None = None) -> list[dict]:
    """Return the top-K assertion callouts, sorted hottest first.

    `rank`:
      - "mean_entropy": rank by per-assertion mean token entropy (default).
      - "max_entropy":  rank by per-assertion peak token entropy.
      - "prominence_in_assert": run scipy prominence-based peak detection over
        the whole token stream, keep only peaks that fall inside AST asserts,
        group by assertion, rank by max prominence within each assertion. If
        no prominence-based peaks land in any assertion for this row, fall
        back to mean_entropy ranking so the Analyzer always gets some
        callout when at least one assertion exists.
      - "testbody_mean_entropy": rank whole test functions by their body
        mean token entropy. Coarser targeting than assertion-level ranking;
        useful when uncertainty is spread across the test body rather than
        concentrated at assertions (e.g., on general-purpose LLMs).
      - "testbody_max_entropy":  same but ranked by peak token entropy
        within each test body.
      - "random": pick K assertions uniformly at random. Baseline for the
        reviewer question "does entropy targeting beat pointing at any K
        assertions?" Seeded per-task from task_id so the same task always
        yields the same random pick across sweeps.
      - "random_test": pick K whole test functions uniformly at random from
        per_testbody_entropy. Testbody-level analog of "random" — the null
        baseline for testbody_max/testbody_mean ranking. Seeded per-task.
    """
    top_k = max(1, top_k)

    if rank == "random":
        import random as _rnd
        all_assertions = per_assertion_entropy(entries)
        if not all_assertions:
            return []
        seed_src = task_id if task_id is not None else "unseeded"
        rng = _rnd.Random(hash(("random_rank", seed_src)) & 0xFFFFFFFF)
        picked = rng.sample(all_assertions, min(top_k, len(all_assertions)))
        return picked

    if rank == "random_test":
        import random as _rnd
        all_tests = per_testbody_entropy(entries)
        if not all_tests:
            return []
        seed_src = task_id if task_id is not None else "unseeded"
        # Use a different tuple seed than "random" so we don't get the same
        # picks if a task happens to have identical assertion and test counts.
        rng = _rnd.Random(hash(("random_test_rank", seed_src)) & 0xFFFFFFFF)
        picked = rng.sample(all_tests, min(top_k, len(all_tests)))
        return picked

    if rank == "max_entropy":
        by_max = sorted(per_assertion_entropy(entries),
                        key=lambda a: a["max_entropy"], reverse=True)
        return by_max[:top_k]

    if rank == "total_entropy":
        # Length-weighted per-assertion entropy: mean × n_tokens. Motivated
        # by the RQ1 within-suite correlations where total_entropy is the
        # strongest per-assertion predictor across all (benchmark, model)
        # cells.
        by_total = sorted(
            per_assertion_entropy(entries),
            key=lambda a: a["mean_entropy"] * a["n_tokens"], reverse=True,
        )
        return by_total[:top_k]

    if rank == "testbody_mean_entropy":
        return per_testbody_entropy(entries)[:top_k]

    if rank == "testbody_mean_entropy_right_eq":
        # Ablation: rank test bodies by mean entropy restricted to tokens on
        # the right-hand side of each assertion's `==` (expected-value
        # tokens). Non-equality assertions fall back to all their tokens.
        # Motivation: within-suite Spearman is ~25% stronger on this filter
        # than on full-testbody tokens (see RQ2 ablation).
        return per_testbody_entropy_right_of_eq(entries)[:top_k]

    if rank == "testbody_max_entropy":
        by_max = sorted(per_testbody_entropy(entries),
                        key=lambda a: a["max_entropy"], reverse=True)
        return by_max[:top_k]

    if rank == "testbody_total_entropy":
        # Length-weighted per-test-body entropy. Strongest per-test
        # predictor in RQ1 correlations for all three models.
        by_total = sorted(
            per_testbody_entropy(entries),
            key=lambda a: a["mean_entropy"] * a["n_tokens"], reverse=True,
        )
        return by_total[:top_k]

    if rank == "prominence_in_assert":
        peak_groups = peaks_grouped_by_assertion(entries, prominence=prominence)
        if peak_groups:
            return peak_groups[:top_k]
        # Fallback: no in-assertion peaks with the given prominence.
        return per_assertion_entropy(entries)[:top_k]

    # default: mean_entropy
    return per_assertion_entropy(entries)[:top_k]


def _compute_entropy_threshold(entries: list[dict], sigma_multiplier: float,
                               sigma_floor: float,
                               rank: str = "testbody_mean_entropy") -> dict:
    """Compute the fixed per-suite entropy threshold from baseline tokens.

    Uses the per-testbody distribution selected by `rank`:
      - "testbody_mean_entropy" (default): full-testbody tokens
      - "testbody_mean_entropy_right_eq": tokens to the right of each `==`
    T = mu + sigma_multiplier * max(sigma, sigma_floor).
    Returns a dict with mu, sigma, sigma_effective, T, and the raw list of
    (test_name, mean_entropy) pairs used, for trace reproducibility.
    """
    if rank == "testbody_mean_entropy_right_eq":
        testbodies = per_testbody_entropy_right_of_eq(entries)
    else:
        testbodies = per_testbody_entropy(entries)
    if not testbodies:
        return {
            "mu": None, "sigma": None, "sigma_effective": None, "T": None,
            "n_testbodies": 0, "testbody_entropies": [],
        }
    ents = [tb["mean_entropy"] for tb in testbodies]
    n = len(ents)
    mu = sum(ents) / n
    if n > 1:
        var = sum((e - mu) ** 2 for e in ents) / (n - 1)
        sigma = var ** 0.5
    else:
        sigma = 0.0
    sigma_eff = max(sigma, sigma_floor)
    T = mu + sigma_multiplier * sigma_eff
    return {
        "mu": mu,
        "sigma": sigma,
        "sigma_effective": sigma_eff,
        "T": T,
        "n_testbodies": n,
        "testbody_entropies": [{"test_name": tb["test_name"],
                                "mean_entropy": tb["mean_entropy"]}
                               for tb in testbodies],
    }


def _hot_testbodies_above_threshold(entries: list[dict], threshold: float,
                                    k_cap: int,
                                    rank: str = "testbody_mean_entropy") -> list[dict]:
    """Select test bodies with mean_entropy > threshold, sorted hottest first,
    capped at k_cap. The entropy distribution used is selected by `rank`
    (matching `_compute_entropy_threshold`). Returns [] if no test body
    exceeds threshold --- the loop's self-termination signal.
    """
    if not entries or threshold is None:
        return []
    if rank == "testbody_mean_entropy_right_eq":
        tbs = per_testbody_entropy_right_of_eq(entries)
    else:
        tbs = per_testbody_entropy(entries)
    over = [tb for tb in tbs if tb["mean_entropy"] > threshold]
    return over[:max(0, k_cap)]


def _regenerate(client, model: str, prompt: str, temperature: float,
                seed: int | None, disable_thinking: bool = False) -> tuple[str, list[dict], str]:
    """Run one Analyzer generation, return (code, entries, finish_reason)."""
    gen = stream_generate(
        client, model, prompt,
        temperature=temperature, top_logprobs=TOP_LOGPROBS, seed=seed,
        disable_thinking=disable_thinking,
    )
    return extract_python(gen.text), gen.entries, gen.finish_reason


def repair_row(row: dict, task: dict, *,
               client: OpenAI, model: str = DEFAULT_MODEL,
               mode: str = "entropy",
               max_iterations: int = 1,
               temperature: float = 0.0,
               stop_on_success: bool = True,
               timeout_sec: int = 60,
               top_k: int = 1,
               top_k_mode: str = "fixed",
               top_k_fallback: int = 3,
               rank: str = "max_entropy",
               prominence: float = 0.8,
               sigma_multiplier: float = 1.5,
               sigma_floor: float = 0.01,
               no_guard: bool = False,
               disable_thinking: bool = False) -> dict:
    """Run the analyzer loop on a single row. Skips rows that already passed.

    top_k_mode:
      - "fixed": use `top_k` unchanged every iteration.
      - "n_failing_dynamic": at each iteration, set K = current state's
        n_failed (falls back to `top_k_fallback` if it's 0). Static-K modes
        (fixed, n_failing_tests) should pre-compute K in the caller and pass
        via `top_k`.
      - "entropy_threshold": compute a fixed per-suite threshold
        T = mu + sigma_multiplier * max(sigma, sigma_floor) from the baseline
        (iteration 0) per-testbody mean_entropy distribution. At each
        iteration, flag test bodies with mean_entropy > T, capped at `top_k`
        (K2). If K1=0 (no body exceeds T), the iteration is a no-op and
        counts as loop termination. Fixed to rank=testbody_mean_entropy.
    """
    trace: dict[str, Any] = {
        "task_id": row.get("task_id"),
        "mode": mode,
        "model": model,
        "top_k": top_k,
        "top_k_mode": top_k_mode,
        "rank": rank,
        "prominence": prominence,
        "baseline_pass_rate": row.get("exec__pass_rate"),
        "iterations": [],
    }

    state = _current_state(row, task)

    # For entropy_threshold mode, compute T once from baseline entries.
    entropy_threshold_info = None
    if top_k_mode == "entropy_threshold":
        entropy_threshold_info = _compute_entropy_threshold(
            state["entries"], sigma_multiplier, sigma_floor, rank=rank,
        )
        trace["sigma_multiplier"] = sigma_multiplier
        trace["sigma_floor"] = sigma_floor
        trace["entropy_threshold"] = {
            "mu": entropy_threshold_info["mu"],
            "sigma": entropy_threshold_info["sigma"],
            "sigma_effective": entropy_threshold_info["sigma_effective"],
            "T": entropy_threshold_info["T"],
            "n_testbodies_baseline": entropy_threshold_info["n_testbodies"],
        }

    trace["iterations"].append({
        "iter": 0,
        "code": state["code"],
        "n_passed": state["n_passed"],
        "n_failed": state["n_failed"],
        "n_collected": state["n_collected"],
        "pass_rate": state["pass_rate"],
        "source": "generator",  # the original generation
        "accepted": True,
        "reject_reason": None,
    })

    if state["pass_rate"] == 1.0 and stop_on_success:
        trace["final_pass_rate"] = state["pass_rate"]
        trace["improved"] = False
        trace["n_iterations"] = 0
        return trace

    for it in range(1, max_iterations + 1):
        # Dynamic K: recompute from current state's n_failed each iteration.
        # Static modes ("fixed", "n_failing_tests") use whatever `top_k` was passed in.
        if top_k_mode == "n_failing_dynamic":
            current_nf = state.get("n_failed") or 0
            iter_k = current_nf if current_nf > 0 else top_k_fallback
        else:
            iter_k = top_k

        # entropy_threshold: select by fixed T instead of top-K rank; K=0 is
        # a valid termination signal (whole suite is now below the bar).
        if top_k_mode == "entropy_threshold":
            T = entropy_threshold_info["T"] if entropy_threshold_info else None
            hots = _hot_testbodies_above_threshold(
                state["entries"], threshold=T, k_cap=top_k, rank=rank,
            )
            k1 = len(hots)
            if k1 == 0:
                trace["iterations"].append({
                    "iter": it,
                    "source": "analyzer",
                    "top_k_used": 0,
                    "k1": 0,
                    "k2": top_k,
                    "entropy_threshold_T": T,
                    "accepted": False,
                    "reject_reason": "no_testbody_above_threshold",
                })
                # Terminate: the entropy signal says the suite has converged.
                break
        else:
            hots = _hot_assertions_from(state["entries"], top_k=iter_k,
                                        rank=rank, prominence=prominence,
                                        task_id=trace["task_id"])
        prompt = _build_prompt(mode, task, state["code"], state["failure_log"], hots)
        try:
            new_code, new_entries, finish = _regenerate(
                client, model, prompt,
                temperature=temperature, seed=2000 + it,
                disable_thinking=disable_thinking,
            )
        except Exception as e:
            trace["iterations"].append({
                "iter": it, "source": "analyzer", "error": repr(e),
                "accepted": False, "reject_reason": "exception",
            })
            break

        exec_result = evaluate_generation(task, new_code, timeout_sec=timeout_sec)
        pr = (exec_result["n_passed"] / exec_result["n_collected"]
              if exec_result["n_collected"] else None)

        # Guard decision: what the accept/reject guard *would* say. Always
        # computed so we can post-hoc analyze with-guard vs no-guard from the
        # same trace even under --no-guard.
        prev_pr = state["pass_rate"] if state["pass_rate"] is not None else -1.0
        guard_accepted = True
        guard_reject_reason = None
        if exec_result["n_collected"] == 0:
            guard_accepted = False
            guard_reject_reason = "no_tests_collected"
        elif pr is None:
            guard_accepted = False
            guard_reject_reason = "pass_rate_undefined"
        elif pr < prev_pr:
            guard_accepted = False
            guard_reject_reason = f"regressed({pr:.3f}<{prev_pr:.3f})"

        # Effective decision: under --no-guard, always accept (unless the run
        # is malformed with no tests collected, which is a hard failure not
        # a regression).
        if no_guard:
            # Still refuse to accept a completely broken generation (no tests
            # at all), because that would leave the loop with an empty state.
            if exec_result["n_collected"] == 0 or pr is None:
                accepted = False
                reject_reason = guard_reject_reason
            else:
                accepted = True
                reject_reason = None
        else:
            accepted = guard_accepted
            reject_reason = guard_reject_reason

        iter_entry = {
            "iter": it,
            "code": new_code,
            "n_passed": exec_result["n_passed"],
            "n_failed": exec_result["n_failed"],
            "n_collected": exec_result["n_collected"],
            "pass_rate": pr,
            "source": "analyzer",
            "finish_reason": finish,
            "top_k_used": len(hots) if top_k_mode == "entropy_threshold" else iter_k,
            "hot_assertion": hots[0] if hots else None,
            "hot_assertions": hots,
            "accepted": accepted,
            "reject_reason": reject_reason,
            # Record the guard's would-be decision separately for post-hoc
            # comparison under --no-guard.
            "guard_would_accept": guard_accepted,
            "guard_would_reject_reason": guard_reject_reason,
        }
        if top_k_mode == "entropy_threshold":
            iter_entry["k1"] = len(hots)
            iter_entry["k2"] = top_k
            iter_entry["entropy_threshold_T"] = (
                entropy_threshold_info["T"] if entropy_threshold_info else None
            )
        trace["iterations"].append(iter_entry)

        if accepted:
            state = {
                "code": new_code,
                "entries": new_entries,
                "n_passed": exec_result["n_passed"],
                "n_failed": exec_result["n_failed"],
                "n_collected": exec_result["n_collected"],
                "pass_rate": pr,
                "failure_log": exec_result.get("pytest_stdout_tail", "")
                               + "\n"
                               + exec_result.get("pytest_stderr_tail", ""),
            }
        # If rejected: state stays as it was; the next iteration retries from
        # the same starting point (different seed via 2000+it).

        if state["pass_rate"] == 1.0 and stop_on_success:
            break

    analyzer_iters = [it for it in trace["iterations"] if it["iter"] > 0]
    trace["n_iterations_attempted"] = len(analyzer_iters)
    trace["n_iterations_accepted"] = sum(1 for it in analyzer_iters if it.get("accepted"))
    trace["n_iterations_rejected"] = sum(1 for it in analyzer_iters if not it.get("accepted"))
    trace["n_iterations"] = trace["n_iterations_accepted"]  # backwards-compat name
    trace["final_pass_rate"] = state["pass_rate"]
    baseline = trace["baseline_pass_rate"] or 0.0
    final = trace["final_pass_rate"] or 0.0
    trace["improved"] = final > baseline
    trace["delta_pass_rate"] = final - baseline
    return trace


# ---- CLI ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="outputs_bench",
                    help="directory containing results.jsonl from a base sweep")
    ap.add_argument("--out", default="outputs_bench/analyzer_traces.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"LLM to use as Analyzer (default {DEFAULT_MODEL}). "
                         "Should match the model that produced the base sweep.")
    ap.add_argument("--mode", choices=("entropy", "naive", "entropy_only",
                                       "entropy_only_strict",
                                       "entropy_only_strict_testbody",
                                       "entropy_only_strict_optout",
                                       "entropy_only_strict_testbody_optout"),
                    default="entropy",
                    help="entropy: entropy pointer + failure log; "
                         "naive: failure log only; "
                         "entropy_only: entropy pointer only (no failure log); "
                         "entropy_only_strict: entropy pointer only, and the "
                         "Analyzer is told to modify only the flagged "
                         "assertions (no free clause); "
                         "entropy_only_strict_testbody: same as "
                         "entropy_only_strict but the pointer names whole "
                         "test functions, and the Analyzer is told to modify "
                         "assertions only inside those tests; "
                         "entropy_only_strict_optout / "
                         "entropy_only_strict_testbody_optout: same strict "
                         "prompts but with a clause allowing the Analyzer to "
                         "preserve a flagged assertion verbatim if, after "
                         "re-deriving, it concludes the current form is "
                         "already correct (opt-out ablation).")
    ap.add_argument("--top-k", type=int, default=1,
                    help="how many hottest assertions to include in the "
                         "entropy prompt (default 1)")
    ap.add_argument("--top-k-mode",
                    choices=("fixed", "n_failing_tests", "n_failing_dynamic",
                             "entropy_threshold"),
                    default="fixed",
                    help="fixed: use --top-k as a constant. "
                         "n_failing_tests: for each task set K = the row's "
                         "exec__n_failed (fair-scope comparison against naive, "
                         "which sees exactly the failing tests in its log). "
                         "K is fixed across iterations. "
                         "n_failing_dynamic: like n_failing_tests but K is "
                         "recomputed each iteration from the current accepted "
                         "state's n_failed (shrinks as tests get fixed). "
                         "Falls back to --top-k if a row has no failures. "
                         "entropy_threshold: compute a fixed per-suite "
                         "threshold T = mu + sigma_multiplier * max(sigma, "
                         "sigma_floor) from the baseline per-testbody "
                         "mean_entropy distribution. At each iteration flag "
                         "test bodies with mean_entropy > T, capped at "
                         "--top-k (K2). If no test body exceeds T, the "
                         "iteration terminates (self-termination signal). "
                         "Only meaningful with --rank testbody_mean_entropy.")
    ap.add_argument("--sigma-multiplier", type=float, default=1.5,
                    help="entropy_threshold: multiplier k in T = mu + k*sigma "
                         "(default 1.5). Higher = fewer flags.")
    ap.add_argument("--sigma-floor", type=float, default=0.01,
                    help="entropy_threshold: minimum sigma used to guard "
                         "against degenerate near-zero baseline variance "
                         "(default 0.01 nats).")
    ap.add_argument("--rank", default="max_entropy",
                    choices=("mean_entropy", "max_entropy", "total_entropy",
                             "prominence_in_assert",
                             "testbody_mean_entropy",
                             "testbody_mean_entropy_right_eq",
                             "testbody_max_entropy",
                             "testbody_total_entropy",
                             "random", "random_test"),
                    help="how to rank assertions/tests for top-K selection. "
                         "'total_entropy' = mean_entropy × n_tokens (length-"
                         "weighted). 'random' picks K assertions uniformly at "
                         "random; 'random_test' picks K whole test functions "
                         "uniformly at random. Both random rankers are seeded "
                         "per-task from task_id.")
    ap.add_argument("--prominence", type=float, default=0.8,
                    help="scipy prominence threshold (nats) when "
                         "--rank=prominence_in_assert")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="pass chat_template_kwargs.enable_thinking=False to the "
                         "model (e.g. GLM-5.2, Qwen3-32B).")
    ap.add_argument("--no-guard", action="store_true",
                    help="disable the accept/reject regression guard; every "
                         "analyzer rewrite is accepted unconditionally (unless "
                         "the generation collected 0 tests). The trace still "
                         "records what the guard would have decided in "
                         "guard_would_accept / guard_would_reject_reason.")
    ap.add_argument("--no-stop-on-success", action="store_true",
                    help="disable the early-exit when the suite's pass rate "
                         "reaches 1.0. By default the loop stops iterating once "
                         "pr=1.0 to avoid regressing a perfect suite; this flag "
                         "forces the loop to run its full --max-iterations "
                         "budget regardless. Useful to measure true oracle-free "
                         "deployment compute (where pytest is unavailable to "
                         "signal early-exit).")
    ap.add_argument("--max-iterations", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--only-failed", action="store_true", default=True,
                    help="skip rows with baseline pass_rate == 1.0 (default on)")
    ap.add_argument("--only-task-ids-from", default=None,
                    help="path to a file with one task_id per line; only process "
                         "those task_ids. Applied after --only-failed. "
                         "Useful for rerunning a subset of rows (e.g. with "
                         "--no-stop-on-success).")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows processed (for smoke tests)")
    ap.add_argument("--resume", action="store_true",
                    help="skip task_ids already present in --out file")
    args = ap.parse_args()

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    in_dir = Path(args.in_dir)
    if not in_dir.is_absolute():
        in_dir = _REPO_ROOT / in_dir
    rows = [json.loads(l) for l in (in_dir / "results.jsonl").read_text().split("\n")
            if l.strip()]

    # We need the original task dicts to rebuild prompts + eval. Reload from
    # the benchmark to get the correct source_module_name (which differs
    # across benchmarks: HumanEval uses `solution_N`, MBPP+ uses
    # `mbpp_solution_N`, ClassEval uses `classeval_solution_N`, etc.).
    tasks_by_id = {}
    for r in rows:
        tid = r.get("task_id")
        if tid not in tasks_by_id:
            tasks_by_id[tid] = _lookup_task_by_id(tid)

    targets = [r for r in rows if not (args.only_failed and r.get("exec__pass_rate") == 1.0)]
    if args.only_task_ids_from:
        wanted = set()
        for line in Path(args.only_task_ids_from).read_text().split("\n"):
            tid = line.strip()
            if tid: wanted.add(tid)
        before = len(targets)
        targets = [r for r in targets if r["task_id"] in wanted]
        print(f"[analyzer] filtered to {len(targets)}/{before} rows "
              f"via --only-task-ids-from (requested {len(wanted)} ids)")
    if args.limit:
        targets = targets[:args.limit]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().split("\n"):
            try:
                t = json.loads(line)
                done_ids.add(t["task_id"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"[analyzer] resume: skipping {len(done_ids)} completed task_ids")
        targets = [r for r in targets if r["task_id"] not in done_ids]

    print(f"[analyzer] {len(targets)}/{len(rows)} rows to process (mode={args.mode}, "
          f"max_iter={args.max_iterations})")

    # `client` imported from llm_loader; replace llm_loader.py if your
    # endpoint isn't OpenAI-compatible.
    open_mode = "a" if args.resume else "w"

    n_improved = 0
    with out_path.open(open_mode) as f:
        bar = tqdm(targets, desc=f"analyzer:{args.mode}", unit="row",
                   dynamic_ncols=True)
        for row in bar:
            t0 = time.monotonic()
            # Determine per-row K
            if args.top_k_mode == "n_failing_tests":
                n_failed = row.get("exec__n_failed") or 0
                row_k = n_failed if n_failed > 0 else args.top_k
            elif args.top_k_mode == "n_failing_dynamic":
                # Iter 1's K = baseline n_failed; subsequent iterations
                # recomputed inside repair_row from current state.
                n_failed = row.get("exec__n_failed") or 0
                row_k = n_failed if n_failed > 0 else args.top_k
            else:
                row_k = args.top_k
            trace = repair_row(
                row, tasks_by_id[row["task_id"]],
                client=client, mode=args.mode, model=args.model,
                max_iterations=args.max_iterations,
                temperature=args.temperature,
                top_k=row_k,
                top_k_mode=args.top_k_mode,
                top_k_fallback=args.top_k,
                rank=args.rank,
                prominence=args.prominence,
                sigma_multiplier=args.sigma_multiplier,
                sigma_floor=args.sigma_floor,
                no_guard=args.no_guard,
                stop_on_success=not args.no_stop_on_success,
                disable_thinking=args.disable_thinking,
            )
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            f.flush()
            elapsed = time.monotonic() - t0
            if trace["improved"]:
                n_improved += 1
            tqdm.write(f"[analyzer] {trace['task_id']} baseline={trace['baseline_pass_rate']} "
                       f"→ final={trace['final_pass_rate']} "
                       f"improved={trace['improved']} ({elapsed:.0f}s)")
            bar.set_postfix_str(f"improved={n_improved}/{bar.n + 1}")

    print(f"[analyzer] wrote {out_path}")
    print(f"[analyzer] improved: {n_improved}/{len(targets)}")


def _lookup_task_by_id(task_id: str) -> dict:
    """Rebuild the full task dict from the task_id prefix. Returns the same
    shape produced by `load_humaneval` / `load_mbpp_plus` / etc., so the
    caller has the correct `source_module_name` for pytest to import."""
    from tasks import load_bench
    if task_id.startswith("Mbpp/"):
        bench = "mbpp_plus"
    elif task_id.startswith("BigCodeBench/"):
        bench = "bigcodebench"
    elif task_id.startswith("ClassEval/"):
        bench = "classeval"
    elif task_id.startswith("CoderEval/"):
        bench = "codereval"
    elif task_id.startswith("APPS/"):
        bench = "apps_intro"
    else:
        bench = "humaneval"
    tasks = load_bench(bench, task_ids=[task_id])
    if not tasks:
        raise ValueError(f"unknown task_id: {task_id}")
    return tasks[0]


if __name__ == "__main__":
    main()
