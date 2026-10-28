"""HumanEval task loader → bench-pipeline task dicts.

Loads the OpenAI HumanEval JSONL from `datasets/human-eval.jsonl.gz`. If the
file is missing, raises FileNotFoundError. To fetch it:
    curl -L -o datasets/human-eval.jsonl.gz \\
      https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

Each task dict has:
    task_id, prompt, source_code, source_module_name

`prompt` is the *instruction we send to the LLM* — it references the source
module by name and asks for pytest tests. `source_code` is the reference
implementation we run coverage against.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = _REPO_ROOT / "datasets" / "human-eval.jsonl.gz"
MBPP_PLUS_PATH = _REPO_ROOT / "datasets" / "mbpp-plus.jsonl.gz"
CLASSEVAL_PATH = _REPO_ROOT / "datasets" / "classeval.jsonl.gz"


TEST_INSTRUCTION_TEMPLATE = """\
You are writing a pytest test file for the Python module below.

Module `{module}` (`{module}.py`):
```python
{source}
```

Write a comprehensive pytest test suite that:
- imports the function(s) from `{module}` (e.g., `from {module} import ...`)
- covers normal cases, edge cases, and boundary conditions
- uses plain `assert` statements (no unittest classes)
- has no external dependencies beyond pytest

Return ONLY the Python test file contents inside a single ```python fenced block.
"""


CLASSEVAL_INSTRUCTION_TEMPLATE = """\
You are writing a pytest test file for the class `{class_name}` in the Python
module below.

Class description:
{class_description}

Module `{module}` (`{module}.py`):
```python
{source}
```

Write a comprehensive pytest test suite that:
- imports `{class_name}` from `{module}` (i.e. `from {module} import {class_name}`)
- covers each public method with normal, edge, and boundary cases
- constructs class instances as needed with reasonable fixture data
- uses plain `assert` statements (no unittest classes)
- has no external dependencies beyond pytest

Return ONLY the Python test file contents inside a single ```python fenced block.
"""


MBPP_INSTRUCTION_TEMPLATE = """\
You are writing a pytest test file for the function `{entry_point}` in the
Python module below.

Task description:
{description}

Module `{module}` (`{module}.py`):
```python
{source}
```

Write a comprehensive pytest test suite that:
- imports `{entry_point}` from `{module}` (i.e. `from {module} import {entry_point}`)
- primarily tests `{entry_point}` — you may test helpers only if it clarifies behavior
- covers normal cases, edge cases, and boundary conditions
- uses plain `assert` statements (no unittest classes)
- has no external dependencies beyond pytest

Return ONLY the Python test file contents inside a single ```python fenced block.
"""


def _stub_source_from_prompt(prompt: str, canonical: str) -> str:
    """HumanEval `prompt` is the signature+docstring; `canonical_solution` is
    just the body. Concatenate them to get the full function definition."""
    return prompt + canonical


def load_humaneval(
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[dict]:
    """Load HumanEval tasks. If `task_ids` is given (e.g. ['HumanEval/50',
    'HumanEval/83']), only those are returned, in that order; `limit` is
    ignored. Otherwise the first `limit` tasks in file order are returned.

    Raises FileNotFoundError if the dataset isn't downloaded.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"HumanEval dataset not found at {DATASET_PATH}. Fetch it with:\n"
            "  curl -L -o datasets/human-eval.jsonl.gz "
            "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
        )

    by_id: dict[str, dict] = {}
    with gzip.open(DATASET_PATH, "rt") as f:
        for line in f:
            obj = json.loads(line)
            task_id = obj["task_id"]
            module = "solution_" + task_id.split("/")[-1]
            source = _stub_source_from_prompt(obj["prompt"], obj["canonical_solution"])
            by_id[task_id] = {
                "task_id": task_id,
                "prompt": TEST_INSTRUCTION_TEMPLATE.format(module=module, source=source),
                "source_code": source,
                "source_module_name": module,
            }
    if task_ids is not None:
        missing = [t for t in task_ids if t not in by_id]
        if missing:
            raise ValueError(f"unknown task_ids: {missing}")
        return [by_id[t] for t in task_ids]
    rows = list(by_id.values())
    return rows[:limit] if limit is not None else rows


def _module_name_from_mbpp_id(task_id: str) -> str:
    # 'Mbpp/2' -> 'mbpp_solution_2'
    return "mbpp_solution_" + task_id.split("/")[-1]


def _first_sentence(text: str) -> str:
    """Trim MBPP task descriptions — often multi-line docstring blocks."""
    text = text.strip()
    for sep in (". ", "\n\n"):
        if sep in text:
            return text.split(sep, 1)[0].strip() + ("." if sep == ". " else "")
    return text


def load_mbpp_plus(
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[dict]:
    """Load MBPP+ tasks. Expects `datasets/mbpp-plus.jsonl.gz` — fetch via:

        python -c "from evalplus.data import get_mbpp_plus; \\
                   import json, gzip; \\
                   d = get_mbpp_plus(); \\
                   open('/tmp/mbpp.jsonl','w').write('\\n'.join(json.dumps(v) for v in d.values())); \\
                   import shutil; shutil.copy('/tmp/mbpp.jsonl','datasets/mbpp-plus.jsonl')"
        gzip datasets/mbpp-plus.jsonl

    Each MBPP+ entry has fields: task_id, prompt, code, entry_point, ...
    We build a `task` dict matching the shape used elsewhere in the pipeline.
    """
    if not MBPP_PLUS_PATH.exists():
        raise FileNotFoundError(
            f"MBPP+ dataset not found at {MBPP_PLUS_PATH}. See docstring for fetch command."
        )

    by_id: dict[str, dict] = {}
    with gzip.open(MBPP_PLUS_PATH, "rt") as f:
        for line in f:
            obj = json.loads(line)
            task_id = obj["task_id"]
            # `canonical_solution` is the reference implementation body/full source.
            # Some tasks store the full source there (with imports); others store
            # only the body. If we detect no `def`, fall back to prompt+solution.
            canonical = obj["canonical_solution"]
            entry_point = obj["entry_point"]
            if "def " in canonical:
                source = canonical
            else:
                # Prepend the prompt's example assert line-stripped signature by
                # constructing a `def entry_point(...)` header — MBPP prompts
                # embed the function inside a docstring, so this fallback rarely
                # fires.
                source = f"def {entry_point}(*args, **kwargs):\n{canonical}"
            desc = _first_sentence(obj.get("prompt", "") or "")
            module = _module_name_from_mbpp_id(task_id)
            by_id[task_id] = {
                "task_id": task_id,
                "prompt": MBPP_INSTRUCTION_TEMPLATE.format(
                    module=module, source=source,
                    entry_point=entry_point, description=desc,
                ),
                "source_code": source,
                "source_module_name": module,
                "entry_point": entry_point,
            }
    if task_ids is not None:
        missing = [t for t in task_ids if t not in by_id]
        if missing:
            raise ValueError(f"unknown task_ids: {missing}")
        return [by_id[t] for t in task_ids]
    rows = list(by_id.values())
    return rows[:limit] if limit is not None else rows


def _module_name_from_classeval_id(task_id: str) -> str:
    # 'ClassEval_0' -> 'classeval_solution_0'
    return "classeval_solution_" + task_id.replace("ClassEval_", "").replace("/", "_")


def load_classeval(
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[dict]:
    """Load ClassEval tasks. 100 class-level Python tasks.

    Each ClassEval entry has fields:
        task_id ('ClassEval_N'),
        solution_code (full class implementation with imports),
        class_name, class_description,
        skeleton, test, methods_info, import_statement, ...

    We wrap task_id as 'ClassEval/N' to match our internal prefix convention.
    """
    if not CLASSEVAL_PATH.exists():
        raise FileNotFoundError(
            f"ClassEval dataset not found at {CLASSEVAL_PATH}."
        )

    by_id: dict[str, dict] = {}
    with gzip.open(CLASSEVAL_PATH, "rt") as f:
        for line in f:
            obj = json.loads(line)
            raw_id = obj["task_id"]  # e.g. "ClassEval_0"
            # Normalize to 'ClassEval/N' for consistency with our other benchmarks
            n = raw_id.replace("ClassEval_", "")
            task_id = f"ClassEval/{n}"
            source = obj["solution_code"]
            class_name = obj["class_name"]
            description = _first_sentence(obj.get("class_description", "") or "")
            module = _module_name_from_classeval_id(raw_id)
            by_id[task_id] = {
                "task_id": task_id,
                "prompt": CLASSEVAL_INSTRUCTION_TEMPLATE.format(
                    module=module, source=source,
                    class_name=class_name, class_description=description,
                ),
                "source_code": source,
                "source_module_name": module,
                "entry_point": class_name,
            }
    if task_ids is not None:
        missing = [t for t in task_ids if t not in by_id]
        if missing:
            raise ValueError(f"unknown task_ids: {missing}")
        return [by_id[t] for t in task_ids]
    rows = list(by_id.values())
    return rows[:limit] if limit is not None else rows


def load_bench(name: str, **kwargs) -> list[dict]:
    """Dispatch — pass `name` in {'humaneval', 'mbpp_plus', 'classeval'}."""
    if name == "humaneval":
        return load_humaneval(**kwargs)
    if name == "mbpp_plus":
        return load_mbpp_plus(**kwargs)
    if name == "classeval":
        return load_classeval(**kwargs)
    raise ValueError(f"unknown benchmark: {name!r}")
