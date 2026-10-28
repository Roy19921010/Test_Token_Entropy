"""Add a Mean summary row to the two AUROC / PR-AUC classification tables
in ``paper/rq2_righteq_tables.tex``:

  tab:rq2-pertest-cls-righteq       (per-testbody within-suite AUROC/PR-AUC)
  tab:rq2-perassert-cls-righteq     (per-assertion within-suite AUROC/PR-AUC)

Each table has 4 numeric columns:
  Full AUROC | Full PR-AUC | Right-of-== AUROC | Right-of-== PR-AUC

The summary row reports the arithmetic mean of each column across the 12
(model, benchmark) rows. The PR-AUC column text in the body is like
``0.537 ($\\times$2.3)`` -- we extract only the leading float for the mean,
and omit the (xN) multiplier in the Mean row because the per-cell null base
rates differ (one global multiplier is not meaningful).

Bolding: within each metric (AUROC or PR-AUC), the larger of the two
variant means is bolded.

Idempotency: strips any previous Mean row (identified by the leading
``\\emph{Mean}`` cell) before inserting the fresh one.

Inputs: ``paper/rq2_righteq_tables.tex``.
Outputs: overwrites in-place; writes ``paper/add_mean_cls_rows.log``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "rq2_righteq_tables.tex"
LOG = ROOT / "paper" / "logs" / "add_mean_cls_rows.log"

TABLES = [
    "tab:rq2-pertest-cls-righteq",
    "tab:rq2-perassert-cls-righteq",
]

# Row shape per data line:
#  "ModelX & Bench & $0.676$-or-0.676 & 0.537 ($\times$2.3) & 0.730 & 0.646 ($\times$2.7) \\"
# Model and Bench cells may be empty strings for continuation rows.
# We extract the four numeric cells by scanning floats in the line.
_FLOAT = re.compile(r"[-+]?\d+\.\d+")


def _parse_rows(block: str) -> list[list[float]]:
    """Return list of [AUROC_full, PRAUC_full, AUROC_re, PRAUC_re] for the
    12 data rows between \\midrule markers.

    We parse by splitting each data row on '&' into cells and extracting
    the first float from each cell. The expected shape of a data row is:

        Model & Bench & AUROC_full & PRAUC_full ($\times$N) & AUROC_re & PRAUC_re ($\times$N) \\\\

    i.e. 6 cells; we take the first float of cells 2..5. This avoids
    accidentally parsing a float inside the model-name cell (e.g. the
    "-5.2" inside "GLM-5.2") as a data value.
    """
    out = []
    for line in block.splitlines():
        s = line.strip()
        if not s.endswith(r"\\"):
            continue
        if r"\emph{Mean}" in s:
            continue
        if "Model" in s and "Bench" in s:  # header
            continue
        if "\\toprule" in s or "\\midrule" in s or "\\bottomrule" in s or "\\cmidrule" in s:
            continue
        # Drop the trailing \\.
        body = s[:-2] if s.endswith(r"\\") else s
        cells = [c.strip() for c in body.split("&")]
        if len(cells) < 6:
            continue
        def _first_float(cell: str) -> float | None:
            m = _FLOAT.search(cell)
            return float(m.group(0)) if m else None
        au_full = _first_float(cells[2])
        pr_full = _first_float(cells[3])
        au_re   = _first_float(cells[4])
        pr_re   = _first_float(cells[5])
        if any(v is None for v in (au_full, pr_full, au_re, pr_re)):
            continue
        out.append([au_full, pr_full, au_re, pr_re])
    return out


def _patch_cls_table(block: str, label: str) -> tuple[str, list[float]]:
    # Strip any previous Mean row (idempotency).
    block = re.sub(
        r"\\midrule\s*\\emph\{Mean\}[^\n]*\\\\\n",
        "",
        block,
    )
    rows = _parse_rows(block)
    if len(rows) < 12:
        raise RuntimeError(
            f"{label}: parsed {len(rows)} rows; expected 12"
        )
    cols = list(zip(*rows[:12]))
    col_means = [mean(c) for c in cols]  # length 4

    # Bold the better variant for AUROC and for PR-AUC.
    au_full_bold = col_means[0] >= col_means[2]
    au_re_bold   = col_means[2] >  col_means[0]
    pr_full_bold = col_means[1] >= col_means[3]
    pr_re_bold   = col_means[3] >  col_means[1]

    def _fmt(v, bold):
        s = f"{v:.3f}"
        return r"\textbf{" + s + "}" if bold else s

    mean_row = (
        r"\midrule" + "\n"
        + r"\emph{Mean} & --- & "
        + f"{_fmt(col_means[0], au_full_bold)} & "
        + f"{_fmt(col_means[1], pr_full_bold)} & "
        + f"{_fmt(col_means[2], au_re_bold)} & "
        + f"{_fmt(col_means[3], pr_re_bold)} "
        + r"\\" + "\n"
    )

    # Insert the Mean row immediately before \bottomrule.
    br_idx = block.rindex(r"\bottomrule")
    block = block[:br_idx] + mean_row + block[br_idx:]
    return block, col_means


def _slice_table(txt: str, label: str) -> tuple[int, int]:
    lbl_idx = txt.index(f"\\label{{{label}}}")
    start = txt.rfind(r"\begin{table*}", 0, lbl_idx)
    end = txt.index(r"\end{table*}", lbl_idx) + len(r"\end{table*}")
    return start, end


def main():
    log = []

    def emit(s):
        log.append(s)
        print(s)

    txt = TEX.read_text()
    emit(f"Reading {TEX}  ({len(txt)} chars)")

    for label in TABLES:
        s, e = _slice_table(txt, label)
        new_block, col_means = _patch_cls_table(txt[s:e], label)
        emit(f"\n=== {label} ===")
        emit(f"  Full AUROC mean       = {col_means[0]:.3f}")
        emit(f"  Full PR-AUC mean      = {col_means[1]:.3f}")
        emit(f"  Right-eq AUROC mean   = {col_means[2]:.3f}")
        emit(f"  Right-eq PR-AUC mean  = {col_means[3]:.3f}")
        emit(f"  AUROC delta           = {col_means[2]-col_means[0]:+.3f}")
        emit(f"  PR-AUC delta          = {col_means[3]-col_means[1]:+.3f}")
        txt = txt[:s] + new_block + txt[e:]

    TEX.write_text(txt)
    emit(f"\nWrote {TEX}  ({len(txt)} chars)")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(log) + "\n")
    print(f"Wrote {LOG}", file=sys.stderr)


if __name__ == "__main__":
    main()
