"""Rebuild the three correlation tables in ``paper/rq2_righteq_tables.tex``
with a cleaner bolding scheme and a new Mean $|\\rho|$ column:

  tab:rq1-suite-righteq       (suite-level cross-suite rho)
  tab:rq2-pertest-righteq     (per-testbody within-suite rho)
  tab:rq2-perassert-righteq   (per-assertion within-suite rho)

Bolding scheme:
  * In each benchmark data column (one of 12), find the single row with the
    largest absolute rho across the eight (4 metric x 2 variant) data rows
    and bold that one number. Twelve bolded cells per table.
  * In the Mean $|\\rho|$ column, bold the single highest absolute mean
    across all eight rows (so exactly one Mean cell is bolded per table).

Numbers:
  * The 12 per-cell rho values are read verbatim from the existing table
    rows (unchanged by this script).
  * The per-row Mean $|\\rho|$ is computed as the FISHER-z WEIGHTED mean of
    the absolute cell rhos. Standard procedure for pooling correlations
    across studies: transform each rho to z = atanh(rho); weight each z
    by its inverse variance (approximately N - 3, with the Spearman
    constant 1.06 cancelling from the normalization); compute the weighted
    mean of z; back-transform via tanh.
    N per cell is read from ``paper/recompute_audit.log``:
      - suite:     [suite] <dir> N=<n>
      - testbody:  [per-testbody] <dir> n_rows_scored=<n>
      - assertion: [per-assertion] <dir> n_scored=<n>

Idempotency: re-running overwrites the previous Mean column and re-bolds
cells correctly.

Inputs: ``paper/rq2_righteq_tables.tex`` and ``paper/recompute_audit.log``.
Outputs: overwrites ``paper/rq2_righteq_tables.tex`` and writes an audit log
to ``paper/add_mean_rho_rows.log``.
"""
from __future__ import annotations

import re
import sys
from math import atanh, tanh
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "rq2_righteq_tables.tex"
AUDIT = ROOT / "paper" / "logs" / "recompute_audit.log"

METRIC_LABELS = [
    r"$H_{\text{mean}}$",
    r"$H_{\text{max}}$",
    r"$\mathrm{PPL}$",
    r"$g_{\text{top1}}$",
]
# Each entry: (table_label, N-source-key, N-regex)
# N-regex matches lines in recompute_audit.log; group 1 = dirname, group 2 = N.
TABLES = [
    ("tab:rq1-suite-righteq",     r"^\[suite\] (\S+) N=(\d+)"),
    ("tab:rq2-pertest-righteq",   r"^\[per-testbody\] (\S+) n_rows_scored=(\d+)"),
    ("tab:rq2-perassert-righteq", r"^\[per-assertion\] (\S+) n_scored=(\d+)"),
]
CELL_ORDER = [
    "data/qwen/humaneval",
    "data/qwen/mbpp_plus",
    "data/qwen/classeval",
    "data/gpt_oss/humaneval",
    "data/gpt_oss/mbpp_plus",
    "data/gpt_oss/classeval",
    "data/glm52/humaneval",
    "data/glm52/mbpp_plus",
    "data/glm52/classeval",
    "data/deepseek_v4/humaneval",
    "data/deepseek_v4/mbpp_plus",
    "data/deepseek_v4/classeval",
]


def fisher_z_weighted_mean(abs_rhos: list[float], Ns: list[int]) -> float:
    """Fisher-z weighted mean of absolute Spearman rhos. Each |rho| is
    transformed to z = atanh(|rho|), weighted by (N - 3) (inverse variance
    for Spearman up to a constant), averaged, and back-transformed via tanh.
    Falls back to a safely-clamped atanh so |rho| = 1.0 doesn't produce
    infinity.
    """
    assert len(abs_rhos) == len(Ns) == 12
    zs = [atanh(min(r, 0.999_999)) for r in abs_rhos]
    ws = [max(1, n - 3) for n in Ns]
    zbar = sum(z * w for z, w in zip(zs, ws)) / sum(ws)
    return tanh(zbar)


def load_Ns_for_table(regex: str) -> dict[str, int]:
    """Return {dirname: N} parsed from the audit log."""
    out: dict[str, int] = {}
    text = AUDIT.read_text()
    pat = re.compile(regex, re.M)
    for m in pat.finditer(text):
        out[m.group(1)] = int(m.group(2))
    return out


def _parse_floats(cell_strs):
    out = []
    for c in cell_strs:
        m = re.search(r"[+-]?\d+\.\d+", c)
        if m:
            out.append(float(m.group(0)))
    return out


def _format_bold(x, bold=False):
    s = f"{x:.3f}"
    return r"$\mathbf{" + s + r"}$" if bold else f"${s}$"


def _patch_table(block: str, label: str, Ns_by_dir: dict[str, int]):
    """Return a new table block with the Mean $|\rho|$ column added."""
    # ---------------------------------------------------------------
    # 1. Strip any previously-injected bottom summary block (old style).
    # ---------------------------------------------------------------
    block = re.sub(
        r"% \[BEGIN auto Mean \|rho\| summary\].*?% \[END auto Mean \|rho\| summary\]\n?",
        "",
        block,
        flags=re.S,
    )

    # ---------------------------------------------------------------
    # 2. Strip any previously-injected right-column Mean markers so the
    #    operation is idempotent.
    # ---------------------------------------------------------------
    #  (We'll do this by rebuilding header + data rows from scratch after
    #   extracting the per-cell numbers.)

    # ---------------------------------------------------------------
    # 3. Extract the data rows.
    # ---------------------------------------------------------------
    rows = []  # list of (mlabel, variant, [12 floats])
    for mlabel in METRIC_LABELS:
        pat_full = re.compile(
            re.escape(mlabel) + r"\s*&\s*full\s*&\s*(.+?)\\\\",
            re.S,
        )
        pat_re = re.compile(
            r"&\s*right-of-\\texttt\{==\}\s*&\s*(.+?)\\\\",
            re.S,
        )
        m_full = pat_full.search(block)
        if not m_full:
            raise RuntimeError(f"{label}: couldn't find {mlabel} 'full' row")
        # Strip trailing injected Mean cell if the row has >12 numeric cells.
        full_body = m_full.group(1)
        full_cells = full_body.split("&")
        full_vals_all = _parse_floats(full_cells)
        if len(full_vals_all) < 12:
            raise RuntimeError(
                f"{label}: {mlabel} full row has {len(full_vals_all)} floats, want >=12"
            )
        # Take the FIRST 12 (benchmark cells). Any trailing cell is a prior
        # injected Mean, which we discard.
        full_vals = full_vals_all[:12]

        m_re = pat_re.search(block, m_full.end())
        if not m_re:
            raise RuntimeError(f"{label}: couldn't find {mlabel} 'right-eq' row")
        re_body = m_re.group(1)
        re_cells = re_body.split("&")
        re_vals_all = _parse_floats(re_cells)
        if len(re_vals_all) < 12:
            raise RuntimeError(
                f"{label}: {mlabel} re row has {len(re_vals_all)} floats, want >=12"
            )
        re_vals = re_vals_all[:12]

        rows.append((mlabel, "full",     full_vals))
        rows.append((mlabel, "right-eq", re_vals))

    # ---------------------------------------------------------------
    # 4. Compute per-row Fisher-z weighted mean |rho|.
    #    Weights: inverse variance of z, proportional to (N - 3).
    # ---------------------------------------------------------------
    Ns = [Ns_by_dir.get(d, 0) for d in CELL_ORDER]
    missing = [CELL_ORDER[i] for i, n in enumerate(Ns) if n <= 3]
    if missing:
        raise RuntimeError(
            f"{label}: missing or too-small N for cells: {missing}. "
            "Run recompute_tables.py first to refresh paper/recompute_audit.log."
        )
    means = {}
    for mlabel, variant, vals in rows:
        abs_vals = [abs(v) for v in vals]
        means[(mlabel, variant)] = fisher_z_weighted_mean(abs_vals, Ns)

    # ---------------------------------------------------------------
    # 4b. Per-column bolding: for each of the 12 benchmark columns, find
    # the single (metric, variant) row with the largest |rho| and record
    # that as the bold position. For the Mean |rho| column, bold the
    # single highest mean across the 8 rows.
    # ---------------------------------------------------------------
    # rows_by_idx[row_idx] = (mlabel, variant, vals)
    rows_by_idx = list(rows)  # 8 entries
    # For each benchmark column c in 0..11, pick the row index with max |rho[c]|.
    bold_cell = {}  # (row_idx, col) -> True
    for c in range(12):
        best_i = max(
            range(len(rows_by_idx)),
            key=lambda i: abs(rows_by_idx[i][2][c]),
        )
        bold_cell[(best_i, c)] = True
    # Bold position for the Mean column.
    best_mean_i = max(
        range(len(rows_by_idx)),
        key=lambda i: means[(rows_by_idx[i][0], rows_by_idx[i][1])],
    )

    # ---------------------------------------------------------------
    # 5. Patch the tabular preamble: add a trailing "r" column.
    # ---------------------------------------------------------------
    preamble_pat = re.compile(r"\\begin\{tabular\}\{(l l c c c \| c c c \| c c c \| c c c)\}", re.S)
    # The actual preamble has no spaces, but let's just match whatever is there.
    def _repreamble(m):
        inner = m.group(1)
        if inner.endswith("r"):
            return m.group(0)  # already patched
        return r"\begin{tabular}{" + inner + "r}"
    # First, strip spaces for robust matching:
    cur_pre_match = re.search(r"\\begin\{tabular\}\{([^}]+)\}", block)
    if cur_pre_match is None:
        raise RuntimeError(f"{label}: no tabular preamble found")
    inner = cur_pre_match.group(1)
    # Drop any trailing r we might have added before.
    inner_clean = inner.rstrip()
    if inner_clean.endswith("r"):
        inner_clean = inner_clean[:-1].rstrip()
    new_inner = inner_clean + "r"
    block = block.replace(cur_pre_match.group(0), r"\begin{tabular}{" + new_inner + "}", 1)

    # ---------------------------------------------------------------
    # 6. Patch the model band row: extend the trailing \multicolumn{3}{c}{...}
    #    group with a trailing empty cell to span the new column, OR add
    #    an empty cell. The simplest valid approach: append " & " before \\
    #    on that row, which gives an empty column header. But we want the
    #    model group over the 12 benchmark columns and the Mean label over
    #    the new 1 column. Let's just append "& \multicolumn{1}{c}{Mean $|\rho|$}"
    #    at the end of the model band, but then the header row below needs
    #    a Mean entry too. Cleaner: leave the model band alone and append
    #    one empty cell; put the Mean label on the per-metric header row.
    # ---------------------------------------------------------------
    # Find the model band row (it's the one with multiple \multicolumn{3}{c}{...} entries ending with \\).
    model_band_pat = re.compile(
        r"(& \\multicolumn\{3\}\{c\|\}\{Qwen3-Coder-480B\}.*?\\multicolumn\{3\}\{c\}\{DeepSeek-V4-Flash\})(.*?)(\\\\)",
        re.S,
    )
    def _repl_band(m):
        head, between, end = m.group(1), m.group(2), m.group(3)
        # Only add if not already there.
        if "Mean" in between:
            return m.group(0)
        return head + r" & \multicolumn{1}{c}{Mean}" + between + end
    block = model_band_pat.sub(_repl_band, block, count=1)

    # Find the per-metric header row ("Metric & Variant & HE & MP & CE & ... \\")
    # Add a trailing "& $|\rho|$" to it.
    hdr_pat = re.compile(
        r"(Metric\s*&\s*Variant\s*&(?:[^\\]|\\(?!\\))*?)(\\\\)",
        re.S,
    )
    def _repl_hdr(m):
        head, end = m.group(1), m.group(2)
        if r"|\rho|" in head or "$|\\rho|$" in head:
            return m.group(0)
        return head.rstrip() + r" & $|\rho|$ " + end
    block = hdr_pat.sub(_repl_hdr, block, count=1)

    # ---------------------------------------------------------------
    # 7. Rewrite each data row: emit 12 benchmark cells (bolded only when
    #    this row has the max |rho| in that column) + the Mean cell
    #    (bolded only if this row has the max Mean across all 8 rows).
    # ---------------------------------------------------------------
    def _format_val(v: float, bold: bool) -> str:
        s = f"{v:+.3f}"
        return r"$\mathbf{" + s + r"}$" if bold else f"${s}$"

    def _build_row_body(row_i: int, vals: list) -> str:
        parts = []
        for c in range(12):
            parts.append(_format_val(vals[c], bold=(row_i, c) in bold_cell))
        mean_val = means[(rows_by_idx[row_i][0], rows_by_idx[row_i][1])]
        parts.append(_format_bold(mean_val, bold=(row_i == best_mean_i)))
        return " " + " & ".join(parts) + " "

    for row_i, (mlabel, variant, vals) in enumerate(rows_by_idx):
        new_body = _build_row_body(row_i, vals)
        if variant == "full":
            pat = re.compile(
                re.escape(mlabel) + r"(\s*&\s*full\s*&)(.+?)(\\\\)",
                re.S,
            )
            m = pat.search(block)
            if m is None:
                raise RuntimeError(f"missing full row for {mlabel}")
            block = block[:m.start(2)] + new_body + block[m.end(2):]
        else:
            full_marker = re.search(
                re.escape(mlabel) + r"\s*&\s*full\s*&.+?\\\\",
                block,
                re.S,
            )
            if full_marker is None:
                raise RuntimeError(f"missing full anchor for {mlabel}")
            tail = block[full_marker.end():]
            re_pat = re.compile(
                r"(&\s*right-of-\\texttt\{==\}\s*&)(.+?)(\\\\)",
                re.S,
            )
            mre = re_pat.search(tail)
            if mre is None:
                raise RuntimeError(f"missing re row for {mlabel}")
            abs_start = full_marker.end() + mre.start(2)
            abs_end = full_marker.end() + mre.end(2)
            block = block[:abs_start] + new_body + block[abs_end:]

    return block, means


def _slice_table(txt: str, label: str) -> tuple[int, int]:
    """Return [start, end) of the \\begin{table*}...\\end{table*} block
    that contains the given label."""
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

    for label, n_regex in TABLES:
        Ns_by_dir = load_Ns_for_table(n_regex)
        s, e = _slice_table(txt, label)
        new_block, means = _patch_table(txt[s:e], label, Ns_by_dir)
        emit(f"\n=== {label}  (Fisher-z weighted) ===")
        for variant in ("full", "right-eq"):
            emit(f"  {variant}")
            for m in METRIC_LABELS:
                emit(f"    {m}: {means[(m, variant)]:.4f}")
        txt = txt[:s] + new_block + txt[e:]

    TEX.write_text(txt)
    emit(f"\nWrote {TEX}  ({len(txt)} chars)")

    audit = ROOT / "paper" / "logs" / "add_mean_rho_rows.log"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("\n".join(log) + "\n")
    print(f"Wrote {audit}", file=sys.stderr)


if __name__ == "__main__":
    main()
