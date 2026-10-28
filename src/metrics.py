"""Metrics derived from OpenAI-compatible per-token logprobs.

Input format (LangChain surfaces this at msg.response_metadata["logprobs"]["content"]):
    [
        {
            "token": "hello",
            "logprob": -0.12,
            "top_logprobs": [{"token": "...", "logprob": -0.12}, ...],
        },
        ...
    ]
"""
import math
import re
from typing import Iterable


# ---- per-token primitives -----------------------------------------------------

def sequence_perplexity(entries: Iterable[dict]) -> float | None:
    lps = [e["logprob"] for e in entries]
    if not lps:
        return None
    return math.exp(-sum(lps) / len(lps))


def token_entropies(entries: Iterable[dict]) -> list[float]:
    """Truncated entropy: -Σ p_i log p_i over the top-k tokens returned by the API.

    Not the true entropy (missing mass beyond top-k), but a useful proxy.
    """
    out = []
    for e in entries:
        ps = [math.exp(t["logprob"]) for t in e.get("top_logprobs", [])]
        out.append(-sum(p * math.log(p) for p in ps if p > 0))
    return out


def chosen_token_probs(entries: Iterable[dict]) -> list[float]:
    return [math.exp(e["logprob"]) for e in entries]


def chosen_token_ranks(entries: Iterable[dict], top_k: int = 20) -> list[int]:
    """Per-token rank of the chosen token within its own top-k distribution.

    Returns 1-indexed rank (1 = top-1 was chosen). If the chosen token doesn't
    appear in the top-k list, the rank is censored at `top_k + 1` — we can't
    tell exactly how far down it was, only that it was beyond our window.
    Same-token duplicates in `top_logprobs` are compared by exact token match
    on the first occurrence.
    """
    out = []
    for e in entries:
        chosen = e.get("token", "")
        tops = e.get("top_logprobs") or []
        rank = None
        for i, t in enumerate(tops):
            if t.get("token") == chosen:
                rank = i + 1
                break
        out.append(rank if rank is not None else top_k + 1)
    return out


def top1_gaps(entries: Iterable[dict]) -> list[float]:
    """Per-token confidence margin: p(top-1) - p(top-2).

    Large gap → the model was confident in its choice over the next-best.
    Small gap → the model was torn between top-2 candidates (a different signal
    from entropy, which spreads across the whole top-k)."""
    out = []
    for e in entries:
        tops = e.get("top_logprobs") or []
        if len(tops) >= 2:
            p1 = math.exp(tops[0]["logprob"])
            p2 = math.exp(tops[1]["logprob"])
            out.append(p1 - p2)
        elif len(tops) == 1:
            out.append(math.exp(tops[0]["logprob"]))
        else:
            out.append(0.0)
    return out


# ---- windowed helpers ---------------------------------------------------------

def rolling_max(values: list[float], window: int) -> list[float]:
    """Max of `values` over a sliding window of `window` samples. Output length
    is len(values) - window + 1 (returns [] if input too short)."""
    if len(values) < window:
        return []
    return [max(values[i:i + window]) for i in range(len(values) - window + 1)]


# ---- positional / region-based ------------------------------------------------

# Token indices that fall inside `def test_*` blocks vs. import lines vs. asserts.
# We work from reconstructed text (join of `token` fields) + a running char->token
# map, so this is BPE-agnostic.

_ASSERT_RE = re.compile(r"\bassert\b")
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+", re.MULTILINE)
_DEF_TEST_RE = re.compile(r"^\s*def\s+test_", re.MULTILINE)


def _char_offsets(entries: list[dict]) -> tuple[str, list[int]]:
    """Reconstruct text and return (text, token_start_offsets)."""
    text_parts: list[str] = []
    offsets: list[int] = []
    pos = 0
    for e in entries:
        offsets.append(pos)
        tok = e.get("token", "")
        text_parts.append(tok)
        pos += len(tok)
    return "".join(text_parts), offsets


def _token_indices_matching(entries: list[dict], predicate) -> list[int]:
    """Return indices of tokens whose start-offset satisfies `predicate(char_idx)`."""
    text, offsets = _char_offsets(entries)
    return [i for i, off in enumerate(offsets) if predicate(text, off)]


def _region_mask_import(text: str, char_idx: int) -> bool:
    """char_idx is inside an import statement — same line starts with `import` or `from`."""
    line_start = text.rfind("\n", 0, char_idx) + 1
    line_end = text.find("\n", char_idx)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    return bool(re.match(r"\s*(?:import|from)\s+", line))


def _region_mask_assert(text: str, char_idx: int) -> bool:
    """char_idx is on a line containing an `assert` keyword."""
    line_start = text.rfind("\n", 0, char_idx) + 1
    line_end = text.find("\n", char_idx)
    if line_end == -1:
        line_end = len(text)
    return bool(_ASSERT_RE.search(text[line_start:line_end]))


def _region_mask_test_body(text: str, char_idx: int) -> bool:
    """char_idx is between `def test_...` and the next top-level `def` / EOF."""
    upto = text[:char_idx]
    starts = list(_DEF_TEST_RE.finditer(upto))
    if not starts:
        return False
    last_test = starts[-1].start()
    # find any non-test top-level def after last_test but before char_idx
    tail = text[last_test:char_idx]
    later_defs = [m for m in re.finditer(r"^\s*def\s+(?!test_)", tail, re.MULTILINE)
                  if m.start() > 0]
    return not later_defs


def region_metrics(entries: list[dict]) -> dict:
    """Entropy aggregates restricted to certain code regions."""
    entries = list(entries)
    if not entries:
        return {f"import_{k}": None for k in ("mean_entropy", "max_entropy", "n_tokens")} | \
               {f"assert_{k}": None for k in ("mean_entropy", "max_entropy", "n_tokens")} | \
               {f"testbody_{k}": None for k in ("mean_entropy", "max_entropy", "n_tokens")}

    ents = token_entropies(entries)
    out: dict = {}
    for region_name, mask_fn in (
        ("import", _region_mask_import),
        ("assert", _region_mask_assert),
        ("testbody", _region_mask_test_body),
    ):
        idxs = _token_indices_matching(entries, mask_fn)
        region_ents = [ents[i] for i in idxs]
        out[f"{region_name}_n_tokens"] = len(region_ents)
        out[f"{region_name}_mean_entropy"] = (sum(region_ents) / len(region_ents)
                                              if region_ents else None)
        out[f"{region_name}_max_entropy"] = max(region_ents) if region_ents else None
    return out


# ---- sequence-level roll-up ---------------------------------------------------

def aggregate_metrics(entries: Iterable[dict], spike_threshold: float = 2.0,
                      window: int = 16) -> dict:
    """Sequence-level roll-up used by the benchmark.

    Entropy is truncated (bounded by log(top_k)). `max_per_token_ppl` = exp(-min
    logprob) surfaces the single most-uncertain token. Windowed and trajectory
    features smooth over single-token spikes to catch sustained uncertainty.
    """
    entries = list(entries)
    if not entries:
        return _empty_agg()

    lps = [e["logprob"] for e in entries]
    ents = token_entropies(entries)
    gaps = top1_gaps(entries)
    ranks = chosen_token_ranks(entries, top_k=20)
    ents_sorted = sorted(ents)
    p95_idx = int(0.95 * (len(ents_sorted) - 1))

    # log K bound for negentropy (K = TOP_LOGPROBS, OpenAI API cap = 20)
    log_K = math.log(20)

    # Trajectory features on entropy
    mean_ent = sum(ents) / len(ents)
    if len(ents) > 1:
        entropy_var = sum((h - mean_ent) ** 2 for h in ents) / (len(ents) - 1)
        entropy_std = math.sqrt(entropy_var)
        # First vs. second half — is uncertainty rising, falling, or flat?
        mid = len(ents) // 2
        first_half_mean = sum(ents[:mid]) / mid if mid > 0 else 0.0
        second_half_mean = sum(ents[mid:]) / (len(ents) - mid) if mid < len(ents) else 0.0
        entropy_trend = second_half_mean - first_half_mean
    else:
        entropy_var = 0.0
        entropy_std = 0.0
        entropy_trend = 0.0

    # Windowed max entropy
    window_maxes = rolling_max(ents, window) if len(ents) >= window else []
    windowed_max_entropy = max(window_maxes) if window_maxes else max(ents)

    result = {
        "num_tokens": len(entries),
        "perplexity": math.exp(-sum(lps) / len(lps)),
        "max_per_token_ppl": math.exp(-min(lps)),
        "mean_entropy": mean_ent,
        "max_entropy": max(ents),
        "entropy_p95": ents_sorted[p95_idx],
        "n_high_entropy_spikes": sum(1 for h in ents if h > spike_threshold),
        # trajectory features
        "entropy_std": entropy_std,
        "entropy_var": entropy_var,
        "entropy_trend": entropy_trend,
        # windowed
        "windowed_max_entropy": windowed_max_entropy,
        "window_size": window,
        # top-1 gap (higher = more confident)
        "mean_top1_gap": sum(gaps) / len(gaps),
        "min_top1_gap": min(gaps),
        # Paper-standard names ------------------------------------------------
        # Log-p(x): mean chosen log-probability (higher = more confident;
        # equal to -log(perplexity) — reported for citation-friendliness)
        "log_px_mean": sum(lps) / len(lps),
        "log_px_min": min(lps),  # least confident chosen token
        # Rank of chosen token in its own top-20 (1 = greedy pick, 21 = censored)
        "mean_chosen_rank": sum(ranks) / len(ranks),
        "max_chosen_rank": max(ranks),
        "n_rank_censored": sum(1 for r in ranks if r > 20),  # chosen token was outside top-20
        # ATN@K = log(K) - H(top-k) averaged. Higher = more confident (lower entropy).
        # Linear transform of mean_entropy (identical Spearman), included by name.
        "atn_at_k": log_K - mean_ent,
        # MTN@K = log(K) - max_H(top-k). Least confident single token in the top-k sense.
        "mtn_at_k": log_K - max(ents),
    }

    # New: positional / region-based
    result.update(region_metrics(entries))
    return result


def _empty_agg() -> dict:
    keys = [
        "num_tokens", "perplexity", "max_per_token_ppl",
        "mean_entropy", "max_entropy", "entropy_p95", "n_high_entropy_spikes",
        "entropy_std", "entropy_var", "entropy_trend",
        "windowed_max_entropy", "window_size",
        "mean_top1_gap", "min_top1_gap",
        "log_px_mean", "log_px_min",
        "mean_chosen_rank", "max_chosen_rank", "n_rank_censored",
        "atn_at_k", "mtn_at_k",
    ]
    out = {k: None for k in keys}
    out["num_tokens"] = 0
    out["n_high_entropy_spikes"] = 0
    out["window_size"] = 16
    out.update({f"import_{k}": None for k in ("mean_entropy", "max_entropy", "n_tokens")})
    out.update({f"assert_{k}": None for k in ("mean_entropy", "max_entropy", "n_tokens")})
    out.update({f"testbody_{k}": None for k in ("mean_entropy", "max_entropy", "n_tokens")})
    return out
