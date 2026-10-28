"""LLM client loader (OpenAI-compatible endpoint).

Exports:
  - ``client`` / ``async_client`` — the sync + async chat.completions clients
    used by src/bench_pipeline.py and src/analyzer_loop.py. Both call
    ``create(..., stream=True, logprobs=True, top_logprobs=TOP_LOGPROBS)``
    at runtime.
  - ``MODELS`` — list of model ids you want to benchmark. The chain scripts
    pass a specific one via ``--model``; this list is just a default pool.
  - ``TOP_LOGPROBS`` — the API's logprobs-per-position cap (OpenAI = 20).

Configuration via environment variables:
  OPENAI_API_KEY  — your API key (required by the openai client; use any
                    non-empty placeholder if your endpoint doesn't check it).
  OPENAI_BASE_URL — e.g. "https://api.openai.com/v1" (default) or your own
                    vLLM / Together / Groq / Fireworks endpoint. Any
                    OpenAI-compatible chat/completions API works.

If your LLM endpoint is NOT OpenAI-compatible, replace this file with a
shim that exposes the same ``client`` and ``async_client`` interface
(a ``chat.completions.create()`` method that streams per-token logprobs).
"""
import os

from openai import OpenAI, AsyncOpenAI

MODELS = [
    # Replace with your own model ids. These are the Hugging Face model
    # names from the original paper; swap for whatever your endpoint
    # serves (e.g. "gpt-4o-mini", "meta-llama/Llama-3.1-70B-Instruct", ...).
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "openai/gpt-oss-120b",
    "zai-org/GLM-5.2-FP8",
    "deepseek-ai/DeepSeek-V4-Flash",
]

TOP_LOGPROBS = 20  # OpenAI API cap

_api_key = os.environ.get("OPENAI_API_KEY") or "placeholder"
_base_url = os.environ.get("OPENAI_BASE_URL")  # None → default api.openai.com

_kwargs = {"api_key": _api_key}
if _base_url:
    _kwargs["base_url"] = _base_url

client = OpenAI(**_kwargs)
async_client = AsyncOpenAI(**_kwargs)
