#!/bin/bash
# Phase 1 — baseline generation.
#
# Produces data/<llm>/<bench>/results.jsonl: one generated test suite per
# benchmark task, with per-token logprobs (top-20) and smell / coverage /
# mutation metrics attached. This is the input to Phase 2 (analyzer loop) and
# Phase 3 (hit-rate rerun).
#
# Set OPENAI_API_KEY (and OPENAI_BASE_URL if you use a non-OpenAI endpoint)
#
# Benches: humaneval (164 tasks), mbpp_plus (378), classeval (100).
# At temp=0.0 with 1 sample each, one LLM × 3 benches runs in ~1-2 h.

set -e
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../../.venv/bin/activate"

LOG=/tmp/baseline_$(date +%Y%m%d_%H%M%S).log
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] baseline generation started ===" > $LOG

# (display_llm, openai_model_id)
# Edit this list for the models your endpoint serves. The ids below are the
# four Hugging Face model names from the original paper; replace with
# whatever your OpenAI-compatible endpoint accepts.
MODELS=(
  "qwen        Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
  "gpt_oss     openai/gpt-oss-120b"
  "glm52       zai-org/GLM-5.2-FP8"
  "deepseek_v4 deepseek-ai/DeepSeek-V4-Flash"
)

run_cell() {
  local LLM_DIR="$1" MODEL="$2" BENCH="$3" OUT_DIR="$4"
  mkdir -p "$OUT_DIR"
  echo "" >> $LOG
  echo "--- [$(date +%H:%M:%S)] $LLM_DIR × $BENCH  model=$MODEL ---" >> $LOG
  python src/run_bench.py \
    --model "$MODEL" \
    --bench "$BENCH" \
    --n-tasks 10000 \
    --samples 1 \
    --temps 0.0 \
    --out-dir "$OUT_DIR" >> $LOG 2>&1
}

for ENTRY in "${MODELS[@]}"; do
  LLM_DIR=$(echo $ENTRY | awk '{print $1}')
  MODEL=$(echo $ENTRY | awk '{print $2}')
  run_cell "$LLM_DIR" "$MODEL" humaneval  "data/$LLM_DIR/humaneval"
  run_cell "$LLM_DIR" "$MODEL" mbpp_plus  "data/$LLM_DIR/mbpp_plus"
  run_cell "$LLM_DIR" "$MODEL" classeval  "data/$LLM_DIR/classeval"
done

echo "" >> $LOG
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] baseline done ===" >> $LOG
echo "see $LOG"
