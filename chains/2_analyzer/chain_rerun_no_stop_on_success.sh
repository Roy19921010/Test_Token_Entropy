#!/bin/bash
# Phase 2 — full-testbody Analyzer rerun (no-stop-on-success).
#
# For each failing-baseline suite, run analyzer_loop with
# --rank testbody_mean_entropy (full-testbody ranking) under
# --no-guard --no-stop-on-success for 5 iterations.
#
# Produces: data/<llm>/<bench>/traces/traces_full_K{K}_optout_5iter_nostopos.jsonl
# Hit-rate columns are added by Phase 3 (chains/3_hit_rate/) into
#   data/<llm>/<bench>/traces_noguard_hr/
#
# Compute budget: 359 suites × 5 iterations = ~1,795 analyzer calls.
# Set OPENAI_API_KEY (and OPENAI_BASE_URL if you use a non-OpenAI endpoint)
# if it expires mid-run.

set -e
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../../.venv/bin/activate"

LOG=/tmp/rerun_no_stop_on_success.log
rm -f $LOG
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] rerun no-stop-on-success started ===" > $LOG

run_cell() {
  local MODEL="$1" DIR="$2" K="$3" IDS_FILE="$4"
  local OUT="$DIR/traces/traces_full_K${K}_optout_5iter_nostopos.jsonl"
  mkdir -p "$DIR/traces"
  echo "" >> $LOG
  echo "--- [$(date +%H:%M:%S)] $DIR K=$K  model=$MODEL  ids=$IDS_FILE ---" >> $LOG
  python src/analyzer_loop.py --in-dir "$DIR" --model "$MODEL" \
    --mode entropy_only_strict_testbody_optout --rank testbody_mean_entropy \
    --top-k-mode fixed --top-k $K \
    --max-iterations 5 --temperature 0.0 \
    --no-guard --no-stop-on-success \
    --only-task-ids-from "$IDS_FILE" \
    --out "$OUT" >> $LOG 2>&1
  echo "DONE $DIR K=$K" >> /tmp/rerun_completed.txt
}

rm -f /tmp/rerun_completed.txt

# Qwen (74 suites)
run_cell "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" "data/qwen/humaneval"  3 "cohorts/qwen_he.txt"
run_cell "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" "data/qwen/mbpp_plus"  2 "cohorts/qwen_mp.txt"
run_cell "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" "data/qwen/classeval"  3 "cohorts/qwen_ce.txt"

# gpt-oss (203 suites — biggest batch)
run_cell "openai/gpt-oss-120b" "data/gpt_oss/humaneval" 3 "cohorts/gptoss_he.txt"
run_cell "openai/gpt-oss-120b" "data/gpt_oss/mbpp_plus" 2 "cohorts/gptoss_mp.txt"
run_cell "openai/gpt-oss-120b" "data/gpt_oss/classeval" 3 "cohorts/gptoss_ce.txt"

# GLM-5.2 (46 suites)
run_cell "zai-org/GLM-5.2-FP8" "data/glm52/humaneval" 3 "cohorts/glm52_he.txt"
run_cell "zai-org/GLM-5.2-FP8" "data/glm52/mbpp_plus" 2 "cohorts/glm52_mp.txt"
run_cell "zai-org/GLM-5.2-FP8" "data/glm52/classeval" 3 "cohorts/glm52_ce.txt"

# DeepSeek-V4 (36 suites)
run_cell "deepseek-ai/DeepSeek-V4-Flash-0731" "data/deepseek_v4/humaneval" 2 "cohorts/deepseek_he.txt"
run_cell "deepseek-ai/DeepSeek-V4-Flash-0731" "data/deepseek_v4/mbpp_plus" 2 "cohorts/deepseek_mp.txt"
run_cell "deepseek-ai/DeepSeek-V4-Flash-0731" "data/deepseek_v4/classeval" 4 "cohorts/deepseek_ce.txt"

echo "" >> $LOG
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] all cells complete ===" >> $LOG
echo "Done. Tail $LOG for progress."
