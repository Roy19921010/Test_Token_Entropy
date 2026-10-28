#!/bin/bash
# Phase 3 — hit-rate enrichment.
#
# Reads: data/<llm>/<bench>/traces/traces_{full|righteq}_K{K}_optout_5iter*.jsonl
# Writes: data/<llm>/<bench>/traces_noguard_hr/<same filename>
#
# For each iteration i, reruns pytest on iter (i-1)'s code to recover the
# per-testbody FAIL/ERROR set, then compares it to the K flagged testbodies
# in iter i's hot_assertions and attaches hit_rate / n_hits / fix_rate columns.
#
# Serial across all 24 files (12 cells × 2 variants) to avoid pytest process
# explosion. ~1-2 h total.

set -e
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../../.venv/bin/activate"

LOG=/tmp/hitrate_noguard_log.txt
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] noguard hit-rate started ===" > $LOG

run_hr() {
  local IN=$1 OUT=$2
  echo "" >> $LOG; echo "--- [$(date +%H:%M:%S)] $IN ---" >> $LOG
  mkdir -p "$(dirname "$OUT")"
  python src/compute_hit_rate.py --in "$IN" --out "$OUT" --resume >> $LOG 2>&1
}

for CELL_K in \
  "data/qwen/humaneval         3" \
  "data/qwen/mbpp_plus         2" \
  "data/qwen/classeval         3" \
  "data/gpt_oss/humaneval      3" \
  "data/gpt_oss/mbpp_plus      2" \
  "data/gpt_oss/classeval      3" \
  "data/glm52/humaneval        3" \
  "data/glm52/mbpp_plus        2" \
  "data/glm52/classeval        3" \
  "data/deepseek_v4/humaneval  2" \
  "data/deepseek_v4/mbpp_plus  2" \
  "data/deepseek_v4/classeval  4"; do
  CELL=$(echo $CELL_K | awk '{print $1}')
  K=$(echo $CELL_K | awk '{print $2}')
  # Full-testbody variant (nostopos)
  IN="$CELL/traces/traces_full_K${K}_optout_5iter_nostopos.jsonl"
  OUT="$CELL/traces_noguard_hr/traces_full_K${K}_optout_5iter_nostopos.jsonl"
  [ -f "$IN" ] && run_hr "$IN" "$OUT"
  # Right-of-== variant
  IN="$CELL/traces/traces_righteq_K${K}_optout_5iter.jsonl"
  OUT="$CELL/traces_noguard_hr/traces_righteq_K${K}_optout_5iter.jsonl"
  [ -f "$IN" ] && run_hr "$IN" "$OUT"
done

echo "" >> $LOG
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] hit-rate done ===" >> $LOG
echo "Done. Tail $LOG for progress."
