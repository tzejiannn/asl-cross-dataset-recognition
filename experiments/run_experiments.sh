#!/usr/bin/env bash
# Trains + evaluates ASL Citizen runs one after another (one GPU job at a time -- two concurrent
# jobs overflowed the 12 GB card with the 42-node graph and crashed the driver once).
# Skips runs whose eval file already exists, so the queue can be restarted safely.
# Checkpoints go to checkpoints/{run}/, training logs to logs/{run}.log, eval results to eval_results/{run}.json.
#
# Usage: bash run_experiments.sh "baseline hands_only ..." "0 1 2"
set -u
cd "$(dirname "$0")"
CONDITIONS=${1:-baseline}
SEEDS=${2:-0}
mkdir -p logs

for c in $CONDITIONS; do
  for s in $SEEDS; do
    run="${c}_s${s}"
    [ -f "eval_results/${run}.json" ] && { echo "$run: done already"; continue; }
    ok=0
    for attempt in 1 2; do  # one retry for transient CUDA errors
      python -u train.py --condition "$c" --seed "$s" --run_name "$run" --trim_hand_span \
        > "logs/${run}.log" 2>&1 && { ok=1; break; }
      echo "$run: training attempt $attempt failed: $(tail -1 "logs/${run}.log")"
    done
    [ $ok = 1 ] || { echo "$run: FAILED twice, stopping queue"; exit 1; }
    echo "$run: $(tail -1 "logs/${run}.log")"
    python evaluate.py --condition "$c" --run_name "$run" 2>&1 | grep -E "top1|Error|Traceback" | sed "s/^/  /"
  done
done
