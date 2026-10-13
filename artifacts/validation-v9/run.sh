#!/bin/bash
cd ~/compute/hatch-v5/research_v5/circles8
/usr/bin/time -l ~/compute/hatch-v5/.venv/bin/python -u inference_v9.py --inputs ~/compute/hatch-v5/artifacts/validation-inputs.json --data-root ~/compute/hatch-v5/dataset --output-dir ~/compute/hatch-v5/artifacts/validation-v9/predictions --threshold 0.24 > ~/compute/hatch-v5/artifacts/validation-v9/inference.log 2>&1 &
pid=$!
base=$(sysctl -n vm.swapusage | sed -E "s/.*used = ([0-9.]+)M.*/\1/")
while kill -0 $pid 2>/dev/null; do
  used=$(sysctl -n vm.swapusage | sed -E "s/.*used = ([0-9.]+)M.*/\1/")
  if awk -v u="$used" -v b="$base" "BEGIN{exit !(u > b + 2500)}"; then echo "watchdog: swap used ${used}M (base ${base}M), stopping v9 $(date)" >> ~/compute/hatch-v5/artifacts/validation-v9/inference.log; pkill -P $pid; kill $pid; fi
  sleep 15
done
echo "v9 finished $(date)" >> ~/compute/hatch-v5/artifacts/validation-v9/inference.log
