#!/bin/bash
cd ~/compute/hatch-v5/research_v5/stacker
/usr/bin/time -l ~/compute/hatch-v5/.venv/bin/python -u inference_v6.py --inputs ~/compute/hatch-v5/artifacts/validation-inputs.json --data-root ~/compute/hatch-v5/dataset --output-dir ~/compute/hatch-v5/artifacts/validation-v6/predictions --threshold 0.14 > ~/compute/hatch-v5/artifacts/validation-v6/inference.log 2>&1 &
pid=$!
echo $pid > ~/compute/hatch-v5/artifacts/validation-v6/pid
while kill -0 $pid 2>/dev/null; do
  used=$(sysctl -n vm.swapusage | sed -E "s/.*used = ([0-9.]+)M.*/\1/")
  if awk -v u="$used" "BEGIN{exit !(u > 3500)}"; then echo "watchdog: swap used ${used}M, stopping v6 $(date)" >> ~/compute/hatch-v5/artifacts/validation-v6/inference.log; pkill -P $pid; kill $pid; fi
  sleep 15
done
echo "v6 finished $(date)" >> ~/compute/hatch-v5/artifacts/validation-v6/inference.log
