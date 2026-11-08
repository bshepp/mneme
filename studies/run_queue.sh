#!/usr/bin/env bash
# Keep up to MAX_JOBS BETSE runs going from a queue file.
# usage: run_queue.sh <queue_file> <run_dir> [max_jobs]
# Each queue line is  <name>|<PHASES>|<SKIP_SEED>  (the last two may be empty),
# e.g.  pert_A_ctrl|sim,plot sim|1   or   settle_A||1
# Lines are taken from the top of the file, so prepend urgent runs. The file
# may be edited while this runs. Runs already in progress (a logs/<name>.out
# without its logs/<name>.done) count toward the limit. Stops when the queue is empty and
# nothing is running; a line "STOP" stops it at once.
set -u
queue="$1"; run_dir="$2"; max=${3:-4}
here="$(cd "$(dirname "$0")" && pwd)"
while true; do
  # A run is in progress when run_one.sh has written logs/<name>.out but
  # not yet logs/<name>.done. (Counting processes over-counts: the pipeline
  # inside run_one.sh forks copies of the script.)
  running=$(( $(ls "$run_dir"/logs/*.out 2>/dev/null | wc -l) - $(ls "$run_dir"/logs/*.done 2>/dev/null | wc -l) ))
  if [ -s "$queue" ] && [ "$running" -lt "$max" ]; then
    line=$(head -n 1 "$queue")
    tail -n +2 "$queue" > "$queue.tmp" && mv "$queue.tmp" "$queue"
    [ "$line" = "STOP" ] && exit 0
    [ -z "$line" ] && continue
    IFS='|' read -r name phases skip <<< "$line"
    echo "$(date '+%F %T') start $name"
    PHASES="$phases" SKIP_SEED="$skip" nohup bash "$here/convergence/run_one.sh" "$name" "$run_dir" \
      > /dev/null 2>&1 &
    sleep 2
    continue
  fi
  if [ ! -s "$queue" ] && [ "$running" -eq 0 ]; then
    echo "$(date '+%F %T') queue empty, nothing running"; exit 0
  fi
  sleep 30
done
