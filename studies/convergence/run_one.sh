#!/usr/bin/env bash
# Run one BETSE config through init, sim and export. Usage: run_one.sh <name>
cd "$(dirname "$0")"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
B=${BETSE:-betse}
n="$1"
mkdir -p logs
# BETSE on Windows insists on a backslash in the log path.
logfile='logs\'"$n"'.betse.log'
{
  date
  for phase in "seed" "init" "sim" "plot init" "plot sim"; do
    echo "== $phase"
    $B --headless --log-file "$logfile" $phase "$n.yaml" 2>&1 \
      | grep -v '^[[:space:]]' \
      | grep -i 'contains\|approx\|completed\|exported in\|error\|exception'
  done
  date
} > "logs/$n.out" 2>&1
touch "logs/$n.done"
