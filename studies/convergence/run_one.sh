#!/usr/bin/env bash
# Run one BETSE config through seed, init, sim and export.
# usage: run_one.sh <name> [run_dir]
# <run_dir> holds <name>.yaml; it defaults to the directory of this script.
# SKIP_SEED=1 when the runs share one pre-seeded world.
# PHASES="sim,plot sim" to run only some phases (comma-separated), for a
#   config whose init file was written by other means.
# BETSE=<path> to use a BETSE other than the one on PATH.
set -u
cd "${2:-$(dirname "$0")}" || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
B=${BETSE:-betse}
n="$1"
mkdir -p logs
# BETSE on Windows insists on a backslash in the log path.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) logfile='logs\'"$n"'.betse.log' ;;
  *) logfile="logs/$n.betse.log" ;;
esac
{
  date
  phases=("seed" "init" "sim" "plot init" "plot sim")
  [ -n "${SKIP_SEED:-}" ] && phases=("init" "sim" "plot init" "plot sim")
  if [ -n "${PHASES:-}" ]; then IFS=',' read -r -a phases <<< "$PHASES"; fi
  for phase in "${phases[@]}"; do
    echo "== $phase"
    # shellcheck disable=SC2086
    $B --headless --log-file "$logfile" $phase "$n.yaml" 2>&1 \
      | grep -v '^[[:space:]]' \
      | grep -i 'contains\|approx\|completed\|exported in\|error\|exception'
  done
  date
} > "logs/$n.out" 2>&1
touch "logs/$n.done"
