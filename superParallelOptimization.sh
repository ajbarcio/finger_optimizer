#!/usr/bin/env bash
# superParallelOptimization.sh

set -eo pipefail

SEEDS=(
    "finger_designs/NPFWORKS_design_3dof_npf_20261119-1350.json"
    "finger_designs/NPFWORKS_THIN_design_3dof_npf_20261119-1641.json"
    "finger_designs/PFWORKS_design_3dof_pf_20261124-1410.json"
)

OBJECTIVES=(
    "thickness"
    "minimize_volume"
    "maximize_volume"
    "bend_radius"
)

SHARED_FLAGS=(
    "--design-lengths"
    "--maxiter" "500"
    "--method" "trust-constr"
)

PYTHON=".venv/bin/python"
SCRIPT="batchOptimizer.py"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TOTAL_CORES=$(nproc)
N_OBJECTIVES=${#OBJECTIVES[@]}
# Each objective gets an equal share — both PF and NPF runs for that
# objective share the same worker budget since they run in series
WORKERS_PER_RUN=$(( TOTAL_CORES / N_OBJECTIVES ))

echo "Launching ${N_OBJECTIVES} objectives in parallel"
echo "  ${WORKERS_PER_RUN} workers each (${TOTAL_CORES} total cores)"
echo "  Each objective runs PF first, then NPF in series"

# One function per objective — runs PF then NPF in series
run_objective() {
    local obj=$1
    local workers=$2

    # ── Run 1: enforce point forces ───────────────────────────────────────
    local log_pf="${LOG_DIR}/PF_${obj}.log"
    echo "  [${obj}] Starting PF run → ${log_pf}"
    $PYTHON -u $SCRIPT \
        "${SEEDS[@]}" \
        -o "$obj" \
        -w "$workers" \
        --enforce-pf \
        "${SHARED_FLAGS[@]}" \
        > "$log_pf" 2>&1
    echo "  [${obj}] PF run complete"

    # ── Run 2: no point forces ────────────────────────────────────────────
    local log_npf="${LOG_DIR}/NPF_${obj}.log"
    echo "  [${obj}] Starting NPF run → ${log_npf}"
    $PYTHON -u $SCRIPT \
        "${SEEDS[@]}" \
        -o "$obj" \
        -w "$workers" \
        --no-enforce-pf \
        "${SHARED_FLAGS[@]}" \
        > "$log_npf" 2>&1
    echo "  [${obj}] NPF run complete"
}

export -f run_objective
export PYTHON SCRIPT SEEDS SHARED_FLAGS LOG_DIR

# Launch one background job per objective
pids=()

cleanup() {
    echo "Caught signal — killing all child processes..."
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait
    echo "All processes terminated."
    exit 1
}
trap cleanup SIGINT SIGTERM

for obj in "${OBJECTIVES[@]}"; do
    run_objective "$obj" "$WORKERS_PER_RUN" &
    pids+=($!)
    echo "  Launched objective: $obj (pid=${pids[-1]})"
done

# Wait for all and report
all_ok=true
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    obj=${OBJECTIVES[$i]}
    if wait "$pid"; then
        echo "  ✓ ${obj} complete (PF + NPF)"
    else
        echo "  ✗ ${obj} FAILED (exit code $?)"
        all_ok=false
    fi
done

for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
done

$all_ok && echo "All objectives complete." || echo "Some objectives failed."