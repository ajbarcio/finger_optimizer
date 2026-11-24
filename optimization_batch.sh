#!/bin/bash
# optimization_batch.sh
# Usage: ./optimization_batch.sh
# Runs a matrix of seeds × objectives with shared config flags.

set -e   # exit on first error

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS=(
    "finger_designs/NPFWORKS_design_3dof_npf_20261119-1350.json"
    "finger_designs/NPFWORKS_THIN_design_3dof_npf_20261119-1641.json"
    "finger_designs/PFWORKS_BRO_design_3dof_pf_20261119-1704.json"
    # threw in a PF-designed finger for fun
)

OBJECTIVES=(
    "thickness"
    "minimize_volume"
    "worst_case_OVL"
    "bend_radius"
)

# Shared flags for all runs
SHARED_FLAGS=(
    "--no-enforce-pf"       # starting with npf designs, don't enforce pf
    "--contain-human"       # default is true, but explicit is clearer
    "--design-lengths"
    "--relax-thickness"
    "--maxiter" "200"
    "--method" "trust-constr" # gonna try with a gradient method since these
                              # should start out green
)

PYTHON=".venv/bin/python"
SCRIPT="batchOptimizer.py"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ─── Run matrix ───────────────────────────────────────────────────────────────

for seed in "${SEEDS[@]}"; do
    for obj in "${OBJECTIVES[@]}"; do

        # Build a readable label for log files
        seed_file="${seed#*/}"
        seed_stem="${seed_file%.json}"
        label="${seed_stem}__${obj}"
        log_file="${LOG_DIR}/${label}.log"

        touch $log_file

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Seed:      $seed"
        echo "  Objective: $obj"
        echo "  Log:       $log_file"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        $PYTHON $SCRIPT \
            "$seed" \
            -o "$obj" \
            "${SHARED_FLAGS[@]}" \
            2>&1 | tee "$log_file"

        echo ""
        echo "  Done: $label"
        echo ""

    done
done

echo "All runs complete."