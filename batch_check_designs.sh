#!/bin/bash

set -e

DESIGNS_DIR="finger_designs"

PYTHON_FLAGS=(
    "-u"
    "-W" "ignore"
)

SUCCESS="HOUSTON THIS ONE IS ALL GOOD"
FAILURE="HOUSTON WE GOT A PROBLEM WITH THIS ONE"

PYTHON=".venv/bin/python"
SCRIPT="interactive_finger_designer.py"

for file in "${DESIGNS_DIR}"/*; do
    [ -f "$file" ] || continue
    SCRIPT_FLAGS=(
    "-ck"
    )
    if [[ ! "${file}" == *"npf"* ]]; then
        SCRIPT_FLAGS+=("-pf")
        echo "Enforcing PF"
    else
        echo "NOT Enforcing PF"
    fi
    output=$($PYTHON "${PYTHON_FLAGS[@]}" "${SCRIPT}" "${SCRIPT_FLAGS[@]}" \
             -f "${file}")
    if [[ "$output" == *"$SUCCESS"* ]]; then
        echo $file $SUCCESS
    elif [[ "$output" == *"$FAILURE"* ]]; then
        echo $file $FAILURE
    else
        echo "  [DEBUG] You fool! you have failed to enter the condition"
        echo $output
    fi
done