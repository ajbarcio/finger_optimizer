# recover_results.py

# THIS IS A TEMPRYARY SCRIPT THAT SHOULD BE CHANGED AS NECESSARY TO RECOVER
# WHATEVER RUN YOU LAST COCKED UP

import numpy as np
import json
import re
import glob
from pathlib import Path
from dataclasses import asdict

import finger_config as cfg
from finger_config import vector_to_joint_params

LOG_DIR    = "logs"
DESIGN_LENGTHS = True   # match your run flags
ENFORCE_PF     = True   # match your run flags -- used only for filename

def parse_log(log_path: str) -> dict[str, np.ndarray]:
    """
    Parse a log file and return {label: final_x_vector} for each seed
    that appears in the log.
    Returns the LAST x vector seen for each label — that's the final state.
    """
    # Match lines like:
    # [LABEL] iter=N t=... obj=... ... x=[v0,v1,...,vN]
    step_pattern = re.compile(
        r'\[([^\]]+)\].*?x=\[([^\]]+)\]',
        flags=re.MULTILINE
    )
    # re.finditer()
    # print(f'looking for: {step_pattern}')
    results = {}
    steps = {}
    with open(log_path) as f:
        content = f.read()
        matches = step_pattern.findall(content)
        for m in matches:
            label = m[0].strip().replace('\n','')
            x_str = m[-1].strip().replace('\n','')
            try:
                x = np.fromstring(x_str, sep=',')
                results[label]=x
                steps[label] = steps.get(label, 0) + 1
            except Exception as e:
                print(f"  bish thats some bullshit: {e}")
        # for line in f:
        #     m = step_pattern.search(line)
        #     if m:
        #         label  = m.group(1).strip()
        #         x_str  = m.group(2).strip()
        #         try:
        #             x = np.fromstring(x_str, sep=',')
        #             results[label] = x   # keeps overwriting → last line wins
        #         except Exception as e:
        #             print(f"  Warning: could not parse x in line: {line[:80]} ({e})")
        #     else:
        #         print('Didnt parse line')
    return results, steps


def save_results(label: str, objective: str, x: np.ndarray,
                 enforce_pf: bool, design_lengths: bool):
    pf_tag  = "PF" if enforce_pf else ""
    smx_path = f"batch_result_{label}_{objective}_{pf_tag}.smx"
    json_path = f"batch_result_{label}_{objective}.json"

    # Save .smx
    np.savetxt(smx_path, x)
    print(f"  Saved {smx_path}")

    # Save .json
    try:
        jps = vector_to_joint_params(x, design_lengths=design_lengths)
        with open(json_path, 'w') as f:
            json.dump([asdict(jp) for jp in jps], f, indent=2)
        print(f"  Saved {json_path}")
        # Quick sanity check
        print(f"    x = {np.array2string(x, precision=4, suppress_small=True)}")
        print(f"    thickness (computed) = "
              f"{cfg.compute_rich_thickness(jps, cfg.build_finger_from_params(jps))[0]:.4f}")
    except Exception as e:
        print(f"  WARNING: json save failed for {label}: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    # Discover all log files
    log_files = sorted(glob.glob(f"{LOG_DIR}/PF_*.log"))
    if not log_files:
        print(f"No log files found in {LOG_DIR}/")
        exit(1)

    print(f"Found {len(log_files)} log files:")
    for lf in log_files:
        print(f"  {lf}")
    print()

    total_saved = 0
    for log_path in log_files:
        # Extract objective from filename: logs/PF_{objective}.log
        log_stem  = Path(log_path).stem          # e.g. "PF_minimize_volume"
        objective = log_stem[len("PF_"):]        # e.g. "minimize_volume"

        print(f"\n── {log_path} (objective={objective}) ─────────────────")
        final_vectors, steps = parse_log(log_path)

        if not final_vectors:
            print(f"  WARNING: no STEP lines found in {log_path}")
            continue

        for label, x in final_vectors.items():
            print(f"\n  [{label}]  n={steps[label]} iters found")
            if len(x) != cfg.n_elements(DESIGN_LENGTHS):
                print(f"  WARNING: expected {cfg.n_elements(DESIGN_LENGTHS)} elements, "
                      f"got {len(x)} — skipping")
                continue
            save_results(label, objective, x, ENFORCE_PF, DESIGN_LENGTHS)
            total_saved += 1

    print(f"\n{'═'*55}")
    print(f"  Done — saved {total_saved} result pairs (.smx + .json)")
    print(f"{'═'*55}")