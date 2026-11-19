# find_seed.py
import numpy as np
from scipy import optimize
from strucMatrices import secondaryDev
from finger import Grasp
import parallel_evaluator
import statistics

# ─── Mirror config from variableOptimizerGlobalParallel.py ───────────────────

def z_score_from_percentile(percentile):
    return statistics.NormalDist().inv_cdf(percentile / 100.0)

numJoints           = 3
numTendons          = numJoints + 1
numFlexs            = 3
numExts             = 3
numJointParameters  = numFlexs * 2 + numExts * 2
numLengthParameters = 2
DESIGN_LENGTHS      = False
RELAX_THICKNESS     = True
lengths             = [1.866, 1.4, 0.75]
overall_length      = 4.0
thickness_lo        = 0.65
thickness_hi        = 0.65 * 1.2
sec_limit           = thickness_lo - 2.25 / 25.4 - 0.125
TENSION_LIMIT       = 55.0
F                   = np.array([0, 5, 0])

overallNumElements  = numJointParameters
if RELAX_THICKNESS:
    overallNumElements += 1

bounds = (
    [(0.125, 0.5)] * numJointParameters
    + [(thickness_lo, thickness_hi)] * (1 if RELAX_THICKNESS else 0)
)

q_ext = [10 * np.pi / 180] * 3
q_int = [45 * np.pi / 180] + [10 * np.pi / 180] * 2
q_flx = [45 * np.pi / 180] * 2 + [10 * np.pi / 180]

z    = z_score_from_percentile(50)
sig  = [10.86, 2.20, 12.5, 6.13, 20.2, 11.05, 10.47, 2.04, 12.68]
means= [26.3,  6.6,  25.3, 25.25, 7.26, 26.3,  31.06, 7.56, 27.08]
vals = [means[i] + z * sig[i] for i in range(len(sig))]

VC_Dir_Grasps = [
    Grasp([[0]*3, [0]*3, [0,  vals[0]/4.448, 0]], q_ext),
    Grasp([[0]*3, [0]*3, [0, -vals[1]/4.448, 0]], q_ext),
    Grasp([[0]*3, [0]*3, [0,  vals[3]/4.448, 0]], q_int),
    Grasp([[0]*3, [0]*3, [0, -vals[4]/4.448, 0]], q_int),
    Grasp([[0]*3, [0]*3, [0,  vals[6]/4.448, 0]], q_flx),
    Grasp([[0]*3, [0]*3, [0, -vals[7]/4.448, 0]], q_flx),
    Grasp([[0]*3, [0]*3, [vals[8]/4.448, 0,  0]], q_flx),
]

# ─── Init worker globals in main process ─────────────────────────────────────

parallel_evaluator.worker_init(
    secondaryDev.R, secondaryDev.D,
    numFlexs, numExts, numTendons,
    numLengthParameters, DESIGN_LENGTHS,
    lengths, overall_length,
    VC_Dir_Grasps,
)

# ─── Seed generation ─────────────────────────────────────────────────────────

def sample_once(rng, thickness):
    """
    Sample one candidate satisfying all linear constraints by construction.
    Returns None if a valid draw isn't possible for the given thickness.
    """
    v = np.zeros(overallNumElements)

    for i in range(numFlexs):
        flex_max_hi = min(0.5, thickness - 0.125)
        if flex_max_hi < 0.125:
            return None
        flex_max    = rng.uniform(0.125, flex_max_hi)
        flex_min    = rng.uniform(0.125, flex_max)
        v[2 * i]     = flex_max
        v[2 * i + 1] = flex_min

    for i in range(numExts):
        paired_flex_max = v[2 * i]
        ext_min_hi = min(sec_limit, thickness - paired_flex_max)
        if ext_min_hi < 0.125:
            return None
        ext_min = rng.uniform(0.125, ext_min_hi)
        ext_max = rng.uniform(ext_min, min(0.5, sec_limit))
        v[2 * numFlexs + 2 * i]     = ext_max
        v[2 * numFlexs + 2 * i + 1] = ext_min

    if RELAX_THICKNESS:
        v[numJointParameters] = thickness

    return v


def score(v):
    """
    Penalised score across all nonlinear constraints.
    Zero = fully feasible. Lower = better.
    """
    s = 0.0

    br = parallel_evaluator.eval_bend_radius(v)
    if br < 0.4:
        s += 10.0 * (0.4 - br) ** 2

    tensions         = parallel_evaluator.eval_tensions(v)
    tension_violation = np.maximum(tensions - TENSION_LIMIT, 0)
    s += 1.0 * np.sum(tension_violation ** 2)

    envelopes         = parallel_evaluator.eval_envelopes(v)
    envelope_violation = np.maximum(envelopes, 0)
    s += 5.0 * np.sum(envelope_violation ** 2)

    c_lim         = parallel_evaluator.eval_c_limit(v)
    c_violation   = np.maximum(-c_lim, 0)
    s += 2.0 * np.sum(c_violation ** 2)

    return s


def print_diagnostics(label, v):
    tensions  = parallel_evaluator.eval_tensions(v)
    envelopes = parallel_evaluator.eval_envelopes(v)
    br        = parallel_evaluator.eval_bend_radius(v)
    c_lim     = parallel_evaluator.eval_c_limit(v)
    thickness = v[numJointParameters] if RELAX_THICKNESS else thickness_lo

    # Linear constraint check
    pair_violations = [
        (i, v[2*i], v[2*i+1])
        for i in range(numJointParameters // 2)
        if v[2*i] < v[2*i+1]
    ]

    print(f"\n  ── {label} ──────────────────────────────────")
    print(f"  Score:          {score(v):.6f}")
    print(f"  Thickness:      {thickness:.4f}  "
          f"[{thickness_lo:.4f}, {thickness_hi:.4f}]")
    print(f"  Max tension:    {np.max(tensions):.3f}  (limit {TENSION_LIMIT})")
    print(f"  Bend radius:    {br:.4f}  (limit 0.40)")
    print(f"  Envelope max:   {np.max(envelopes):.4f}  (want <= 0)")
    print(f"  Envelope min:   {np.min(envelopes):.4f}")
    print(f"  c-limit min:    {np.min(c_lim):.4f}  (want >= 0)")
    print(f"  Pair violations:{pair_violations if pair_violations else 'none'}")
    print(f"  v = {np.array2string(v, precision=4, suppress_small=True)}")
    print(f"  ────────────────────────────────────────────")


def generate_seed(n_attempts=300, seed=42, verbose=True):
    rng        = np.random.default_rng(seed)
    best_v     = None
    best_score = np.inf

    print(f"Sampling {n_attempts} random candidates...")
    for attempt in range(n_attempts):
        # Anneal: start thick, allow thinner over time
        progress  = attempt / n_attempts
        t_lo      = thickness_lo + (thickness_hi - thickness_lo) * max(0.0, 1.0 - progress / 0.6)
        thickness = rng.uniform(t_lo, thickness_hi)

        v = sample_once(rng, thickness)
        if v is None:
            continue

        s = score(v)
        if s < best_score:
            best_score = s
            best_v     = v.copy()
            if verbose:
                envelopes = parallel_evaluator.eval_envelopes(v)
                print(f"  [{attempt:4d}] score={s:.5f}  "
                      f"thick={thickness:.4f}  "
                      f"t_max={np.max(parallel_evaluator.eval_tensions(v)):.2f}  "
                      f"br={parallel_evaluator.eval_bend_radius(v):.4f}  "
                      f"env_max={np.max(envelopes):.4f}")
            if s < 1e-6:
                print(f"\n  Fully feasible seed found at attempt {attempt}!")
                break

    print_diagnostics("Best random seed", best_v)

    # ── Polish ───────────────────────────────────────────────────────────────
    print("\n  Polishing with Nelder-Mead...")

    def penalized(v):
        # Soft-enforce linear constraints during polish
        penalty = 0.0
        v_c = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
        penalty += 1e4 * np.sum((v - v_c) ** 2)
        for i in range(numFlexs):
            if v[2*i] < v[2*i+1]:
                penalty += 1e4 * (v[2*i+1] - v[2*i]) ** 2
        for i in range(numExts):
            paired_flex_max = v[2 * i]
            ext_min         = v[2 * numFlexs + 2 * i + 1]
            thickness       = v[numJointParameters] if RELAX_THICKNESS else thickness_lo
            total           = paired_flex_max + ext_min
            if total > thickness:
                penalty += 1e4 * (total - thickness) ** 2
        return score(v) + penalty

    res = optimize.minimize(
        penalized,
        best_v,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol':   1e-6,
            'fatol':   1e-7,
            'adaptive': True,
        }
    )

    # Clip polished result back to bounds
    polished_v = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])

    # Re-enforce max>=min pairs by averaging any violated pairs
    for i in range(numJointParameters // 2):
        if polished_v[2*i] < polished_v[2*i+1]:
            mid = (polished_v[2*i] + polished_v[2*i+1]) / 2.0
            polished_v[2*i]     = mid
            polished_v[2*i + 1] = mid

    print_diagnostics("Polished seed", polished_v)
    return polished_v


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seed_v = generate_seed(n_attempts=300, seed=42, verbose=True)

    out_file = "seed.smx"
    np.savetxt(out_file, seed_v)
    print(f"\n  Saved to {out_file}")
    print(f"\n  Paste this into variableOptimizerGlobalParallel.py as v0:")
    print(f"  v0 = {seed_v.tolist()}")