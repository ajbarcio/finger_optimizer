from strucMatrices import VariableStrucMatrix, secondaryDev
from utils import *
from finger import Finger, StructureKineMismatch, Grasp
from scipy import optimize
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult
from scipy.stats import qmc
import statistics
import time
import sys
import multiprocessing as mp
from multiprocessing import Pool

from finger import valeroFinger, dof_constraints
import parallel_evaluator

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Global pool — created once in __main__, referenced by constraint wrappers ─
_pool: Pool = None
_objective_mode: str = None
_numJointParameters_global: int = None


def extract_uniform_value(val):
    if not isinstance(val, (list, np.ndarray)):
        return val
    if len(val)==0:
        raise ValueError("List cannot be empty")

    first = val[0]
    if not all(x == first for x in val):
        raise ValueError("All elements in the list must be identical")

    return first

def z_score_from_percentile(percentile):
    probability = percentile / 100.0
    return statistics.NormalDist().inv_cdf(probability)


class stallException(Exception):
    pass


def createFingerFromVector(v) -> Finger:
    """Main-process version for post-processing only."""
    return parallel_evaluator.build_finger(v)


# ─── Vectorized constraint wrappers ───────────────────────────────────────────
# DE with vectorized=True passes population as (N, S).
# Each wrapper must return (M, S).

def _wrap(arr: np.ndarray) -> np.ndarray:
    """Squeeze (M, 1) → (M,) for scalar probes; leave (M, S) alone."""
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr

def _vectorized_tensions(population: np.ndarray) -> np.ndarray:
    population = np.atleast_2d(population)
    if population.shape[0] != overallNumElements:
        population = population.T          # (N, S) → ensure N is design-vec dim
    vectors = [population[:, i] for i in range(population.shape[1])]
    results = _pool.map(parallel_evaluator.eval_tensions, vectors)
    return _wrap(np.column_stack(results))        # (num_tensions, S)


def _vectorized_bend_radius(population: np.ndarray) -> np.ndarray:
    population = np.atleast_2d(population)
    if population.shape[0] != overallNumElements:
        population = population.T
    vectors = [population[:, i] for i in range(population.shape[1])]
    results = _pool.map(parallel_evaluator.eval_bend_radius, vectors)
    return _wrap(np.array(results).reshape(1, -1))   # (1, S))


def _vectorized_c_limit(population: np.ndarray) -> np.ndarray:
    population = np.atleast_2d(population)
    if population.shape[0] != overallNumElements:
        population = population.T
    vectors = [population[:, i] for i in range(population.shape[1])]
    results = _pool.map(parallel_evaluator.eval_c_limit, vectors)
    return _wrap(np.column_stack(results))        # (num_links, S)


def _vectorized_envelopes(population: np.ndarray) -> np.ndarray:
    population = np.atleast_2d(population)
    if population.shape[0] != overallNumElements:
        population = population.T
    vectors = [population[:, i] for i in range(population.shape[1])]
    results = _pool.map(parallel_evaluator.eval_envelopes, vectors)
    return _wrap(np.column_stack(results))        # (num_poses, S)


def _vectorized_objective(population: np.ndarray) -> np.ndarray:
    population = np.atleast_2d(population)
    if population.shape[0] != overallNumElements:
        population = population.T
    args = [
        (population[:, i], _objective_mode, _numJointParameters_global)
        for i in range(population.shape[1])
    ]
    results = _pool.map(parallel_evaluator.eval_objective, args)
    return _wrap(np.array(results))              # (S,)


# ─── FingerEvaluator (callbacks + post-processing only) ───────────────────────
class FingerEvaluator:

    OBJECTIVE_WORST_CASE_OVL = "worst_case_OVL"
    OBJECTIVE_WORST_CASE_TENSION = "worst_case_tension"
    OBJECTIVE_THICKNESS = "thickness"
    OBJECTIVE_STRENGTH_INCREASE = "strength_increase"

    def __init__(self, grasps: list[Grasp] = None,
                 objective_mode: str = "worst_case_OVL"):
        self._v_save = None
        self._finger = None
        self._cb_last_lines = 0
        self._threshold_times = 0
        self.optimalities = []
        self.worst_case_tensions = []
        self._curr_wct = None
        self._curr_wcr = None
        self.worst_pose_for_strength = None
        self.best_objective = np.inf
        self.best_x = None
        self.objective_history = []
        self._eval_count = 0
        self._start_time = None
        self.grasps = grasps
        self.objective_mode = objective_mode

    def _get_finger(self, v):
        v = np.asarray(v)
        if self._v_save is None or not np.array_equal(v, self._v_save):
            self._v_save = v.copy()
            self._finger = parallel_evaluator.build_finger(v)
        return self._finger

    def global_callback(self, xk, convergence=0.0):
        if self._start_time is None:
            self._start_time = time.time()
        elapsed = time.time() - self._start_time
        self._eval_count += 1

        # Cheap serial re-evaluation for display — happens once per generation
        self._curr_wct = float(np.max(parallel_evaluator.eval_tensions(xk)))
        self._curr_wcr = float(parallel_evaluator.eval_bend_radius(xk))
        obj_val = _vectorized_objective(xk.reshape(-1, 1))[0]

        self.objective_history.append(obj_val)
        self.worst_case_tensions.append(self._curr_wct)

        if obj_val < self.best_objective:
            self.best_objective = obj_val
            self.best_x = xk.copy()

        print("\r" + "\033[F" * self._cb_last_lines, end="")
        lines = [
            f"─── Global Optimization Progress ───",
            f"  Elapsed time:    {elapsed:.1f}s",
            f"  Generations:     {self._eval_count}",
            f"  Convergence:     {convergence:.6f}",
            f"  Current obj:     {obj_val:.6f}",
            f"  Best obj:        {self.best_objective:.6f}",
            f"  Worst tension:   {self._curr_wct:.3f}",
            f"  Worst radius:    {self._curr_wcr:.4f}",
            f"  Current best x:  "
            f"{np.array2string(xk, precision=4, suppress_small=True)}",
            f"────────────────────────────────────",
        ]
        print("\r\033[2K" + "\n".join(lines), end="", flush=True)
        self._cb_last_lines = len(lines)

    def worst_case_OVL(self, v):
        Fing = self._get_finger(v)
        worst = optimize.minimize(
            Fing.structure.get_magnitude,
            x0=[0] * Fing.numJoints,
            bounds=[(0, np.pi / 2)] * Fing.numJoints
        )
        self.worst_pose_for_strength = worst.x
        return -worst.fun

    def worst_case_tension(self, v):
        t = parallel_evaluator.eval_tensions(v)
        self._curr_wct = float(np.max(t))
        return self._curr_wct


# ─── Constraint + bound builders ─────────────────────────────────────────────

def linear_constraints(overallNumElements, numJointParameters,
                       numFlexs, numExts,
                       numConstraintParameters, numLengthParameters,
                       RELAX_THICKNESS, DESIGN_LENGTHS,
                       initial_overall_thickness, growth_factor,
                       overall_length):
    """Pure function — no reference to evaluator or any global state."""

    bounds = (
        [(.125, .5)] * numJointParameters
        + [(initial_overall_thickness,
            initial_overall_thickness * growth_factor)]
          * (numConstraintParameters if RELAX_THICKNESS else 0)
        + [(0.875, 2)] * (numLengthParameters if DESIGN_LENGTHS else 0)
    )

    # Each max >= its associated min
    max_min_jacobian = np.array([
        np.roll(row, shift) for row, shift in
        zip(
            -np.eye(numJointParameters // 2, numJointParameters, k=1)
            + np.eye(numJointParameters // 2, numJointParameters, k=0),
            np.arange(numJointParameters)
        )
    ])
    if DESIGN_LENGTHS or RELAX_THICKNESS:
        max_min_jacobian = np.hstack([
            max_min_jacobian,
            np.zeros((max_min_jacobian.shape[0],
                      overallNumElements - max_min_jacobian.shape[1]))
        ])

    # Flexion + extension <= overall thickness
    overall_thickness_jacobian = np.zeros([numFlexs, numJointParameters])
    for i in range(numFlexs):
        overall_thickness_jacobian[i, 2 * i] = 1
        overall_thickness_jacobian[i, 2 * numFlexs + 2 * i + 1] = 1

    # Max extension constrained by overall thickness
    second_thickness_jacobian = np.hstack([
        np.zeros([numExts * 2, numFlexs * 2]),
        np.eye(numExts * 2)
    ])

    if RELAX_THICKNESS:
        overall_thickness_jacobian = np.hstack([
            overall_thickness_jacobian,
            -np.ones((overall_thickness_jacobian.shape[0],
                      numConstraintParameters))
        ])
        second_thickness_jacobian = np.hstack([
            second_thickness_jacobian,
            -np.ones((second_thickness_jacobian.shape[0],
                      numConstraintParameters))
        ])

    if DESIGN_LENGTHS:
        overall_thickness_jacobian = np.hstack([
            overall_thickness_jacobian,
            np.zeros((overall_thickness_jacobian.shape[0],
                      numLengthParameters))
        ])
        second_thickness_jacobian = np.hstack([
            second_thickness_jacobian,
            np.zeros((second_thickness_jacobian.shape[0],
                      numLengthParameters))
        ])

    constraints_objects = [
        LinearConstraint(
            A=max_min_jacobian,
            lb=0, ub=np.inf,
            keep_feasible=True
        ),
        LinearConstraint(
            A=overall_thickness_jacobian,
            lb=-np.inf if RELAX_THICKNESS else 0,
            ub=0 if RELAX_THICKNESS else initial_overall_thickness,
            keep_feasible=True
        ),
        LinearConstraint(
            A=second_thickness_jacobian,
            lb=-np.inf if RELAX_THICKNESS else 0,
            ub=(0 if RELAX_THICKNESS
                else initial_overall_thickness) - 2.25 / 25.4 - .125,
            keep_feasible=True
        ),
    ]

    if DESIGN_LENGTHS:
        overall_length_jacobian = np.array(
            [0] * (overallNumElements - numLengthParameters)
            + [1] * numLengthParameters
        )
        constraints_objects.append(
            LinearConstraint(
                A=overall_length_jacobian,
                lb=0, ub=overall_length - 0.75,
                keep_feasible=True
            )
        )

    for i, c in enumerate(constraints_objects):
        if c.A.shape[1] != overallNumElements:
            raise AssertionError(
                f"Linear constraint {i} has wrong width: "
                f"{c.A.shape[1]} != {overallNumElements}"
            )

    # print(constraints_objects[0].ub, constraints_objects[1].ub)

    return bounds, constraints_objects


def nonlinear_constraints(CONTAIN_HUMAN, TENSION_LIMIT):
    """
    Returns NonlinearConstraint objects using vectorized pool-based wrappers.
    vectorized=True tells DE that our functions accept (N, S) and return (M, S).
    """
    constraints_objects = []

    if CONTAIN_HUMAN:
        constraints_objects.append(NonlinearConstraint(
            fun=_vectorized_envelopes,
            lb=-np.inf,
            ub=0,
            # vectorized=True,
        ))

    # constraints_objects.append(NonlinearConstraint(
    #     fun=_vectorized_tensions,
    #     lb=0,
    #     ub=TENSION_LIMIT,
    #     vectorized=True,
    # ))

    constraints_objects.append(NonlinearConstraint(
        fun=_vectorized_bend_radius,
        lb=0.40,
        ub=np.inf,
        # vectorized=True,
    ))

    constraints_objects.append(NonlinearConstraint(
        fun=_vectorized_c_limit,
        lb=0,
        ub=np.inf,
        # vectorized=True,
    ))

    return constraints_objects

# ─── Configuration ────────────────────────────────────────────────────────────

OBJECTIVE_MODE          = FingerEvaluator.OBJECTIVE_THICKNESS
GLOBAL_METHOD           = "differential_evolution"
POLISH                  = False
MAX_ITER                = 200
TOL                     = 1e-3
SEED                    = 42
TENSION_LIMIT           = 50.0
RELAX_THICKNESS         = True
DESIGN_LENGTHS          = False
CONTAIN_HUMAN           = True

numJoints               = 3
numTendons              = numJoints + 1
numFlexs                = 3
numExts                 = 3
numJointParameters      = numFlexs * 2 + numExts * 2
numConstraintParameters = 1
numLengthParameters     = 2

overallNumElements = numJointParameters
if RELAX_THICKNESS:
    overallNumElements += numConstraintParameters
if DESIGN_LENGTHS:
    overallNumElements += numLengthParameters

F                        = np.array([0, 5, 0])
lengths                  = [1.866, 1.4, 0.75]
overall_length           = 4
initial_overall_thickness = 0.65
growth_factor            = 1.2
R                        = secondaryDev.R
D                        = secondaryDev.D

# ─── Initial Population ───────────────────────────────────────────────────────

def make_linearly_feasible_population(n_pop, bounds, lin_constraints,
                                      relax_thickness=RELAX_THICKNESS,
                                      design_lengths=DESIGN_LENGTHS,
                                      rng_seed=42,
                                      oversample=8):
    """
    Generate n_pop individuals satisfying all linear constraints,
    using the already-built constraint matrices to both check and fix.
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # lin_constraints is a list of LinearConstraint objects, each has .A, .lb, .ub
    # Unpack them in the order they're built:
    #   [0] max_min:          lb=0,       ub=inf
    #   [1] overall_thick:    lb=-inf/0,  ub=0/initial_overall_thickness
    #   [2] second_thick:     lb=-inf/0,  ub=0/sec_limit
    #   [3] overall_length    (only if design_lengths)
    A_maxmin  = lin_constraints[0].A
    lb_maxmin = extract_uniform_value(lin_constraints[0].lb)   # 0
    A_thick1  = lin_constraints[1].A
    ub_thick1 = extract_uniform_value(lin_constraints[1].ub)   # 0 (relax) or initial_overall_thickness
    lb_thick1 = extract_uniform_value(lin_constraints[1].lb)
    A_thick2  = lin_constraints[2].A
    ub_thick2 = extract_uniform_value(lin_constraints[2].ub)
    lb_thick2 = extract_uniform_value(lin_constraints[2].lb)



    def is_feasible(v):
        """Check all linear constraints using the actual matrices."""
        for lc in lin_constraints:
            Av = lc.A @ v
            if np.any(Av < lc.lb - 1e-9) or np.any(Av > lc.ub + 1e-9):
                return False
        return True

    def fix(v):
        """
        Analytically project v onto the linear feasible region using the
        constraint matrices to identify and fix violations.
        Returns None if the vector can't be made feasible.
        """
        v = v.copy()

        # ── Fix 1: max >= min (A_maxmin @ v >= 0) ────────────────────────────
        # Each row of A_maxmin is [0..0, +1, -1, 0..0]
        # violation when A_maxmin[i] @ v < 0  →  v[max_idx] < v[min_idx]
        for row in A_maxmin:
            val = row @ v
            if val < 0:
                # Find the +1 and -1 positions in this row
                max_idx = np.argmax(row)
                min_idx = np.argmin(row)
                mid = (v[max_idx] + v[min_idx]) / 2.0
                v[max_idx] = mid
                v[min_idx] = mid

        # ── Fix 2: overall thickness (A_thick1 @ v <= ub_thick1) ─────────────
        # Each row sums flex_max + ext_min (and subtracts thickness if relaxed)
        # violation when A_thick1[i] @ v > ub_thick1
        for row in A_thick1:
            val = row @ v
            # print(val)
            # print(ub_thick1)
            if val > ub_thick1 + 1e-9:
                excess = val - ub_thick1
                # Positive entries in this row are the ones contributing the excess
                pos_idx = np.where(row > 0)[0]
                total_pos = np.sum(v[pos_idx])
                if total_pos > 1e-12:
                    scale = (total_pos - excess) / total_pos
                    v[pos_idx] *= scale

        # ── Fix 3: second thickness (A_thick2 @ v <= ub_thick2) ──────────────
        for row in A_thick2:
            val = row @ v
            if val > ub_thick2 + 1e-9:
                excess  = val - ub_thick2
                pos_idx = np.where(row > 0)[0]
                total_pos = np.sum(v[pos_idx])
                if total_pos > 1e-12:
                    scale = (total_pos - excess) / total_pos
                    v[pos_idx] *= scale

        # ── Fix 4: overall length (if design_lengths) ────────────────────────
        if design_lengths:
            lc_len = lin_constraints[3]
            val    = lc_len.A @ v
            if val > lc_len.ub + 1e-9:
                # Scale the length variables down proportionally
                len_idx   = np.where(lc_len.A > 0)[0]
                total_len = np.sum(v[len_idx])
                if total_len > 1e-12:
                    v[len_idx] *= lc_len.ub / total_len

        # ── Fix 5: re-check max>=min after scaling (scaling may break it) ────
        for row in A_maxmin:
            val = row @ v
            if val < 0:
                max_idx    = np.argmax(row)
                min_idx    = np.argmin(row)
                mid        = (v[max_idx] + v[min_idx]) / 2.0
                v[max_idx] = mid
                v[min_idx] = mid

        # ── Fix 6: clip to bounds ─────────────────────────────────────────────
        v = np.clip(v, lo, hi)

        # ── Final check using the actual matrices ─────────────────────────────
        if not is_feasible(v):
            return None
        return v

    # ── Sobol oversample → fix → collect ─────────────────────────────────────
    sampler    = qmc.Sobol(d=overallNumElements, scramble=True, seed=rng_seed)
    raw        = qmc.scale(sampler.random(n_pop * oversample), lo, hi)

    population = []
    n_rejected = 0
    for row in raw:
        v = fix(row)
        if v is not None:
            population.append(v)
        else:
            n_rejected += 1
        if len(population) == n_pop:
            break

    print(f"  Sobol population: {len(population)}/{n_pop} accepted, "
          f"{n_rejected} rejected after fixing")

    if len(population) < n_pop:
        print(f"  Warning: short by {n_pop - len(population)} — "
              f"increase oversample factor (currently {oversample})")

    pop = np.array(population)

    # ── Sanity check every individual against all constraint matrices ─────────
    n_infeasible = sum(1 for v in pop if not is_feasible(v))
    if n_infeasible > 0:
        print(f"  WARNING: {n_infeasible} individuals still infeasible after fixing!")
    else:
        print(f"  All {len(pop)} individuals verified linearly feasible.")

    return pop

# ─── Grasp definitions ────────────────────────────────────────────────────────

q_ext = [10 * np.pi / 180] * 3
q_int = [45 * np.pi / 180] + [10 * np.pi / 180] * 2
q_flx = [45 * np.pi / 180] * 2 + [10 * np.pi / 180]

z   = z_score_from_percentile(50)
sig = [10.86, 2.20, 12.5, 6.13, 20.2, 11.05, 10.47, 2.04, 12.68]
means = [26.3, 6.6, 25.3, 25.25, 7.26, 26.3, 31.06, 7.56, 27.08]
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


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Expose config to vectorized wrappers via module-level globals
    _objective_mode = OBJECTIVE_MODE
    _numJointParameters_global = numJointParameters

    # ── Build the persistent worker pool ─────────────────────────────────────
    # initializer runs parallel_evaluator.worker_init() once in each child process,
    # populating all the config globals the worker functions need.
    init_args = (
            R, D,
            numFlexs, numExts, numTendons,
            numLengthParameters, DESIGN_LENGTHS,
            lengths, overall_length,
            VC_Dir_Grasps,          # picklable list of Grasp objects
        )

    n_workers = mp.cpu_count()
    _pool = Pool(
        processes=n_workers,
        initializer=parallel_evaluator.worker_init,
        initargs=init_args
    )
    print(f"Worker pool started: {n_workers} processes")

    # also give an extra worker_init call
    parallel_evaluator.worker_init(*init_args)

    try:
        # grasps    = VC_Dir_Grasps
        grasps = []
        evaluator = FingerEvaluator(grasps, objective_mode=OBJECTIVE_MODE)

        bounds, lin_constraints = linear_constraints(
            overallNumElements, numJointParameters,
            numFlexs, numExts,
            numConstraintParameters, numLengthParameters,
            RELAX_THICKNESS, DESIGN_LENGTHS,
            initial_overall_thickness, growth_factor,
            overall_length,
        )
        nl_constraints = nonlinear_constraints(CONTAIN_HUMAN, TENSION_LIMIT)
        all_constraints = lin_constraints + nl_constraints



        ## OLD, trying to fix this but running a lotta shit on my computer rn
        # v0 = (
        #     [0.2955, 0.2481, 0.4409, 0.3633, 0.4214, 0.2341,
        #      0.3686, 0.2239, 0.3608, 0.181,  0.3224, 0.2252]
        #     + ([initial_overall_thickness + .05] if RELAX_THICKNESS  else [])
        #     + (lengths[:numLengthParameters]      if DESIGN_LENGTHS   else [])
        # )
        # assert len(v0) == overallNumElements, (
        #     f"v0 length {len(v0)} != overallNumElements {overallNumElements}"
        # )

        ## NEW strat
        # init_pop_linearly_good = make_linearly_feasible_population(
        #             20*overallNumElements,
        #             bounds,
        #             lin_constraints
        #         )
        # Later will load this from seed finder, for now just waiting for
        # polishing to finish so we use the unpolished version
        # v0 = [0.41324, 0.37058, 0.31430, 0.29718,
        #       0.34517, 0.31221, 0.31880, 0.20174,
        #       0.30581, 0.29049, 0.42034, 0.24193,
        #       0.76356]
        v0 = [0.4218283505438559, 0.41959014378853987, 0.3665921892958935,
              0.3665921892958935, 0.33506187673461274, 0.33506187673461274,
              0.5, 0.17327029829363153, 0.4485202348905297, 0.3379887996098218,
              0.25933335083433945, 0.125, 0.78]

        ## ANOTHER strat from the introspection we did
        v0 = ([.125,0.25]*numFlexs+[1.87868*0.125,.125]*numExts+[]
            + ([initial_overall_thickness + .05] if RELAX_THICKNESS  else [])
            + (lengths[:numLengthParameters]      if DESIGN_LENGTHS   else [])
        )

        print(f"\n{'═' * 60}")
        print(f"  Running global optimization")
        print(f"  Method:    {GLOBAL_METHOD}")
        print(f"  Objective: {OBJECTIVE_MODE}")
        print(f"  Workers:   {n_workers} (pool-managed)")
        print(f"  Polish:    {POLISH}")
        print(f"  Max iter:  {MAX_ITER}")
        print(f"  Seed:      {SEED}")
        print(f"{'═' * 60}\n")

        try:
            if GLOBAL_METHOD == "differential_evolution":
                result = optimize.differential_evolution(
                    _vectorized_objective,
                    bounds=bounds,
                    constraints=all_constraints,
                    callback=evaluator.global_callback,
                    workers=1,          # pool-managed — do NOT double-parallelise
                    vectorized=True,    # objective + constraints both accept (N,S)
                    maxiter=MAX_ITER,
                    tol=TOL,
                    seed=SEED,
                    polish=POLISH,
                    # init=init_pop_linearly_good,
                    init='sobol',
                    x0=v0,
                    mutation=(0.5, 1.5),
                    recombination=0.7,
                    popsize=20,
                    updating='deferred', # required for vectorized=True
                    disp=False,
                )

            elif GLOBAL_METHOD == "shgo":
                # shgo doesn't support vectorized=True or a pool directly,
                # so we hand it serial wrappers that internally use the pool
                # for each minimizer call — still faster than no pool at all.
                def serial_objective(v):
                    return _vectorized_objective(v.reshape(-1, 1))[0]

                def serial_tensions(v):
                    return _vectorized_tensions(v.reshape(-1, 1))[:, 0]

                def serial_bend_radius(v):
                    return _vectorized_bend_radius(v.reshape(-1, 1))[0, 0]

                def serial_c_limit(v):
                    return _vectorized_c_limit(v.reshape(-1, 1))[:, 0]

                def serial_envelopes(v):
                    return _vectorized_envelopes(v.reshape(-1, 1))[:, 0]

                shgo_nl = []
                if CONTAIN_HUMAN:
                    shgo_nl.append(NonlinearConstraint(
                        fun=serial_envelopes, lb=-np.inf, ub=0
                    ))
                shgo_nl.append(NonlinearConstraint(
                    fun=serial_tensions, lb=0, ub=TENSION_LIMIT
                ))
                shgo_nl.append(NonlinearConstraint(
                    fun=serial_bend_radius, lb=0.40, ub=np.inf
                ))
                shgo_nl.append(NonlinearConstraint(
                    fun=serial_c_limit, lb=0, ub=np.inf
                ))

                result = optimize.shgo(
                    serial_objective,
                    bounds=bounds,
                    constraints=lin_constraints + shgo_nl,
                    options={"maxiter": MAX_ITER, "disp": True},
                    minimizer_kwargs={
                        "method": "trust-constr",
                        "constraints": lin_constraints + shgo_nl,
                        "options": {"maxiter": 50, "gtol": 1e-3},
                    },
                )

            else:
                raise ValueError(f"Unknown global method: {GLOBAL_METHOD}")

        except KeyboardInterrupt:
            print("\n\nOptimization terminated by user.")
            if evaluator.best_x is not None:
                result = OptimizeResult(
                    x=evaluator.best_x,
                    fun=evaluator.best_objective,
                    success=False,
                    message="Terminated by user",
                )
            else:
                print("No valid result found before termination.")
                raise SystemExit(1)

        except stallException as e:
            print(f"\n\n{e}")
            result = OptimizeResult(
                x=evaluator.best_x if evaluator.best_x is not None else v0,
                fun=(evaluator.best_objective
                     if evaluator.best_objective < np.inf else np.nan),
                success=False,
                message=str(e),
            )

    finally:
        # Always shut the pool down cleanly, even on exception or Ctrl-C
        _pool.close()
        _pool.join()
        print("\nWorker pool shut down.")

    # ─── Results summary ──────────────────────────────────────────────────────
    print("\n\n")
    print(f"{'═' * 60}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Success:    {getattr(result, 'success', 'N/A')}")
    print(f"  Message:    {getattr(result, 'message', 'N/A')}")
    print(f"  Obj value:  {result.fun}")
    print(f"  Evals:      {evaluator._eval_count}")
    print(f"  Solution:   "
          f"{np.array2string(result.x, precision=4, suppress_small=True)}")
    print(f"{'═' * 60}\n")

    v_result = result.x
    name = (
        "VCCP_Introspected_V0"
        + ("_RT" if RELAX_THICKNESS else "")
        + ("_DL" if DESIGN_LENGTHS  else "")
    )
    np.savetxt(f"{name}.smx", v_result)

    if len(evaluator.objective_history) > 0:
        plt.figure("Objective History")
        plt.plot(evaluator.objective_history, 'b-')
        plt.xlabel("Callback iteration")
        plt.ylabel("Objective value")
        plt.title(f"Global Optimization Convergence ({OBJECTIVE_MODE})")
        plt.grid(True)

    if len(evaluator.worst_case_tensions) > 0:
        plt.figure("Tension History")
        plt.plot(evaluator.worst_case_tensions, 'b-')
        plt.xlabel("Callback iteration")
        plt.ylabel("Worst-case tension")
        plt.title(f"Tension over Optimization ({OBJECTIVE_MODE})")
        plt.grid(True)

    # ─── Post-processing ──────────────────────────────────────────────────────
    for _ in range(5):
        print("")

    resultFinger = createFingerFromVector(v_result)

    resultFinger.structure.minFactor = 1 / optimize.minimize_scalar(
        lambda theta: -resultFinger.structure.controllability(
            [theta] * resultFinger.numJoints
        ),
        bracket=(0, np.pi / 2),
        bounds=(0, np.pi / 2),
    ).fun

    for q_disp in (0, np.pi / 2):
        print(resultFinger.structure([q_disp] * resultFinger.numJoints))

    for function in resultFinger.structure.effortFunctions:
        if isinstance(function,
                       VariableStrucMatrix.convergent_circles_extension_joint):
            print("type: extension")
        elif isinstance(function,
                        VariableStrucMatrix
                        .ideal_convergent_circles_joint_with_limit):
            print("type: flexion")
        print(f"  idx: {function.idx}")
        print(f"  c:   {function.c}")
        print(f"  r:   {function.r}")

    qs       = np.linspace(0, np.pi / 2, 75)
    tvecs    = []
    tvecs2   = []
    conditions  = []
    magnitudes  = []

    for q in qs:
        tensions = resultFinger.grip_to_tensions(
            [q] * resultFinger.numJoints,
            resultFinger.grasp_to_grip(Grasp(
                [F] * resultFinger.numJoints,
                [q] * resultFinger.numJoints,
                frame="EE"
            ))
        )
        tensions2 = resultFinger.grip_to_tensions(
            [q] * resultFinger.numJoints,
            resultFinger.tip_wrench_at_pose_to_grip(
                [q] * resultFinger.numJoints,
                -F * 0.1,
                frame="EE"
            )
        )
        tvecs.append(tensions)
        tvecs2.append(tensions2)
        conditions.append(
            resultFinger.structure.controllability([q] * resultFinger.numJoints)
        )
        magnitudes.append(
            resultFinger.structure.get_magnitude([q] * resultFinger.numJoints)
        )

    print("\n\n\n\n\n")
    print(f"Objective mode: {OBJECTIVE_MODE}")

    if OBJECTIVE_MODE == FingerEvaluator.OBJECTIVE_WORST_CASE_OVL:
        print("Lowest OVL:")
        print(evaluator.worst_case_OVL(v_result))
        print(f"  happens at: {evaluator.worst_pose_for_strength}")
        print("  confirming:")
        print(f"  {resultFinger.structure.get_magnitude(evaluator.worst_pose_for_strength)}")
    elif OBJECTIVE_MODE == FingerEvaluator.OBJECTIVE_WORST_CASE_TENSION:
        print("Worst-case tension (objective):")
        print(result.fun)

    print("Worst-case tension (from constraint):")
    print(evaluator.worst_case_tension(v_result))

    plt.figure("Conditions")
    plt.plot(qs, conditions)
    plt.xlabel("Joint angle (rad)")
    plt.ylabel("Controllability")
    plt.grid(True)

    plt.figure("Magnitudes")
    magnitudes = np.array(magnitudes)
    plt.plot(qs, magnitudes / np.min(magnitudes))
    plt.xlabel("Joint angle (rad)")
    plt.ylabel("Normalized magnitude")
    plt.grid(True)

    resultFinger.structure.plotCapability(
        [0] * resultFinger.numJoints, enforcePosTension=True
    )
    resultFinger.structure.plotCapability(
        [np.pi / 2] * resultFinger.numJoints, enforcePosTension=True
    )
    plt.show()