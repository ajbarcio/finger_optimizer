# batch_optimizer.py
# Runs multiple optimizations from different JointParams starting points.
# All finger logic from finger_config, all worker logic from parallel_evaluator.

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from scipy.optimize import (differential_evolution, minimize,
                             NonlinearConstraint, LinearConstraint,
                             OptimizeResult)
from scipy.stats import qmc
import time
import json
import hashlib
import argparse
import warnings
warnings.filterwarnings("ignore")

import finger_config as cfg
import parallel_evaluator
from finger_config import (JointParams, load_joint_params,
                            joint_params_to_vector, vector_to_joint_params,
                            build_finger_from_vector, build_finger_from_params)
from finger import Grasp, valeroFinger, dof_constraints
from strucMatrices import secondaryDev

import concurrent.futures

# ─── Configuration ────────────────────────────────────────────────────────────
# All toggled via argparse — see parse_args() below

OBJECTIVE_MODE   = "thickness"    # "thickness" |
                                  # "worst_case_OVL" | "worst_case_tension" |
                                  # "minimize_volume" | "maximize_volume"
METHOD           = "de"           # "de" | "trust-constr" | "shgo"
DESIGN_LENGTHS   = False
CONTAIN_HUMAN    = True
ENFORCE_PF       = True
TENSION_LIMIT    = cfg.TENSION_LIMIT
WORKERS          = mp.cpu_count()-1
MAX_ITER         = 2000
TOL              = 1e-3
SEED             = 42
POLISH           = False

# ─── Grasps (mirror from variableOptimizerGlobalParallel) ────────────────────

import statistics

def _z(pct):
    return statistics.NormalDist().inv_cdf(pct / 100.0)

def make_grasps() -> list[Grasp]:
    q_ext = [10 * np.pi / 180] * 3
    q_int = [45 * np.pi / 180] + [10 * np.pi / 180] * 2
    q_flx = [45 * np.pi / 180] * 2 + [10 * np.pi / 180]
    z    = _z(50)
    sig  = [10.86, 2.20, 12.5, 6.13, 20.2, 11.05, 10.47, 2.04, 12.68]
    means= [26.3,  6.6,  25.3, 25.25, 7.26, 26.3,  31.06, 7.56, 27.08]
    vals = [means[i] + z * sig[i] for i in range(len(sig))]
    F    = np.array([0, 5, 0])
    return [
        Grasp([[0]*3, [0]*3, [0,  vals[0]/4.448, 0]], q_ext),
        Grasp([[0]*3, [0]*3, [0, -vals[1]/4.448, 0]], q_ext),
        Grasp([[0]*3, [0]*3, [0,  vals[3]/4.448, 0]], q_int),
        Grasp([[0]*3, [0]*3, [0, -vals[4]/4.448, 0]], q_int),
        Grasp([[0]*3, [0]*3, [0,  vals[6]/4.448, 0]], q_flx),
        Grasp([[0]*3, [0]*3, [0, -vals[7]/4.448, 0]], q_flx),
        Grasp([[0]*3, [0]*3, [vals[8]/4.448, 0,  0]], q_flx),
    ]

# ─── Pool ─────────────────────────────────────────────────────────────────────

_pool: Pool          = None
_cache_key:    str   = None
_cache_results: list = None


def _init_pool(grasps: list):
    global _pool
    init_args = (grasps, DESIGN_LENGTHS, TENSION_LIMIT, ENFORCE_PF)   # 3 args to match new worker_init
    _pool = Pool(
        processes=WORKERS,
        initializer=parallel_evaluator.worker_init,
        initargs=init_args,
    )
    parallel_evaluator.worker_init(*init_args)   # init main process too
    print(f"Pool started: {WORKERS} workers")


# ─── Linear constraints ───────────────────────────────────────────────────────

def make_linear_constraints() -> tuple[list, list[tuple]]:
    """
    Build LinearConstraint objects and bounds from finger_config layout.
    New 3-param flexor layout: (f_min, f_min_inner, f_max) per joint.
    """
    N   = cfg.n_elements(DESIGN_LENGTHS)
    NF  = cfg.NUM_FLEX_PARAMS    # 9
    NE  = cfg.NUM_EXT_PARAMS     # 6
    NJP = cfg.NUM_JOINT_PARAMS   # 15

    bds = cfg.bounds(DESIGN_LENGTHS)
    constraints = []

    # ── 1a. f_max >= f_min_inner ──────────────────────────────────────────────
    A = np.zeros((cfg.NUM_JOINTS, N))
    for i in range(cfg.NUM_JOINTS):
        A[i, i*3+2] =  1   # f_max
        A[i, i*3+1] = -1   # f_min_inner
    constraints.append(LinearConstraint(A, lb=0, ub=np.inf, keep_feasible=True))

    # ── 1b. f_min_inner >= f_min ──────────────────────────────────────────────
    A = np.zeros((cfg.NUM_JOINTS, N))
    for i in range(cfg.NUM_JOINTS):
        A[i, i*3+1] =  1   # f_min_inner
        A[i, i*3+0] = -1   # f_min
    constraints.append(LinearConstraint(A, lb=0, ub=np.inf, keep_feasible=True))

    # ── 1c. e_max >= e_min ────────────────────────────────────────────────────
    A = np.zeros((cfg.NUM_JOINTS, N))
    for i in range(cfg.NUM_JOINTS):
        A[i, NF + i*2]   =  1
        A[i, NF + i*2+1] = -1
    constraints.append(LinearConstraint(A, lb=0, ub=np.inf, keep_feasible=True))

    # ── 2. f_max + e_min <= thickness ─────────────────────────────────────────
    A = np.zeros((cfg.NUM_JOINTS, N))
    for i in range(cfg.NUM_JOINTS):
        A[i, i*3+2]      =  1   # f_max
        A[i, NF + i*2+1] =  1   # e_min
    constraints.append(LinearConstraint(A, lb=-np.inf, ub=cfg.THICKNESS_LIMIT,
                                        keep_feasible=True))

    ## I can't remember what this does so I'm doing away with it for now
    # ── 3. e_max <= sec_limit ─────────────────────────────────────────────────
    # sec_limit = cfg.INITIAL_THICKNESS - 2.25/25.4 - 0.125
    # A = np.zeros((cfg.NUM_JOINTS, N))
    # for i in range(cfg.NUM_JOINTS):
    #     A[i, NF + i*2] = 1
    #     if RELAX_THICKNESS:
    #         A[i, -1]    = -1
    # ub = 0 if RELAX_THICKNESS else sec_limit
    # constraints.append(LinearConstraint(A, lb=-np.inf, ub=ub,
    #                                     keep_feasible=True))

    # ── 4. Overall length ─────────────────────────────────────────────────────
    if DESIGN_LENGTHS:
        # lengths are at indices NJP..NJP+1 (before thickness)
        A = np.zeros(N)
        A[NJP]   = 1
        A[NJP+1] = 1
        constraints.append(LinearConstraint(A, lb=0,
                                            ub=cfg.OVERALL_LENGTH - 0.75,
                                            keep_feasible=True))

    return constraints, bds


# ─── Population eval cache + vectorized wrappers ──────────────────────────────
# Same pattern as variableOptimizerGlobalParallel [3]

def _eval_population(population: np.ndarray) -> list[dict]:
    global _cache_key, _cache_results
    population = np.atleast_2d(population)
    if population.shape[0] != cfg.n_elements(DESIGN_LENGTHS):
        population = population.T
    pop_hash = hashlib.md5(population.tobytes()).hexdigest()
    if pop_hash == _cache_key:
        return _cache_results
    vectors        = [population[:, i] for i in range(population.shape[1])]
    _cache_results = _pool.map(parallel_evaluator.eval_all, vectors)
    _cache_key     = pop_hash
    return _cache_results

def _wrap(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr

def _vec_objective(population: np.ndarray) -> np.ndarray:
    population = np.atleast_2d(population)
    if population.shape[0] != cfg.n_elements(DESIGN_LENGTHS):
        population = population.T

    results = _eval_population(population)

    if OBJECTIVE_MODE == "thickness":
        # Thickness is always the last element per finger_config layout
        return np.array([r['thickness'] for r in results])

    if OBJECTIVE_MODE == "worst_case_tension":
        return np.array([float(np.max(r['tensions'])) for r in results])

    if OBJECTIVE_MODE == "worst_case_OVL":
        # Requires a nested minimize — expensive, done serially per individual
        def _ovl(v):
            Fing = build_finger_from_vector(v, DESIGN_LENGTHS)
            from scipy.optimize import minimize as _min
            res = _min(Fing.structure.get_magnitude,
                       x0=[0.0]*Fing.numJoints,
                       bounds=[(0, np.pi/2)]*Fing.numJoints)
            return float(-res.fun)
        return np.array([_ovl(population[:, i])
                         for i in range(population.shape[1])])

    if OBJECTIVE_MODE == "minimize_volume":
        return np.array([np.linalg.norm(r['task_volumes']) for r in results])

    if OBJECTIVE_MODE == "maximize_volume":
        return np.array([-np.linalg.norm(r['task_volumes']) for r in results])

    if OBJECTIVE_MODE == "bend_radius":
        # we want to maximize bend radius, so return the negative
        return np.array([-r['bend_radius'] for r in results])

    raise ValueError(f"Unknown objective: {OBJECTIVE_MODE}")

def _vec_thickness(population: np.ndarray) -> np.ndarray:
    results = _eval_population(population)
    per_joint = np.column_stack([r['per_joint_thickness'] for r in results])
    return _wrap(per_joint-cfg.THICKNESS_LIMIT)

def _vec_tensions(population: np.ndarray) -> np.ndarray:
    results = _eval_population(population)
    return _wrap(np.column_stack([r['tensions'] for r in results]))

def _vec_bend_radius(population: np.ndarray) -> np.ndarray:
    results = _eval_population(population)
    return _wrap(np.array([r['bend_radius'] for r in results]).reshape(1, -1))

def _vec_c_limit(population: np.ndarray) -> np.ndarray:
    results = _eval_population(population)
    return _wrap(np.column_stack([r['c_limit'] for r in results]))

def _vec_envelopes(population: np.ndarray) -> np.ndarray:
    results = _eval_population(population)
    return _wrap(np.column_stack([r['envelopes'] for r in results]))

def make_nonlinear_constraints() -> list:
    nl = []
    if CONTAIN_HUMAN:
        nl.append(NonlinearConstraint(_vec_envelopes, lb=-np.inf, ub=0))
    if not CONTAIN_HUMAN:
        nl.append(NonlinearConstraint(_vec_tensions,     lb=0, ub=TENSION_LIMIT))
    nl.append(NonlinearConstraint(_vec_bend_radius,  lb=0.472, ub=np.inf))
    nl.append(NonlinearConstraint(_vec_c_limit,      lb=0, ub=np.inf))
    nl.append(NonlinearConstraint(_vec_thickness,    lb=-np.inf, ub=0))
    return nl

# ─── Single run ───────────────────────────────────────────────────────────────

def run_one(v0: np.ndarray,
            lin_constraints: list,
            nl_constraints:  list,
            bds:             list[tuple],
            label:           str = "") -> OptimizeResult:
    """
    Run one optimization from starting point v0.
    Returns OptimizeResult with .x, .fun, .success.
    """
    global _cache_key, _cache_results, _pool
    _cache_key     = None   # flush cache between runs
    _cache_results = None

    all_constraints = lin_constraints + nl_constraints
    best_x   = [v0.copy()]
    best_fun = [np.inf]
    t_start  = time.perf_counter()

    def callback(xk, convergence=0.0):
        obj = _vec_objective(xk.reshape(-1, 1))[0]
        if obj < best_fun[0]:
            best_fun[0] = obj
            best_x[0]   = xk.copy()
        elapsed = time.perf_counter() - t_start
        print(f"  [{label}]  "
              f"obj={obj:.5f}  best={best_fun[0]:.5f}  "
              f"conv={convergence:.4f}  t={elapsed:.1f}s",
              flush=True)

    try:
        if METHOD == "de":
            result = differential_evolution(
                _vec_objective,
                bounds=bds,
                constraints=all_constraints,
                callback=callback,
                vectorized=True,
                workers=1,
                updating='deferred',
                init='sobol',
                x0=v0,
                mutation=(0.5, 1.5),
                recombination=0.7,
                popsize=15,
                maxiter=MAX_ITER,
                tol=TOL,
                seed=SEED,
                polish=POLISH,
            )

        elif METHOD == "trust-constr":
            def serial_obj(v):
                return _vec_objective(v.reshape(-1, 1))[0]

            def serial_tensions(v):
                return _vec_tensions(v.reshape(-1, 1))[:, 0] \
                       if _vec_tensions(v.reshape(-1,1)).ndim > 1 \
                       else _vec_tensions(v.reshape(-1,1))

            # Serial wrappers for trust-constr (doesn't support vectorized)
            serial_nl = []
            if CONTAIN_HUMAN:
                serial_nl.append(NonlinearConstraint(
                    fun=lambda v: _vec_envelopes(v.reshape(-1,1)).ravel(),
                    lb=-np.inf, ub=0))
            serial_nl.append(NonlinearConstraint(
                fun=lambda v: _vec_tensions(v.reshape(-1,1)).ravel(),
                lb=0, ub=TENSION_LIMIT))
            serial_nl.append(NonlinearConstraint(
                fun=lambda v: float(_vec_bend_radius(v.reshape(-1,1)).ravel()),
                lb=0.40, ub=np.inf))
            serial_nl.append(NonlinearConstraint(
                fun=lambda v: _vec_c_limit(v.reshape(-1,1)).ravel(),
                lb=0, ub=np.inf))

            result = minimize(
                serial_obj,
                v0,
                bounds=bds,
                constraints=lin_constraints + serial_nl,
                method='trust-constr',
                options={'maxiter': MAX_ITER, 'gtol': TOL, 'verbose': 1},
            )

        else:
            raise ValueError(f"Unknown method: {METHOD}")

    except KeyboardInterrupt:
        print(f"\n  [{label}] Interrupted")
        result = OptimizeResult(x=best_x[0], fun=best_fun[0],
                                success=False, message="KeyboardInterrupt")

    return result


# ─── Batch runner ─────────────────────────────────────────────────────────────

# def run_batch(seed_paths: list[str]) -> dict[str, OptimizeResult]:
#     """
#     Load each .json seed file, convert to design vector, run optimization.
#     Returns {filename_stem: OptimizeResult}.
#     """
#     lin_constraints, bds = make_linear_constraints()
#     nl_constraints       = make_nonlinear_constraints()
#     results = {}

#     for path in seed_paths:
#         label = Path(path).stem
#         print(f"\n{'═'*55}")
#         print(f"  Seed: {path}")
#         print(f"  Objective: {OBJECTIVE_MODE}  Method: {METHOD}")
#         print(f"{'═'*55}")

#         try:
#             jps = load_joint_params(path)
#             v0 = joint_params_to_vector(jps, DESIGN_LENGTHS)
#             print(f"  v0 = {np.array2string(v0, precision=4)}")
#             print(f"  n_elements = {len(v0)}")

#         except Exception as e:
#             print(f"  Failed to load {path}: {e}")
#             import traceback; traceback.print_exc()
#             continue

#         result = run_one(v0, lin_constraints, nl_constraints, bds, label=label)
#         results[label] = result

#         # ── Save .smx (flat vector) ───────────────────────────────────────────
#         out_smx = f"batch_result_{label}_{OBJECTIVE_MODE}.smx"
#         np.savetxt(out_smx, result.x)
#         print(f"\n  [{label}] Done — "
#               f"fun={result.fun:.6f}  success={result.success}")
#         print(f"  Saved → {out_smx}")

#         # ── Save .json (round-trips back to GUI) ──────────────────────────────
#         try:
#             result_jps = vector_to_joint_params(result.x,
#                                                 design_lengths=DESIGN_LENGTHS)
#             out_json = f"batch_result_{label}_{OBJECTIVE_MODE}.json"
#             import json
#             with open(out_json, 'w') as f:
#                 from dataclasses import asdict
#                 json.dump([asdict(jp) for jp in result_jps], f, indent=2)
#             print(f"  Saved → {out_json}")
#         except Exception as e:
#             print(f"  Failed to save json for {label}: {e}")

#     return results


# ─── Parallel runner ──────────────────────────────────────────────────────────

def _run_seed_process(args: dict) -> tuple[str, OptimizeResult]:
    """Runs in its own outer process with its own inner pool."""
    import hashlib, json
    from dataclasses import asdict
    from multiprocessing import Pool
    from pathlib import Path

    # Unpack — no globals, everything explicit
    path           = args['path']
    n_inner        = args['n_inner_workers']
    objective_mode = args['objective_mode']
    method         = args['method']
    design_lengths = args['design_lengths']
    tension_limit  = args['tension_limit']
    enforce_pf     = args['enforce_pf']
    contain_human  = args['contain_human']
    max_iter       = args['max_iter']
    tol            = args['tol']
    seed_          = args['seed']
    polish         = args['polish']
    grasps         = args['grasps']
    label          = Path(path).stem

    import signal
    def _worker_ignore_siginit():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # ── Own inner pool ────────────────────────────────────────────────────────
    init_args = (grasps, design_lengths, tension_limit, enforce_pf)
    print(f"  [{label}] inner pool started ({n_inner} workers), "
          f"loading seed...", flush=True)
    inner_pool = Pool(processes=n_inner,
                      initializer=lambda: (parallel_evaluator.worker_init(*init_args),
                                           _worker_ignore_siginit()),
                    #   initargs=init_args
                      )
    parallel_evaluator.worker_init(*init_args)   # init this outer process too [3]

    # ── Local cache (list so closures can rebind) ─────────────────────────────
    _ck = [None]
    _cr = [None]

    def eval_pop(population):
        population = np.atleast_2d(population)
        if population.shape[0] != cfg.n_elements(design_lengths):
            population = population.T
        h = hashlib.md5(population.tobytes()).hexdigest()
        if h == _ck[0]:
            return _cr[0]
        vectors = [population[:, i] for i in range(population.shape[1])]
        _cr[0]  = inner_pool.map(parallel_evaluator.eval_all, vectors)
        _ck[0]  = h
        return _cr[0]

    def wrap(arr):
        return arr[:, 0] if (arr.ndim == 2 and arr.shape[1] == 1) else arr

    # ── Local vectorized wrappers (closures over eval_pop) ────────────────────
    def vec_obj(pop):
        pop = np.atleast_2d(pop)
        if pop.shape[0] != cfg.n_elements(design_lengths): pop = pop.T
        res = eval_pop(pop)
        if objective_mode == "thickness":
            return np.array([r['thickness'] for r in res])
        if objective_mode == "worst_case_tension":
            return np.array([float(np.max(r['tensions'])) for r in res])
        if objective_mode == "minimize_volume":
            return np.array([float(np.linalg.norm(r['task_volumes'])) for r in res])
        if objective_mode == "maximize_volume":
            return np.array([-float(np.linalg.norm(r['task_volumes'])) for r in res])
        if objective_mode == "bend_radius":
            return np.array([-r['bend_radius'] for r in res])
        raise ValueError(f"Unknown objective: {objective_mode}")

    # Used for constraints
    def vec_envelopes(pop):
        return wrap(np.column_stack([r['envelopes'] for r in eval_pop(pop)]))
    def vec_tensions(pop):
        return wrap(np.column_stack([r['tensions'] for r in eval_pop(pop)]))
    def vec_bend_radius(pop):
        return wrap(np.array([r['bend_radius'] for r in eval_pop(pop)]).reshape(1,-1))
    def vec_c_limit(pop):
        return wrap(np.column_stack([r['c_limit'] for r in eval_pop(pop)]))
    def vec_thickness(pop):
        return wrap(np.column_stack(
            [r['per_joint_thickness'] for r in eval_pop(pop)]) - cfg.THICKNESS_LIMIT)

    # ── Constraints ───────────────────────────────────────────────────────────
    lin_constraints, bds = make_linear_constraints()   # pure, safe to call here
    nl = []
    if contain_human:
        nl.append(NonlinearConstraint(vec_envelopes,  lb=-np.inf, ub=0))
    else:
        nl.append(NonlinearConstraint(vec_tensions,       lb=0, ub=tension_limit))
    nl.append(NonlinearConstraint(vec_bend_radius,    lb=0.472, ub=np.inf))
    nl.append(NonlinearConstraint(vec_c_limit,        lb=0, ub=np.inf))
    nl.append(NonlinearConstraint(vec_thickness,      lb=-np.inf, ub=0))

    # ── Load seed ─────────────────────────────────────────────────────────────
    try:
        jps = load_joint_params(path)
        print(f"  [{label}] Loaded joint params")
        v0  = joint_params_to_vector(jps, design_lengths)
        print(f"  [{label}] Initial vector loaded: {v0}")
    except Exception as e:
        inner_pool.close(); inner_pool.join()
        return label, OptimizeResult(x=np.zeros(cfg.n_elements(design_lengths)),
                                     fun=np.inf, success=False,
                                     message=f"Seed load failed: {e}")

    # ── Optimize ──────────────────────────────────────────────────────────────
    best_x   = [v0.copy()]
    best_fun = [np.inf]
    t0       = time.perf_counter()

    def callback_de(xk, convergence=0.0):
        obj = vec_obj(xk.reshape(-1,1))[0]
        if obj < best_fun[0]:
            best_fun[0] = obj
            best_x[0]   = xk.copy()
        t = time.perf_counter() - t0
        vec_str = np.array2string(best_x[0], precision=6,
                                separator=',', suppress_small=True)
        print(f"GEN t={t:.1f} obj={obj:.6f} best={best_fun[0]:.6f} "
            f"conv={convergence:.6f} x={vec_str}", flush=True)

    def callback_tc(intermediate_result):
        xk  = intermediate_result.x
        obj = vec_obj(xk.reshape(-1,1))[0]
        t   = time.perf_counter() - t0

        con_vals = {
            # 'tensions':  float(np.max(vec_tensions(xk.reshape(-1,1)))),
            'bend_r':    float(vec_bend_radius(xk.reshape(-1,1)).ravel()[0]),
            'thickness': float(np.max(vec_thickness(xk.reshape(-1,1)))),
            'c_limit':   float(np.min(vec_c_limit(xk.reshape(-1,1)))),
        }
        if contain_human:
            con_vals['envelope'] = float(np.max(vec_envelopes(xk.reshape(-1,1))))

        if obj < best_fun[0]:
            best_fun[0] = obj
            best_x[0]   = xk.copy()

        con_str = "  ".join(f"{k}={v:.6f}" for k, v in con_vals.items())
        vec_str = np.array2string(xk, precision=6,
                                separator=',', suppress_small=True)
        print(f"  [{label}] iter={intermediate_result.nit} t={t:.1f} "
            f"obj={obj:.6f} {con_str} x={vec_str}", flush=True)

    try:
        print(f"  [{label}] starting {method} optimization, "
              f"n={len(v0)} vars, {len(lin_constraints+nl)} constraints",
                flush=True)
        if method == "de":
            result = differential_evolution(
                vec_obj, bounds=bds, constraints=lin_constraints+nl,
                callback=callback_de, vectorized=True, workers=1,
                updating='deferred', init='sobol', x0=v0,
                mutation=(0.5,1.5), recombination=0.7, popsize=15,
                maxiter=max_iter, tol=tol, seed=seed_, polish=polish,
            )
        elif method == "trust-constr":
            s = lambda fn: (lambda v: fn(v.reshape(-1,1)).ravel())
            s_nl = []
            if contain_human:
                s_nl.append(NonlinearConstraint(s(vec_envelopes), lb=-np.inf, ub=0))
            else:
                s_nl.append(NonlinearConstraint(s(vec_tensions), lb=0, ub=tension_limit))
            s_nl.append(NonlinearConstraint(
                lambda v: float(vec_bend_radius(v.reshape(-1,1)).ravel()[0]),
                lb=0.472, ub=np.inf))
            s_nl.append(NonlinearConstraint(s(vec_c_limit), lb=0, ub=np.inf))
            s_nl.append(NonlinearConstraint(s(vec_thickness), lb=-np.inf, ub=0))
            result = minimize(lambda v: vec_obj(v.reshape(-1,1))[0], v0,
                              bounds=bds, constraints=lin_constraints+s_nl,
                              method='trust-constr',
                              callback=callback_tc,
                              options={'maxiter': max_iter, 'gtol': tol,
                                       'verbose': 1})
        else:
            raise ValueError(f"Unknown method: {method}")
    except KeyboardInterrupt:
        result = OptimizeResult(x=best_x[0], fun=best_fun[0],
                                success=False, message="KeyboardInterrupt")
    except Exception as e:
        import traceback
        print(f"    [{label}] OPTIMIZATION FAILED because: {e}", flush=True)
        traceback.print_exc()
        result = OptimizeResult(x=best_x[0], fun=best_fun[0],
                                        success=False, message=str(e))
    finally:
        inner_pool.close(); inner_pool.join()
        print(f"  [{label}] inner pool shut down.", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savetxt(f"batch_result_{label}_{objective_mode}_{"PF" if enforce_pf else ""}.smx", result.x)
    try:
        result_jps = vector_to_joint_params(result.x, design_lengths=design_lengths)
        with open(f"batch_result_{label}_{objective_mode}_{"PF" if enforce_pf else ""}.json", 'w') as f:
            json.dump([asdict(jp) for jp in result_jps], f, indent=2)
    except Exception as e:
        print(f"  [{label}] json save failed: {e}")

    return label, result

# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(seed_paths: list[str]) -> dict[str, OptimizeResult]:
    n_seeds   = len(seed_paths)
    n_inner   = max(1, WORKERS // n_seeds)
    if CONTAIN_HUMAN:
        grasps = []
    else:
        grasps    = make_grasps()

    print(f"  {n_seeds} seeds × {n_inner} inner workers "
          f"= {n_seeds * n_inner}/{WORKERS} cores")

    args_list = [
        dict(path=path, n_inner_workers=n_inner,
             objective_mode=OBJECTIVE_MODE, method=METHOD,
             design_lengths=DESIGN_LENGTHS, tension_limit=TENSION_LIMIT,
             enforce_pf=ENFORCE_PF, contain_human=CONTAIN_HUMAN,
             max_iter=MAX_ITER, tol=TOL, seed=SEED, polish=POLISH,
             grasps=grasps)
        for path in seed_paths
    ]

    results = {}
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_seeds) as ex:
            futures = {ex.submit(_run_seed_process, a): a['path'] for a in args_list}
            for fut in concurrent.futures.as_completed(futures):
                path = futures[fut]
                try:
                    label, result = fut.result()
                    results[label] = result
                    print(f"\n  [{label}] done — fun={result.fun:.6f} "
                        f"success={result.success}")
                except Exception as e:
                    import traceback
                    print(f"\n  [{Path(path).stem}] FAILED:")
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("\nCtrl-C received — cancelling remaining seeds...", flush=True)
        for fut in futures:
            fut.cancel()
    return results

# ─── Post-processing / reporting ──────────────────────────────────────────────

def report(results: dict[str, OptimizeResult]):
    """Print a summary table of all batch results."""
    print(f"\n{'═'*55}")
    print(f"  BATCH COMPLETE — {len(results)} runs")
    print(f"{'═'*55}")
    print(f"  {'label':<30} {'obj':>10} {'success':>8}")
    print(f"  {'-'*48}")
    for label, res in sorted(results.items(), key=lambda x: x[1].fun):
        print(f"  {label:<30} {res.fun:>10.5f} {str(res.success):>8}")
    print(f"{'═'*55}\n")

    # Best result
    best_label = min(results, key=lambda k: results[k].fun)
    best_res   = results[best_label]
    print(f"  Best: {best_label}  obj={best_res.fun:.6f}")
    try:
        best_jps = vector_to_joint_params(best_res.x, design_lengths=DESIGN_LENGTHS)
        for jp in best_jps:
            print(f"    joint {jp.joint_idx}: "
                    f"f=({jp.f_min:.4f}, {jp.f_min_inner:.4f}, {jp.f_max:.4f})  "
                    f"e=({jp.e_min:.4f}, {jp.e_max:.4f})  "
                    f"l={jp.length:.4f}")
    except Exception as e:
        print(f"Could not decode output result because {e}")
        print(f" Tried to decode {best_res.x}")


# ─── Argparse ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch finger optimizer — runs one optimization per seed file"
    )
    parser.add_argument(
        'seeds', nargs='+',
        help="Path(s) to .json JointParams seed files"
    )
    parser.add_argument(
        '-o', '--objective',
        choices=['thickness',
                 'worst_case_OVL',
                 'worst_case_tension',
                 'minimize_volume',
                 'maximize_volume',
                 'bend_radius'],
        default='thickness',
        help="Objective function (default: thickness)"
    )
    parser.add_argument(
        '-m', '--method',
        choices=['de', 'trust-constr'],
        default='de',
        help="Optimization method (default: de)"
    )
    parser.add_argument(
        '--no-contain-human', dest='contain_human',
        action='store_false', default=True,
        help="Disable human capability envelope constraint"
    )
    parser.add_argument(
        '--design-lengths', action='store_true', default=False,
        help="Allow link lengths to be design variables"
    )
    parser.add_argument(
        '-t', '--tension-limit', type=float, default=cfg.TENSION_LIMIT,
        help=f"Max allowable tension (default: {cfg.TENSION_LIMIT})"
    )
    parser.add_argument(
        '-th', '--thickness-limit', type=float, default=cfg.THICKNESS_LIMIT,
        help=f"Max allowable thickness (default: {cfg.TENSION_LIMIT})"
    )
    parser.add_argument(
        '--maxiter', type=int, default=200,
        help="Max iterations (default: 200)"
    )
    parser.add_argument(
        '--enforce-pf', action='store_true', default=True,
        help="Evaluate task-space capability in terms of point forces"
    )
    parser.add_argument(
        '--no-enforce-pf', dest='enforce_pf',
        action='store_false',
        help="Evaluate task-space capability allowing for out-of-plane torques"
    )
    parser.add_argument(
        '--polish', action='store_true', default=False,
        help="Run local polish after DE"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=mp.cpu_count(),
        help=f"Worker processes (default: {mp.cpu_count()})"
    )
    return parser.parse_known_args()


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    args, _ = parse_args()

    # Push args into module globals
    global OBJECTIVE_MODE, METHOD, CONTAIN_HUMAN
    global DESIGN_LENGTHS, TENSION_LIMIT, MAX_ITER, POLISH, SEED, WORKERS
    global ENFORCE_PF
    OBJECTIVE_MODE  = args.objective
    METHOD          = args.method
    CONTAIN_HUMAN   = args.contain_human
    ENFORCE_PF      = args.enforce_pf
    DESIGN_LENGTHS  = args.design_lengths
    TENSION_LIMIT   = args.tension_limit
    MAX_ITER        = args.maxiter
    POLISH          = args.polish
    SEED            = args.seed
    WORKERS         = args.workers

    print(f"\n{'═'*55}")
    print(f"  Batch Optimizer")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Objective:  {OBJECTIVE_MODE}")
    print(f"  Method:     {METHOD}")
    print(f"  Workers:    {WORKERS}")
    print(f"  Lengths:    {'designed' if DESIGN_LENGTHS else 'fixed'}")
    print(f"  Human env:  {CONTAIN_HUMAN}, with Enforce pf: {ENFORCE_PF}")
    print(f"  Tension:    {TENSION_LIMIT}")
    print(f"{'═'*55}\n")


    # if CONTAIN_HUMAN:
    #     grasps = []
    # else:
    #     grasps = make_grasps()

    # parallel_evaluator.worker_init still expects the old signature [1]

    # _init_pool(grasps)
    # print(f"  Pool started: {WORKERS} workers\n")

    try:
        results = run_batch(args.seeds)
        report(results)
    finally:
        print("  Done.")

def test():
    import finger_config as cfg
    from finger_config import JointParams, joint_params_to_vector

    grasps = make_grasps()
    _init_pool(grasps)   # starts the pool

    try:
        # Build a test population of 5 individuals
        base_jps = [JointParams(i) for i in range(cfg.NUM_JOINTS)]
        v0 = joint_params_to_vector(base_jps)
        rng = np.random.default_rng(42)
        lo  = np.array([b[0] for b in cfg.bounds(DESIGN_LENGTHS)])
        hi  = np.array([b[1] for b in cfg.bounds(DESIGN_LENGTHS)])
        pop = rng.uniform(lo, hi, size=(5, len(v0))).T  # (N, S)

        print(f"Population shape: {pop.shape}")

        # Test each vectorized wrapper
        import time

        t0 = time.perf_counter()
        results = _eval_population(pop)
        print(f"_eval_population: {(time.perf_counter()-t0)*1000:.1f}ms  "
              f"({len(results)} results)")

        t0 = time.perf_counter()
        obj = _vec_objective(pop)
        print(f"_vec_objective:   {(time.perf_counter()-t0)*1000:.1f}ms  {obj}")

        t0 = time.perf_counter()
        ten = _vec_tensions(pop)
        print(f"_vec_tensions:    {(time.perf_counter()-t0)*1000:.1f}ms  "
              f"shape={ten.shape}  max={ten.max():.3f}")

        t0 = time.perf_counter()
        br = _vec_bend_radius(pop)
        print(f"_vec_bend_radius: {(time.perf_counter()-t0)*1000:.1f}ms  {br}")

        t0 = time.perf_counter()
        cl = _vec_c_limit(pop)
        print(f"_vec_c_limit:     {(time.perf_counter()-t0)*1000:.1f}ms  "
              f"shape={cl.shape}")

        if CONTAIN_HUMAN:
            t0 = time.perf_counter()
            env = _vec_envelopes(pop)
            print(f"_vec_envelopes:   {(time.perf_counter()-t0)*1000:.1f}ms  "
                  f"shape={env.shape}")

        # Test cache hit — second call should be ~0ms
        t0 = time.perf_counter()
        _eval_population(pop)
        print(f"cache hit:        {(time.perf_counter()-t0)*1000:.1f}ms  "
              f"(should be ~0)")

        # Test linear constraints are buildable
        lin, bds = make_linear_constraints()
        print(f"Linear constraints: {len(lin)}  bounds: {len(bds)}")
        for i, lc in enumerate(lin):
            assert lc.A.shape[1] == len(v0), \
                f"Constraint {i} width {lc.A.shape[1]} != {len(v0)}"
        print("Linear constraint shapes: OK")

        print("\nAll smoke tests PASSED")

    finally:
        _pool.close()
        _pool.join()
        print("Pool shut down.")

if __name__ == "__main__":
    main()
    # test()