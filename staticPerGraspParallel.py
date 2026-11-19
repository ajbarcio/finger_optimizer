from finger import valero, valeroFinger, dof_constraints, Finger
from strucMatrices import StrucMatrix, diagonal, D_ICS
from utils import hull_containment_scalar, colors
from BRANCH_human_finger_dev.ValeroCuervasModel import pose_dict

from variableOptimizerGlobal import FingerEvaluator, stallException

import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import hashlib

# ─── Valero Finger Information ────────────────────────────────────────────────

qf = pose_dict['Flexion']
qi = pose_dict['Intermediate']
qe = pose_dict['Extension']

poses      = [qf, qi, qe]
pose_names = ['Flexion', 'Intermediate', 'Extension']

capability_f = valeroFinger.get_convex_capability(qf, valero.Fo(qf),
                                                  dof_constraints[4]['task_dof'],
                                                  space='task')
capability_i = valeroFinger.get_convex_capability(qi, valero.Fo(qi),
                                                  dof_constraints[4]['task_dof'],
                                                  space='task')
capability_e = valeroFinger.get_convex_capability(qe, valero.Fo(qe),
                                                  dof_constraints[4]['task_dof'],
                                                  space='task')

VC_capabilities = [capability_f, capability_i, capability_e]
VC_constraints  = {
    name: {'pose': pose, 'capability': capability}
    for name, pose, capability in zip(pose_names, poses, VC_capabilities)
}

# ─── Designed Finger Parameters ───────────────────────────────────────────────

D           = D_ICS
effort_tjps = np.array(list(zip(
    np.array(np.where(D != 0))[0, :],
    np.array(np.where(D != 0))[1, :]
)))
lengths     = [1.4, 1.4, 1.2]
bounds      = [(.125, 0.5)] * len(effort_tjps)

# ─── Worker globals (populated by worker_init in each process) ────────────────

_pose_name       = None
_pose            = None
_ref_capability  = None

def worker_init(pose_name, pose, ref_capability):
    global _pose_name, _pose, _ref_capability
    _pose_name      = pose_name
    _pose           = pose
    _ref_capability = ref_capability


def _build_finger(v):
    v = np.asarray(v)
    R = np.zeros_like(D, dtype=float)
    for i, tjp in enumerate(effort_tjps):
        R[*tjp] = v[i]
    S    = StrucMatrix(R, D, F=[50] * D.shape[-1], name="staticFinger_worker")
    return Finger(S, lengths)


def eval_all(v):
    """
    Build finger once, return both objective and constraint value.
    Keyed on pose set in worker_init.
    """
    Fing = _build_finger(v)
    capability_hull = Fing.get_convex_capability(
        _pose[1:], constrained_dof=[], output_dim=2, space='task'
    )
    containment = hull_containment_scalar(_ref_capability, capability_hull)
    return {
        'objective':   capability_hull.volume,
        'containment': containment,
    }


# ─── Per-pose optimizer ───────────────────────────────────────────────────────

# Module-level population eval cache (keyed by pose_name + population hash)
_cache_key:     str        = None
_cache_results: list[dict] = None
_pool:          Pool       = None


def _eval_population(population: np.ndarray) -> list[dict]:
    global _cache_key, _cache_results
    population = np.atleast_2d(population)
    if population.shape[0] != len(effort_tjps):
        population = population.T
    pop_hash  = hashlib.md5(population.tobytes()).hexdigest()
    if pop_hash == _cache_key:
        return _cache_results
    vectors       = [population[:, i] for i in range(population.shape[1])]
    _cache_results = _pool.map(eval_all, vectors)
    _cache_key     = pop_hash
    return _cache_results


def _wrap(arr: np.ndarray) -> np.ndarray:
    """Squeeze (M, 1) → (M,) for scalar probes; leave (M, S) alone."""
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _vec_objective(population: np.ndarray) -> np.ndarray:
    results = _eval_population(population)
    return np.array([r['objective'] for r in results])        # (S,)


def _vec_containment(population: np.ndarray) -> np.ndarray:
    results = _eval_population(population)
    return _wrap(
        np.array([r['containment'] for r in results]).reshape(1, -1)
    )                                                          # (1, S) or (1,)


def run_pose(pose_name: str) -> OptimizeResult:
    """
    Run DE for a single pose. The pool is already initialised for this pose
    before this is called.
    """
    global _cache_key, _cache_results

    # Reset cache between poses
    _cache_key     = None
    _cache_results = None

    constraint = NonlinearConstraint(
        fun=_vec_containment,
        lb=-np.inf,
        ub=0.0,
    )

    best_x   = [None]
    best_fun = [np.inf]

    def callback(xk, convergence=0.0):
        obj = _vec_objective(xk.reshape(-1, 1))[0]
        con = _vec_containment(xk.reshape(-1, 1))[0]
        if obj < best_fun[0]:
            best_fun[0] = obj
            best_x[0]   = xk.copy()
        print(f"  [{pose_name}]  obj={obj:.5f}  "
              f"containment={con:.4f}  "
              f"convergence={convergence:.4f}  "
              f"best={best_fun[0]:.5f}")

    try:
        result = differential_evolution(
            _vec_objective,
            bounds=bounds,
            constraints=[constraint],
            callback=callback,
            vectorized=True,
            workers=1,            # pool is managed externally
            updating='deferred',  # required for vectorized=True
            init='sobol',
            mutation=(0.5, 1.5),
            recombination=0.7,
            popsize=15,
            maxiter=500,
            tol=1e-4,
            seed=42,
            polish=True,          # local trust-constr polish after DE
        )
    except KeyboardInterrupt:
        print(f"\n  [{pose_name}] Interrupted — returning best so far")
        result = OptimizeResult(
            x   = best_x[0] if best_x[0] is not None else
                  np.array([0.4] * len(effort_tjps)),
            fun = best_fun[0],
            success = False,
            message = "KeyboardInterrupt",
        )

    return result


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    replace = False

    if not replace:
        n_workers = mp.cpu_count()

        for pose_name in VC_constraints.keys():
            print(f"\n{'═' * 50}")
            print(f"  Optimizing pose: {pose_name}")
            print(f"  Workers: {n_workers}")
            print(f"{'═' * 50}")

            pose           = VC_constraints[pose_name]['pose']
            ref_capability = VC_constraints[pose_name]['capability']

            # Fresh pool per pose — each pool is initialised for exactly one pose
            _pool = Pool(
                processes=n_workers,
                initializer=worker_init,
                initargs=(pose_name, pose, ref_capability),
            )
            # Also init main process so callback calls work
            worker_init(pose_name, pose, ref_capability)

            try:
                results[pose_name] = run_pose(pose_name)
            finally:
                _pool.close()
                _pool.join()
                print(f"  Pool for {pose_name} shut down.")

        # Save
        for name, result in results.items():
            np.savetxt(f"result_{name}.smx", result.x)
            print(f"  Saved result_{name}.smx")

    else:
        for name in VC_constraints.keys():
            x              = np.loadtxt(f"result_{name}.smx")
            results[name]  = OptimizeResult(x=x)
        print("Results loaded.")

    print("\n\n\n\n\n")

    # ─── Plotting ─────────────────────────────────────────────────────────────
    for name, result in results.items():
        plt.figure()
        ax = plt.gca()

        print(result.x)

        evaluator      = StaticFingerEvaluator(name)
        result_finger  = evaluator.build_finger(result.x)

        result_capability = result_finger.get_convex_capability(
            pose_dict[name][1:],
            constrained_dof=[],
            output_dim=2,
            space='task'
        )
        vc_capability = VC_constraints[name]['capability']

        compare_ax = result_finger.plot_Capability(
            result_capability, ax=ax, color=colors[-1]
        )
        valeroFinger.plot_Capability(vc_capability, ax=compare_ax, color=colors[0])

        print(f"For VC Pose {name}")
        print(f"The minimal moment arms required at a pose are")
        print(f"{result_finger.structure.S}")

    plt.show()