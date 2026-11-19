from finger import valero, valeroFinger, dof_constraints, Finger
from strucMatrices import StrucMatrix, diagonal, D_ICS
from utils import hull_containment_scalar, colors
from BRANCH_human_finger_dev.ValeroCuervasModel import pose_dict

from variableOptimizerGlobal import FingerEvaluator, stallException

import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import concurrent.futures
import hashlib

# ─── Finger parameters ────────────────────────────────────────────────────────

D           = D_ICS
effort_tjps = np.array(list(zip(
    np.array(np.where(D != 0))[0, :],
    np.array(np.where(D != 0))[1, :]
)))
lengths = [1.4, 1.4, 1.2]
bounds  = [(.125, 0.5)] * len(effort_tjps)

# ─── Valero capabilities (computed once in main process) ──────────────────────

def _compute_vc_constraints():
    qf = pose_dict['Flexion']
    qi = pose_dict['Intermediate']
    qe = pose_dict['Extension']
    poses      = [qf, qi, qe]
    pose_names = ['Flexion', 'Intermediate', 'Extension']
    capabilities = [
        valeroFinger.get_convex_capability(
            q, valero.Fo(q), dof_constraints[4]['task_dof'], space='task'
        )
        for q in poses
    ]
    return {
        name: {'pose': pose, 'capability': cap}
        for name, pose, cap in zip(pose_names, poses, capabilities)
    }


# ─── Worker-level globals (one set per worker process) ───────────────────────
# These are populated by worker_init, called once per worker at pool startup.

_pose_name      = None
_pose           = None
_ref_capability = None


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
    S = StrucMatrix(R, D, F=[50] * D.shape[-1], name="staticFinger_worker")
    return Finger(S, lengths)


def eval_all(v):
    Fing            = _build_finger(np.asarray(v))
    capability_hull = Fing.get_convex_capability(
        _pose[1:], constrained_dof=[], output_dim=2, space='task'
    )
    return {
        'objective':   capability_hull.volume,
        'containment': hull_containment_scalar(_ref_capability, capability_hull),
    }


# ─── Per-pose optimization routine ───────────────────────────────────────────
# This entire function runs in its own outer process (one per pose).
# It spawns its own inner Pool for population-level parallelism.

def optimize_pose(pose_name, pose, ref_capability, n_inner_workers, seed=42):
    """
    Runs in an outer process. Owns its own inner Pool.
    Returns (pose_name, OptimizeResult).
    """
    # Must set spawn so inner Pool doesn't fork a fork
    mp.set_start_method('spawn', force=True)

    cache_key     = [None]
    cache_results = [None]

    inner_pool = Pool(
        processes=n_inner_workers,
        initializer=worker_init,
        initargs=(pose_name, pose, ref_capability),
    )
    # Init this (outer) process too for callback use
    worker_init(pose_name, pose, ref_capability)

    def _eval_population(population):
        population = np.atleast_2d(population)
        if population.shape[0] != len(effort_tjps):
            population = population.T
        pop_hash = hashlib.md5(population.tobytes()).hexdigest()
        if pop_hash == cache_key[0]:
            return cache_results[0]
        vectors           = [population[:, i] for i in range(population.shape[1])]
        cache_results[0]  = inner_pool.map(eval_all, vectors)
        cache_key[0]      = pop_hash
        return cache_results[0]

    def _wrap(arr):
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0]
        return arr

    def _vec_objective(population):
        return np.array([r['objective'] for r in _eval_population(population)])

    def _vec_containment(population):
        return _wrap(
            np.array([r['containment']
                      for r in _eval_population(population)]).reshape(1, -1)
        )

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
        # Each pose prints with its name as prefix so interleaved output is readable
        print(f"  [{pose_name:12s}]  obj={obj:.5f}  "
              f"cont={con:.4f}  "
              f"conv={convergence:.4f}  "
              f"best={best_fun[0]:.5f}",
              flush=True)

    try:
        result = differential_evolution(
            _vec_objective,
            bounds=bounds,
            constraints=[constraint],
            callback=callback,
            vectorized=True,
            workers=1,
            updating='deferred',
            init='sobol',
            mutation=(0.5, 1.5),
            recombination=0.7,
            popsize=15,
            maxiter=500,
            tol=1e-4,
            seed=seed,
            polish=True,
        )
    except KeyboardInterrupt:
        print(f"\n  [{pose_name}] Interrupted — returning best so far", flush=True)
        result = OptimizeResult(
            x       = best_x[0] if best_x[0] is not None
                      else np.array([0.4] * len(effort_tjps)),
            fun     = best_fun[0],
            success = False,
            message = "KeyboardInterrupt",
        )
    finally:
        inner_pool.close()
        inner_pool.join()
        print(f"  [{pose_name}] Inner pool shut down.", flush=True)

    return pose_name, result


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    VC_constraints = _compute_vc_constraints()
    results        = {}
    replace        = True

    if not replace:
        n_poses        = len(VC_constraints)                   # 3
        total_cores    = mp.cpu_count()
        # Give each pose an equal share of cores for its inner pool.
        # E.g. 12 cores, 3 poses → 4 inner workers each.
        # Floor at 1 so it always works even on small machines.
        n_inner        = max(1, total_cores // n_poses)

        print(f"Parallelism: {n_poses} poses × {n_inner} inner workers "
              f"= {n_poses * n_inner}/{total_cores} cores")

        # Launch one outer process per pose
        futures = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_poses
        ) as outer_executor:
            for pose_name, info in VC_constraints.items():
                fut = outer_executor.submit(
                    optimize_pose,
                    pose_name,
                    info['pose'],
                    info['capability'],
                    n_inner,
                    seed=42,
                )
                futures[fut] = pose_name

            for fut in concurrent.futures.as_completed(futures):
                pose_name = futures[fut]
                try:
                    name, result   = fut.result()
                    results[name]  = result
                    print(f"\n  [{name}] Done. "
                          f"fun={result.fun:.6f}  "
                          f"success={result.success}")
                except Exception as e:
                    import traceback
                    print(f"\n  [{pose_name}] FAILED:")
                    traceback.print_exc()
                    # Store a placeholder so plotting still runs for other poses
                    results[pose_name] = OptimizeResult(
                        x       = np.array([0.4] * len(effort_tjps)),
                        fun     = np.inf,
                        success = False,
                        message = str(e),
                    )

        # Save
        for name, result in results.items():
            np.savetxt(f"result_{name}.smx", result.x)
            print(f"  Saved result_{name}.smx")

    else:
        for name in ['Flexion', 'Intermediate', 'Extension']:
            x             = np.loadtxt(f"result_{name}.smx")
            results[name] = OptimizeResult(x=x)
        print("Results loaded.")

    print("\n\n\n\n\n")

    # ─── Plotting ─────────────────────────────────────────────────────────────
    # Recompute VC_constraints in main process for plotting
    # (may already exist if not replace, but need it for the load branch too)
    if 'VC_constraints' not in dir():
        VC_constraints = _compute_vc_constraints()

    for name, result in results.items():
        plt.figure()
        ax = plt.gca()

        print(result.x)

        # Reconstruct finger in main process for plotting
        v = np.asarray(result.x)
        R = np.zeros_like(D, dtype=float)
        for i, tjp in enumerate(effort_tjps):
            R[*tjp] = v[i]
        S             = StrucMatrix(R, D, F=[50] * D.shape[-1], name=f"plot_{name}")
        result_finger = Finger(S, lengths)

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
        still_compare_ax = valeroFinger.plot_Capability(vc_capability, ax=compare_ax, color=colors[0])
        still_compare_ax.set_title(f"local optimizer for {name} pose")

        print(f"For VC Pose {name}")
        print(f"The minimal moment arms required at a pose are")
        print(f"{result_finger.structure.S}")
        print(f"With an OTV of {result_finger.structure.get_magnitude()}")

    plt.show()