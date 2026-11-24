# parallel_evaluator.py
# Worker functions only — All structure comes from finger_config.py

import numpy as np
import functools
from scipy.spatial import ConvexHull
from finger import valeroFinger, dof_constraints, Finger
from utils import hull_containment_scalar
import finger_config as cfg

# ─── Worker-process globals (set by worker_init) ──────────────────────────────
_grasps          = None
_design_lengths  = False
_tension_limit   = cfg.TENSION_LIMIT
_enforce_pf      = True

def worker_init(grasps, design_lengths=False,
                        tension_limit=cfg.TENSION_LIMIT,
                        enforce_pf=True):
    global _grasps, _design_lengths, _tension_limit, _enforce_pf
    _grasps         = grasps
    _design_lengths = design_lengths
    _tension_limit  = tension_limit
    _enforce_pf     = enforce_pf
    # Note: R, D, lengths etc. all come from finger_config — no need to pass them

@functools.lru_cache(maxsize=8)
def _cached_build_finger(v_tuple: tuple,
                          design_lengths: bool,
                          tension_limit: float) -> Finger:
    return cfg.build_finger_from_vector(
        np.array(v_tuple),
        design_lengths=design_lengths,
        tension_limit=tension_limit,
    )

def _get_finger(v: np.ndarray) -> Finger:
    return _cached_build_finger(tuple(v), _design_lengths, _tension_limit)

# eval helpers

def _tensions(finger: Finger):
    all_tensions = []
    for g in _grasps:
        grip = finger.grasp_to_grip(g)
        all_tensions.extend(finger.grip_to_tensions(g.q, grip))
    return all_tensions

def _worst_case_bend_radius(finger: Finger):
    rs = [np.min((fn.r, (fn.r1 if hasattr(fn, 'r1') else np.inf)))
                                     for fn in finger.structure.effortFunctions]
    return float(np.min(rs))

def _c_limit_on_lengths(finger: Finger):
    length_limits = np.array([finger.lengths[i] if i==len(finger.lengths)-1 \
                                                else finger.lengths[i]/2 \
                                           for i in range(len(finger.lengths))])
    curve_lengths = np.array([
              finger.structure.indexed_effort_functions[i,i+1].c*np.cos(np.pi/4)
                                              for i in range(finger.numJoints)])
    exceedances = length_limits-curve_lengths
    return exceedances

def _get_capabilities_for_eval(finger: Finger) -> list[ConvexHull]:
    design_caps = []
    for name, pose in valeroFinger.structure.poses.items():
        if _enforce_pf:
            constrained_dof=dof_constraints[finger.numJoints]['task_dof']
        else:
            constrained_dof = []
        designCap = finger.get_convex_capability(
            pose[1:],
            constrained_dof=constrained_dof,
            output_dim=2,
            space='task'
        )
        design_caps.append(designCap)
    return design_caps

def _contains_human(task_capacities):
    containment = []
    # design_caps = _get_capabilities_for_eval(finger)
    human_caps = [valeroFinger.inPlaneCapabilities[name] for name in
                                            valeroFinger.structure.poses.keys()]
    for humanCap, designCap in zip(human_caps, task_capacities):
        containment.append(hull_containment_scalar(humanCap, designCap))
    return containment

# ─── eval_all: single call per individual, builds finger once ─────────────────

def eval_all(v: np.ndarray) -> dict:

    Fing = _get_finger(v)
    jps = cfg.vector_to_joint_params(v, design_lengths=_design_lengths)
    design_capacities = _get_capabilities_for_eval(Fing)

    all_tensions = _tensions(Fing)
    bend_radius = _worst_case_bend_radius(Fing)
    c_limit = _c_limit_on_lengths(Fing)
    containment = _contains_human(design_capacities)
    overall_thickness, contribution_per_joint = cfg.compute_rich_thickness(jps,
                                                                           Fing)

    return {
        'tensions':    np.array(all_tensions),
        'bend_radius': bend_radius,
        'c_limit':     c_limit,
        'envelopes':   np.array(containment),
        'task_volumes': np.array([task_cap.volume for
                                                 task_cap in design_capacities]),
        'thickness': overall_thickness,
        'per_joint_thickness': contribution_per_joint
    }

# In parallel_evaluator.py — test worker functions directly
if __name__ == "__main__":
    from finger import Grasp
    import finger_config as cfg
    import numpy as np

    # Minimal init — no pool needed, just call worker_init in main process
    grasps = []  # empty grasps — just tests finger build + hull
    worker_init(grasps, design_lengths=False, tension_limit=cfg.TENSION_LIMIT)

    # Test vector — middle of bounds
    v = cfg.joint_params_to_vector(
        [cfg.JointParams(i) for i in range(cfg.NUM_JOINTS)],
        overall_thickness=cfg.INITIAL_THICKNESS,
    )
    print(f"v = {v}")

    # Test finger build
    Fing = _get_finger(v)
    print(f"Finger built: {Fing.numJoints} joints, lengths={Fing.lengths}")

    # Test cache hit
    Fing2 = _get_finger(v)
    assert Fing is Fing2, "Cache miss — should be same object"
    print(f"Cache: OK  {_cached_build_finger.cache_info()}")

    # Test eval_all (no grasps so tensions will be empty)
    result = eval_all(v)
    print(f"eval_all keys:    {list(result.keys())}")
    print(f"bend_radius:      {result['bend_radius']:.4f}")
    print(f"c_limit:          {result['c_limit']}")
    print(f"envelopes:        {result['envelopes']}")
    print("parallel_evaluator smoke test PASSED")