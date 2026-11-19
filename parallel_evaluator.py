# worker.py
# All worker functions must be at module level for multiprocessing pickling.
# This module is imported by both the main process and worker processes.

import numpy as np
from strucMatrices import VariableStrucMatrix
from finger import Finger, valeroFinger, dof_constraints
from utils import hull_containment_scalar
from scipy.spatial import ConvexHull

import functools

# ─── These are set once by the main process via worker_init ──────────────────
_R = None
_D = None
_numFlexs = None
_numExts = None
_numTendons = None
_numLengthParameters = None
_DESIGN_LENGTHS = None
_lengths = None
_overall_length = None
_grasps = None


# cache to make more efficient
@functools.lru_cache(maxsize=8)
def _cached_build_finger(v_tuple: tuple) -> Finger:
    """
    Cached finger build, keyed on the design vector as a tuple.
    lru_cache requires hashable arguments — hence tuple, not ndarray.
    """
    return build_finger(np.array(v_tuple))


def _get_finger(v: np.ndarray) -> Finger:
    """Convert to tuple for hashing, then hit the cache."""
    return _cached_build_finger(tuple(v))

def worker_init(R, D, numFlexs, numExts, numTendons,
                numLengthParameters, DESIGN_LENGTHS,
                lengths, overall_length, grasps):
    """Called once per worker process at startup. Populates globals."""
    global _R, _D, _numFlexs, _numExts, _numTendons
    global _numLengthParameters, _DESIGN_LENGTHS, _lengths, _overall_length, _grasps
    _R = R
    _D = D
    _numFlexs = numFlexs
    _numExts = numExts
    _numTendons = numTendons
    _numLengthParameters = numLengthParameters
    _DESIGN_LENGTHS = DESIGN_LENGTHS
    _lengths = lengths
    _overall_length = overall_length
    _grasps = grasps


def build_finger(v) -> Finger:
    """
    Stateless finger construction — no shared call counter, no caching.
    Safe to call from any worker process.
    """
    v = np.asarray(v)
    numJointParameters = _numFlexs * 2 + _numExts * 2

    fs, es = [], []
    for i in range(_numFlexs + _numExts):
        if i < _numFlexs:
            fs.append((0, v[2 * i], v[2 * i + 1]))
        else:
            es.append((v[2 * i + 1], v[2 * i]))

    VSM = VariableStrucMatrix(
        _R, _D, F=[50] * _numTendons,
        ranges=[es[0]] + [fs[0]] * 3
                + [es[1]] + [fs[1]] * 2
                + [es[2]] + [fs[2]],
        types=[VariableStrucMatrix.convergent_circles_extension_joint]
              + [VariableStrucMatrix.ideal_convergent_circles_joint_with_limit] * 3
              + [VariableStrucMatrix.convergent_circles_extension_joint]
              + [VariableStrucMatrix.ideal_convergent_circles_joint_with_limit] * 2
              + [VariableStrucMatrix.convergent_circles_extension_joint]
              + [VariableStrucMatrix.ideal_convergent_circles_joint_with_limit],
        name="VSM_worker"
    )

    if _DESIGN_LENGTHS:
        free_lengths = v[numJointParameters + 1:]  # skip thickness param if present
        distal_length = _overall_length - np.sum(free_lengths)
        v_lens = np.array([free_lengths[0], free_lengths[1], distal_length])
    else:
        v_lens = _lengths

    return Finger(VSM, v_lens, tensionLimit=np.max(VSM.F))


# ─── Per-individual evaluation functions (each takes one vector) ─────────────

def eval_objective(args):
    """Evaluate the thickness objective for one design vector."""
    v, objective_mode, numJointParameters = args
    # Only thickness mode needs parallelizing here; extend as needed
    if objective_mode == "thickness":
        return float(v[numJointParameters])
    else:
        Fing = build_finger(v)
        from scipy import optimize as _opt
        worst = _opt.minimize(
            Fing.structure.get_magnitude,
            x0=[0.0] * Fing.numJoints,
            bounds=[(0, np.pi / 2)] * Fing.numJoints
        )
        return float(-worst.fun)


def eval_tensions(v):
    """Return tension array for all grasps for one design vector."""
    Fing = build_finger(v)
    all_tensions = []
    for g in _grasps:
        grip = Fing.grasp_to_grip(g)
        tensions = Fing.grip_to_tensions(g.q, grip)
        all_tensions.extend(tensions)
    return np.array(all_tensions)


def eval_bend_radius(v):
    """Return worst-case (minimum) bend radius for one design vector."""
    Fing = build_finger(v)
    rs = [fn.r for fn in Fing.structure.effortFunctions]
    return float(np.min(rs))


def eval_c_limit(v):
    """Return c-limit exceedance array for one design vector."""
    Fing = build_finger(v)
    return np.array([
        Fing.lengths[i] / 2
        - Fing.structure.indexed_effort_functions[i, i + 1].c * np.cos(np.pi / 4)
        for i in range(len(Fing.lengths))
    ])


def eval_envelopes(v):
    """Return per-pose hull containment scalars for one design vector."""
    Fing = build_finger(v)
    containment = []
    for name, pose in valeroFinger.structure.poses.items():
        humanCap = valeroFinger.inPlaneCapabilities[name]
        designCap = Fing.get_convex_capability(
            pose[:-1],
            constrained_dof=dof_constraints[Fing.numJoints]['task_dof'],
            space='task'
        )
        assert isinstance(humanCap, ConvexHull) and isinstance(designCap, ConvexHull)
        containment.append(hull_containment_scalar(humanCap, designCap))
    return np.array(containment)