# finger_config.py
# Single source of truth for finger structure, design vector layout,
# and all conversion functions. Import this everywhere.

from dataclasses import dataclass, asdict, fields
from strucMatrices import VariableStrucMatrix, secondaryDev
from finger import Finger
import numpy as np

# ─── Structural constants ─────────────────────────────────────────────────────

R         = secondaryDev.R
D         = secondaryDev.D
NUM_FLEXS = 3
NUM_EXTS  = 3
NUM_JOINTS = 3
NUM_TENDONS = NUM_JOINTS + 1

# LENGTHS_DEFAULT  = [1.866, 1.4, 0.75]
LENGTHS_DEFAULT = [1.4, 1.4, 1.2]
OVERALL_LENGTH   = np.sum(LENGTHS_DEFAULT)
TENSION_LIMIT    = 50.0
THICKNESS_LIMIT = 0.72

# ─── Design vector layout ─────────────────────────────────────────────────────
# This is the ONLY place the layout is defined.
# Everything else derives from it.
#
# Layout:
#   [f0_min, f0_min_inner, f0_max,        # joint 0 flexors  (0,1,2)
#    f1_min, f1_min_inner, f1_max,        # joint 1 flexors  (3,4,5)
#    f2_min, f2_min_inner, f2_max,        # joint 2 flexors  (6,7,8)
#    e0_max, e0_min,                      # joint 0 extensors (9,10)
#    e1_max, e1_min,                      # joint 1 extensors (11,12)
#    e2_max, e2_min,                      # joint 2 extensors (13,14)
#    ——— if DESIGN LENGTHS ———
#    l0, l1                               # and we let distal be computed
#                                         #       based on overall length
#    ——— if RELAX_THICKNESS ———
#    thickness]                           # overall thickness (15)
#

NUM_FLEX_PARAMS = NUM_FLEXS * 3          # f_min, f_min_inner, f_max per joint
NUM_EXT_PARAMS  = NUM_EXTS  * 2          # e_max, e_min per joint
NUM_JOINT_PARAMS = NUM_FLEX_PARAMS + NUM_EXT_PARAMS   # 15
NUM_LEN_PARAMS = 2

def n_elements(design_lengths=False) -> int:
    n = NUM_JOINT_PARAMS
    if design_lengths:  n += NUM_LEN_PARAMS
    return n

def bounds(design_lengths=False) -> list[tuple]:
    b = (
        [(0.0,   0.5)] * NUM_FLEX_PARAMS   # flexor params (f_min can be 0)
        + [(0.125, 0.5)] * NUM_EXT_PARAMS  # extensor params
        + [(0.875, 2.0)] * (2 if design_lengths else 0)
    )
    return b

# ─── JointParams dataclass ────────────────────────────────────────────────────

@dataclass
class JointParams:
    joint_idx:   int
    e_max:       float = 0.25
    e_min:       float = 0.125
    f_max:       float = 0.42
    f_min:       float = 0.0
    f_min_inner: float = 0.20
    length:      float = 1.4

    def to_vector_slice(self) -> dict:
        """Returns the values this joint contributes to the design vector."""
        return {
            'flex': (self.f_min, self.f_min_inner, self.f_max),
            'ext':  (self.e_max, self.e_min),
            'length': self.length,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'JointParams':
        return cls(**d)


def load_joint_params(path: str) -> list[JointParams]:
    import json
    with open(path) as f:
        return [JointParams.from_dict(d) for d in json.load(f)]


def compute_rich_thickness(jps: list[JointParams], finger: Finger):
    # ── Compute overall thickness ─────────────────────────────────────────

    thicknesses = [
        jp.f_max + jp.e_min
        for jp in jps
    ]
    thicknesses += [
        jp.e_max + jp.f_min + np.sqrt(2)/2*
(finger.structure.indexed_effort_functions[jp.joint_idx, jp.joint_idx+1].c -
finger.structure.indexed_effort_functions[jp.joint_idx, jp.joint_idx+1].r)
        for jp in jps
    ]
    # print(len(thicknesses))
    overall_thicknesses = np.vstack((np.array(thicknesses[:len(thicknesses)//2]),
                                   np.array(thicknesses[len(thicknesses)//2:])))
    overall_thicknesses = np.max(overall_thicknesses, axis=0)
    overall_thickness = max(overall_thicknesses)

    return overall_thickness, overall_thicknesses


# ─── Conversion functions ─────────────────────────────────────────────────────

def joint_params_to_vector(jps: list[JointParams],
                            design_lengths: bool = False) -> np.ndarray:
    """
    JointParams list → flat design vector.
    This is the canonical conversion — use this everywhere.
    """
    v = np.empty(n_elements(design_lengths))

    for jp in jps:
        i = jp.joint_idx
        v[i * 3]     = jp.f_min
        v[i * 3 + 1] = jp.f_min_inner
        v[i * 3 + 2] = jp.f_max
        v[NUM_FLEX_PARAMS + i * 2]     = jp.e_max
        v[NUM_FLEX_PARAMS + i * 2 + 1] = jp.e_min
        if design_lengths and i<NUM_LEN_PARAMS:
            v[NUM_JOINT_PARAMS + i] = jp.length

    return v

def vector_to_joint_params(v: np.ndarray,
                            design_lengths: bool = False,
                            lengths_default: list = None) -> list[JointParams]:
    """
    Flat design vector → JointParams list.
    Inverse of joint_params_to_vector.
    """
    if lengths_default is None:
        lengths_default = LENGTHS_DEFAULT

    jps = []
    for i in range(NUM_JOINTS):
        # length = (v[NUM_JOINT_PARAMS + i]
        #           if design_lengths
        #           else lengths_default[i])
        jps.append(JointParams(
            joint_idx   = i,
            f_min       = float(v[i * 3]),
            f_min_inner = float(v[i * 3 + 1]),
            f_max       = float(v[i * 3 + 2]),
            e_max       = float(v[NUM_FLEX_PARAMS + i * 2]),
            e_min       = float(v[NUM_FLEX_PARAMS + i * 2 + 1]),
        ))
        if design_lengths:
            jps[-1].length = float(v[NUM_JOINT_PARAMS + i]) \
                          if i < NUM_LEN_PARAMS \
                          else float(OVERALL_LENGTH-np.sum(v[-NUM_LEN_PARAMS:]))
        else:
            jps[-1].length = float(lengths_default[i]) \
             if i < NUM_LEN_PARAMS \
             else float(OVERALL_LENGTH-np.sum(lengths_default[:NUM_LEN_PARAMS]))
    return jps

# ─── Finger builder ───────────────────────────────────────────────────────────

def build_finger_from_params(jps: list[JointParams],
                              tension_limit: float = TENSION_LIMIT) -> Finger:
    """
    JointParams → Finger. Single canonical implementation.
    Used by GUI, batch optimizer, and parallel evaluator.
    """
    lengths = [jp.length for jp in jps]
    fs = [(jp.f_min, jp.f_max, jp.f_min_inner) for jp in jps]
    es = [(jp.e_min, jp.e_max) for jp in jps]

    ranges = [
        val
        for i in range(NUM_JOINTS)
        for val in [es[i]] + [fs[i]] * (NUM_JOINTS - i)
    ]
    types = [
        VariableStrucMatrix.convergent_circles_extension_joint if x == -1
        else VariableStrucMatrix.ideal_convergent_circles_joint_with_limit
        for x in D[np.where(D != 0)]
    ]

    VSM = VariableStrucMatrix(
        R, D, F=[50] * NUM_TENDONS,
        ranges=ranges,
        types=types,
        name="Finger"
    )
    return Finger(VSM, lengths[:NUM_JOINTS], tensionLimit=tension_limit)


def build_finger_from_vector(v: np.ndarray,
                              design_lengths: bool = False,
                              tension_limit: float = TENSION_LIMIT) -> Finger:
    """
    Flat vector → Finger. Thin wrapper used by parallel evaluator.
    """
    jps = vector_to_joint_params(v, design_lengths=design_lengths)
    return build_finger_from_params(jps, tension_limit=tension_limit)

if __name__ == "__main__":
    jps = load_joint_params('finger_designs/PFWORKS_BRO_design_3dof_pf_20261119-1704.json')
    finger = build_finger_from_params(jps)
    print(compute_rich_thickness(jps, finger))
    # print()