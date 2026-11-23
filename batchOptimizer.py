from interactive_finger_designer import JointParams, load_joint_params
from interactive_finger_designer import _compute_rich_thickness
from interactive_finger_designer import build_finger
import numpy as np

# this is inefficient and mayyyybe is allowed to be innefficient because we're 
# only using it to generate intiial guesses

def vector_from_joint_params(jps: list[JointParams]) -> np.ndarray:
    design_vector = np.empty(len(jps)*5+(len(jps) if _DESIGN_LENGTHS else 0)+1, dtype=float)
    for jp in jps:
        design_vector[jp.joint_idx*3]   = jp.f_min
        design_vector[jp.joint_idx*3+1] = jp.f_min_inner
        design_vector[jp.joint_idx*3+2] = jp.f_max
        design_vector[_num_flexs*3+2*jp.joint_idx]   = jp.e_max
        design_vector[_num_flexs*3+2*jp.joint_idx+1] = jp.e_min
        if _DESIGN_LENGTHS:
            design_vector(_num_exts*2+_num_flexs*3+jp.joint_idx) = jp.length
    design_vector[-1], _ = _compute_rich_thickness(jps, build_finger(jps))