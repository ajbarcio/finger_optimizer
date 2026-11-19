from finger import valero, valeroFinger, dof_constraints, Finger
from strucMatrices import StrucMatrix, diagonal, D_ICS
from utils import hull_containment_scalar, colors
from BRANCH_human_finger_dev.ValeroCuervasModel import pose_dict

from variableOptimizerGlobal import FingerEvaluator, stallException

import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from matplotlib import pyplot as plt

import json

# This is meant to be a tiny lightweight module. Gonna import constraints and
# parameters as globals

## ======= Valero Finger Information =======

qf = pose_dict['Flexion']
qi = pose_dict['Intermediate']
qe = pose_dict['Extension']

poses = [qf, qi, qe]
pose_names = ['Flexion', 'Intermediate', 'Extension']

capability_f = valeroFinger.get_convex_capability(qf, # pose
                                                  valero.Fo(qf), # tendon efforts
                                                  dof_constraints[4]['task_dof'], # in-plane constraints
                                                  space='task')
capability_i = valeroFinger.get_convex_capability(qi,
                                                  valero.Fo(qi),
                                                  dof_constraints[4]['task_dof'],
                                                  space='task')
capability_e = valeroFinger.get_convex_capability(qe,
                                                  valero.Fo(qe),
                                                  dof_constraints[4]['task_dof'],
                                                  space='task')

VC_capabilities = [capability_f, capability_i, capability_e]

VC_constraints = {
                  name: {'pose': pose,
                         'capability': capability} for name, pose, capability
                                     in zip(pose_names, poses, VC_capabilities)}

## ======= 'Designed' Finger Parameters =======

D = D_ICS
effort_tjps = np.array(list(zip(np.array(np.where(D!=0))[0,:],np.array(np.where(D!=0))[1,:])))
# print(effort_tjps)
lengths = [1.4, 1.4, 1.2]

class StaticFingerEvaluator(FingerEvaluator):

    def __init__(self, pose_name):
        super().__init__()
        self.pose_name = pose_name
        self.pose = VC_constraints[pose_name]['pose']
        self.ref_capability = VC_constraints[pose_name]['capability']

    def build_finger(self, v):
        v = np.asarray(v)
        if not hasattr(self, "bf_called"):
            self.bf_called = 1
        else:
            self.bf_called+=1
        R = np.zeros_like(D, dtype=float)
        # print(R.shape)
        assert len(v) == len(effort_tjps)
        for i, tjp in enumerate(effort_tjps):
            # print(v[i], tjp)
            R[*tjp] = v[i]
            # print(v[i], tjp, R[*tjp])
        # print(R, D)
        S = StrucMatrix(R, D, F=[50]*D.shape[-1],
                        name="staticFinger_{self.bf_called}_{self.pose_name}")
        Fing = Finger(S, lengths)
        return Fing

    def _get_finger(self, v):
        v = np.asarray(v)
        if self._v_save is None or not np.array_equal(v, self._v_save):
            self._v_save = v.copy()
            self._finger = self.build_finger(v)
        return self._finger

    def objective(self, v):
        """
        For this silly static finger case, the objective is smallest capability
        volume
        """

        self._eval_count+=1
        Fing = self._get_finger(v)
        capability_hull = Fing.get_convex_capability(self.pose[1:],
                                                     constrained_dof=[],
                                                     output_dim=2,
                                                     space='task')
        return capability_hull.volume

    # Override for scalar version of envelopes human
    def envelopes_human(self, v):
        Fing = self._get_finger(v)
        design_capability = Fing.get_convex_capability(self.pose[1:],
                                                     constrained_dof=[],
                                                     output_dim=2,
                                                     space='task')
        assert ((isinstance(self.ref_capability, ConvexHull)) and
                (isinstance(design_capability, ConvexHull)))
        return hull_containment_scalar(self.ref_capability, design_capability)

if __name__ == "__main__":
    results = {}
    replace = True

    if not replace:
        for VC_Pose in VC_constraints.keys():
            name = VC_Pose
            evaluator = StaticFingerEvaluator(name)
            print(f"Trying {name} this loop")

            include_constraint = NonlinearConstraint(
                fun=evaluator.envelopes_human,
                lb=-np.inf,
                ub=0.0
            )

            v0 = [0.4] * len(effort_tjps)

            try:
                result = minimize(
                    evaluator.objective,
                    v0,
                    bounds=[(.125, 0.5)] * len(effort_tjps),
                    constraints=[include_constraint],
                    options={
                        "verbose": 1,
                        "gtol": 1e-4,
                    },
                    callback=evaluator.callback,
                    method="trust-constr"
                )
            except KeyboardInterrupt:
                print("Optimization terminated by user, returning intermediate result")
                result = evaluator.curr_res
            except stallException:
                print("Optimization terminated by step too small")
                result = evaluator.curr_res
            except Exception as e:
                import traceback
                traceback.print_exc()
                result = evaluator.curr_res

            results[name] = result

        # Save: just the x vectors, one file per pose
        for name, result in results.items():
            np.savetxt(f"result_{name}_local.smx", result.x)
        print("Results saved.")

    else:
        # Load: reconstruct result-like objects from saved x vectors
        for name in VC_constraints.keys():
            x = np.loadtxt(f"result_{name}_local.smx")
            results[name] = OptimizeResult(x=x)
        print("Results loaded.")

    print(f"\n\n\n\n\n")

    # ── Plotting — identical whether we ran or loaded ─────────────────────────
    for name, result in results.items():
        plt.figure()
        ax = plt.gca()

        print(result.x)

        # Reconstruct the evaluator purely for build_finger — no optimization state needed
        evaluator = StaticFingerEvaluator(name)
        result_finger = evaluator.build_finger(result.x)

        result_capability = result_finger.get_convex_capability(
            pose_dict[name][1:],
            constrained_dof=[],
            output_dim=2,
            space='task'
        )
        vc_capability = VC_constraints[name]['capability']

        compare_ax = result_finger.plot_Capability(result_capability, ax=ax, color=colors[-1])
        still_compare_ax = valeroFinger.plot_Capability(vc_capability, ax=compare_ax, color=colors[0])
        still_compare_ax.set_title(f"local optimizer for {name} pose")

        print(f"For VC Pose {name}")
        print(f"The minimal moment arms required at a pose are")
        print(f"{result_finger.structure.S}")
        print(f"With an OTV of {result_finger.structure.get_magnitude()}")


    plt.show()