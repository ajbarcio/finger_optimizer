from strucMatrices import VariableStrucMatrix, secondaryDev
from utils import *
from finger import Finger, StructureKineMismatch, Grasp
from scipy import optimize
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult
import statistics
import time

from multiprocessing import Pool

def z_score_from_percentile(percentile):
    """
    Takes a percentile (0 to 100) and returns the corresponding Z-score.
    """
    probability = percentile / 100.0
    return statistics.NormalDist().inv_cdf(probability)

class stallException(Exception):
    pass

def createFingerFromVector(v) -> Finger:
    v = np.asarray(v)

    if not hasattr(createFingerFromVector, "called"):
        createFingerFromVector.called = 1
    else:
        createFingerFromVector.called += 1
    fs = []
    es = []
    for i in range(numFlexs + numExts):
        if i < numFlexs:
            fs.append((0, v[2 * i], v[2 * i + 1]))
        else:
            es.append((v[2 * i + 1], v[2 * i]))
    ranges = [es[0]] + [fs[0]] * 3 + [es[1]] + [fs[1]] * 2 + [es[2]] + [fs[2]]

    VSM = VariableStrucMatrix(
        R, D, F=[50] * numTendons,
        ranges=[es[0]] + [fs[0]] * 3
               + [es[1]] + [fs[1]] * 2
               + [es[2]] + [fs[2]],
        types=[VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit] * 3
              + [VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit] * 2
              + [VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit],
        name=f"VSM{createFingerFromVector.called}"
    )

    if DESIGN_LENGTHS:
        free_lengths = v[-numLengthParameters:]
        distal_length = overall_length - np.sum(free_lengths)
        v_lens = np.array([free_lengths[0], free_lengths[1], distal_length])
    else:
        v_lens = lengths

    # v_lens = v[-numLengthParameters:] if DESIGN_LENGTHS else lengths
    Fing = Finger(VSM, v_lens, tensionLimit=np.max(VSM.F))
    return Fing

class FingerEvaluator:
    """
    Evaluator class for finger optimization.

    Supports multiple objective modes:
        - "worst_case_OVL": minimize (negate of) the worst-case output-velocity-leverage
        - "worst_case_tension": minimize the worst-case tendon tension across all grasps
    """

    OBJECTIVE_WORST_CASE_OVL = "worst_case_OVL"
    OBJECTIVE_WORST_CASE_TENSION = "worst_case_tension"
    OBJECTIVE_THICKNESS = "thickness"

    def __init__(self, grasps: list[Grasp] = None, objective_mode: str = "worst_case_OVL"):
        self._v_save = None
        self._v_prev = None
        self.curr_res = None
        self._finger = None
        self._cb_last_lines = 0
        self._threshold_times = 0
        self.optimalities = []
        self.worst_case_tensions = []
        self._curr_wct = None
        self.worst_pose_for_strength = None
        self.best_objective = np.inf
        self.best_x = None
        self.objective_history = []
        self._eval_count = 0
        self._start_time = None

        self.grasps = grasps
        self.objective_mode = objective_mode
        self._stall_count = 0
        self._prev_best_x = None

    def _get_finger(self, v):
        v = np.asarray(v)
        if self._v_save is None or not np.array_equal(v, self._v_save):
            self._v_save = v.copy()
            self._finger = createFingerFromVector(v)
        return self._finger

    def objective(self, v):
        """
        Unified objective function. Dispatches based on self.objective_mode.
        """
        self._eval_count += 1
        if self.objective_mode == self.OBJECTIVE_WORST_CASE_OVL:
            return self.worst_case_OVL(v)
        elif self.objective_mode == self.OBJECTIVE_WORST_CASE_TENSION:
            return self.worst_case_tension(v)
        elif self.objective_mode == self.OBJECTIVE_THICKNESS:
            return v[numJointParameters]
        else:
            raise ValueError(f"Unknown objective mode: {self.objective_mode}")

    def global_callback(self, xk, convergence=0.0):
        """Callback for differential_evolution and similar global methods."""
        if self._start_time is None:
            self._start_time = time.time()

        elapsed = time.time() - self._start_time
        obj_val = self.objective(xk)
        self.objective_history.append(obj_val)
        self.worst_case_tensions.append(self._curr_wct)

        if obj_val < self.best_objective:
            self.best_objective = obj_val
            self.best_x = xk.copy()

        # Clear previous output
        print("\r" + "\033[F" * self._cb_last_lines, end="")

        lines = [
            f"─── Global Optimization Progress ───",
            f"  Elapsed time:    {elapsed:.1f}s",
            f"  Generations:     {self._eval_count}",
            f"  Convergence:     {convergence:.6f}",
            f"  Current obj:     {obj_val:.6f}",
            f"  Best obj:        {self.best_objective:.6f}",
            f"  Worst tension:   {self._curr_wct}",
            f"  Current best x:  {np.array2string(xk, precision=4, suppress_small=True)}",
            f"────────────────────────────────────",
        ]

        print("\r\033[2K" + "\n".join(lines), end="", flush=True)
        self._cb_last_lines = len(lines)

        if elapsed > 60*5:
            return True

    def callback(self, intermediate_result: OptimizeResult):
        """Callback for local polish (trust-constr)."""
        self.curr_res = intermediate_result
        print("\r" + "\033[F" * self._cb_last_lines, end="")

        self.optimalities.append(intermediate_result.optimality)
        self.worst_case_tensions.append(self._curr_wct)
        lines = [
            f"── Local Polish ──",
            f"  iter: {intermediate_result.niter}",
            f"  optimality: {intermediate_result.optimality:.3e}",
            f"  worst case tension: {self._curr_wct}",
            f"  x: " + np.array2string(
                intermediate_result.x,
                precision=4,
                suppress_small=True
            ) + "                                                                  ",
        ]
        if self._v_prev is not None:
            step = np.linalg.norm(intermediate_result.x - self._v_prev, ord=np.inf)
            if step < 1e-3:
                self._threshold_times += 1
                lines.append(f"  step lower than 1 thou {self._threshold_times} times in a row")
            else:
                self._threshold_times = 0
        if self._threshold_times > 10:
            raise stallException(
                f"Optimizer stalled out at low step value after {self._threshold_times} in a row with a change less than 1 thou"
            )
        print("\r\033[2K" + "\n".join(lines), end="", flush=True)
        self._cb_last_lines = len(lines)
        self._v_prev = intermediate_result.x

    def condition(self, v):
        Fing = self._get_finger(v)
        res = optimize.minimize_scalar(
            lambda theta: -Fing.structure.controllability([theta] * Fing.numJoints),
            bracket=(0, np.pi / 2), bounds=(0, np.pi / 2)
        )
        return abs(res.fun)

    def magnitude_scale(self, v):
        Fing = self._get_finger(v)
        return Fing.structure.get_magnitude([0] * Fing.numJoints) / Fing.structure.get_magnitude(
            [np.pi / 2] * Fing.numJoints)

    def ultimate_magnitude(self, v):
        Fing = self._get_finger(v)
        return -Fing.structure.get_magnitude([np.pi / 2] * Fing.numJoints)

    def worst_case_OVL(self, v):
        Fing = self._get_finger(v)
        worst_strength = optimize.minimize(
            Fing.structure.get_magnitude,
            x0=[0] * Fing.numJoints,
            bounds=[(0, np.pi / 2)] * Fing.numJoints
        )
        self.worst_pose_for_strength = worst_strength.x
        assert np.isfinite(worst_strength.fun)
        return -worst_strength.fun

    def strength_increase(self, v):
        Fing = self._get_finger(v)
        return (np.linalg.norm(
            Fing.grip_to_tensions([np.pi / 2] * Fing.numJoints,
                                  Fing.grasp_to_grip(Grasp([F] * Fing.numJoints, [np.pi / 2] * Fing.numJoints, frame="EE"))))
                /
                np.linalg.norm(
                    Fing.grip_to_tensions([0] * Fing.numJoints,
                                          Fing.grasp_to_grip(Grasp([F] * Fing.numJoints, [0] * Fing.numJoints, frame="EE"))))
                )

    def grasp_worst_case_tension(self, Fing: Finger, grasp: Grasp):
        grip = Fing.grasp_to_grip(grasp)
        tension = Fing.grip_to_tensions(grasp.q, grip)
        return np.max(tension)

    def vector_grasps_constraint(self, v):
        Fing = self._get_finger(v)
        all_tensions = []
        for g in self.grasps:
            grip = Fing.grasp_to_grip(g)
            tensions = Fing.grip_to_tensions(g.q, grip)
            all_tensions.extend(tensions)
        all_tensions = np.array(all_tensions)
        self._curr_wct = np.max(all_tensions)
        return all_tensions

    def all_grasps_constraint(self, v):
        Fing = self._get_finger(v)
        tensions = np.array([self.grasp_worst_case_tension(Fing, g) for g in self.grasps])
        self._curr_wct = np.max(tensions)
        return tensions

    def continuous_power_constraint(self, Fing):
        res1 = optimize.minimize_scalar(
            lambda theta: -np.max(Fing.grip_to_tensions(
                [theta] * Fing.numJoints,
                Fing.grasp_to_grip(Grasp([F] * Fing.numJoints, [theta] * Fing.numJoints, frame="EE")))),
            bracket=(0, np.pi / 2), bounds=(0, np.pi / 2)
        )
        res2 = optimize.minimize_scalar(
            lambda theta: -np.max(Fing.grip_to_tensions(
                [theta] * Fing.numJoints,
                Fing.tip_wrench_at_pose_to_grip([theta] * Fing.numJoints, -F * 0.1, frame="EE"))),
            bracket=(0, np.pi / 2), bounds=(0, np.pi / 2)
        )
        objectiveRet = np.max([-res1.fun, -res2.fun])
        return objectiveRet

    def worst_case_tension(self, v):
        tensions = self.all_grasps_constraint(v)
        objectiveRet = np.max(tensions)
        self._curr_wct = objectiveRet
        return objectiveRet

def find_feasible_point(evaluator, bounds, constraints_objects, v0):
    """
    Solve a feasibility problem: minimize constraint violation.
    """
    def feasibility_objective(v):
        # We want all tensions <= TENSION_LIMIT
        tensions = evaluator.all_grasps_constraint(v)
        violation = np.maximum(tensions - TENSION_LIMIT, 0)
        return np.sum(violation**2)

    # Try from multiple starting points
    best_v = None
    best_viol = np.inf

    starts = [
        v0,
        [0.3, 0.2] * numFlexs + [0.3, 0.25] * numExts,
        [0.35, 0.25] * numFlexs + [0.25, 0.15] * numExts,
        [0.4, 0.3] * numFlexs + [0.2, 0.15] * numExts,
    ]

    for x0 in starts:
        try:
            print(f"for starting point {x0}")
            res = optimize.minimize(
                feasibility_objective,
                x0,
                bounds=bounds,
                constraints=[c for c in constraints_objects if isinstance(c, LinearConstraint)],
                method="trust-constr",
                options={"maxiter": 50, "gtol": 1e-6}
            )
            print("got through the feasibility optimization")
            if res.fun < best_viol:
                best_viol = res.fun
                best_v = res.x
                if best_viol < 1e-10:
                    break  # found strictly feasible
        except Exception as e:
            print(f"  Start failed: {e}")
            continue

    if best_viol < 1e-10:
        print(f"  Found feasible point! Max tension violation: {best_viol:.2e}")
    else:
        print(f"found a decent starting point at {best_v}")
        print(f"  Best violation: {best_viol:.4f} — may not be fully feasible")

    return best_v, best_viol

def find_feasible_point_simple(evaluator, bounds, v0, verbose=True):
    """
    Quick and dirty: Nelder-Mead with penalty. No Jacobians, no constraint handling overhead.
    """
    call_count = [0]

    def penalized_feasibility(v):
        call_count[0] += 1
        penalty = 0.0

        # Bound violations
        v_clipped = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
        penalty += 1000 * np.sum((v - v_clipped)**2)

        # max >= min
        for i in range(numJointParameters // 2):
            if v[2*i] < v[2*i+1]:
                penalty += 1000 * (v[2*i+1] - v[2*i])**2

        # Thickness constraints
        for i in range(numFlexs):
            thick = v[2*i] + v[2*numFlexs + 2*i + 1]
            if thick > 0.65:
                penalty += 1000 * (thick - 0.65)**2

        sec_limit = 0.65 - 2.25/25.4 - 0.125
        for i in range(numExts * 2):
            val = v[numFlexs*2 + i]
            if val > sec_limit:
                penalty += 1000 * (val - sec_limit)**2

        # Tension constraint — the expensive one
        tensions = evaluator.all_grasps_constraint(v)
        max_t = np.max(tensions)
        if max_t > TENSION_LIMIT:
            penalty += 100 * (max_t - TENSION_LIMIT)**2

        if verbose and call_count[0] % 10 == 0:
            print(f"  eval #{call_count[0]}: max_tension={max_t:.2f}, penalty={penalty:.2f}")

        return penalty

    print("  Running Nelder-Mead feasibility search...")
    res = optimize.minimize(
        penalized_feasibility,
        v0,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4, "adaptive": True}
    )
    print(f"  Done after {call_count[0]} evals, final penalty={res.fun:.6f}")

    # Clip to bounds
    v_result = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])

    # Verify
    tensions = evaluator.all_grasps_constraint(v_result)
    print(f"  Max tension at result: {np.max(tensions):.2f} (limit: {TENSION_LIMIT})")

    return v_result, res.fun

# # Use it before the main optimization:
# print("Searching for feasible starting point...")
# v_feasible, violation = find_feasible_point(evaluator, bounds, constraints_objects, v0)

# # Seed population
# n_pop = 20 * numElements
# init_pop = np.random.uniform(0.125, 0.5, size=(n_pop, numElements))
# init_pop[0] = v_feasible

# # Also generate more feasible-ish points by perturbing the feasible one
# for i in range(1, min(20, n_pop)):
#     perturbation = np.random.normal(0, 0.02, size=numElements)
#     candidate = np.clip(v_feasible + perturbation, 0.125, 0.5)
#     init_pop[i] = candidate

# ─── Grasp definitions ───────────────────────────────────────────────────────

q_ext = [10 * np.pi / 180] * 3
q_int = [45 * np.pi / 180] + [10 * np.pi / 180] * 2
q_flx = [45 * np.pi / 180] * 2 + [10 * np.pi / 180]

z = z_score_from_percentile(80)
sig = [10.86, 2.20, 12.5, 6.13, 20.2, 11.05, 10.47, 2.04, 12.68]
means = [26.3, 6.6, 25.3, 25.25, 7.26, 26.3, 31.06, 7.56, 27.08]
vals = [means[i] + z * sig[i] for i in range(len(sig))]

VC_Dir_Grasps = [
    # Extended pose:
    Grasp([[0]*3,[0]*3,[0,vals[0]/4.448,0]], q_ext),
    Grasp([[0]*3,[0]*3,[0,-vals[1]/4.448,0]], q_ext),
    # Grasp([[0]*3,[0]*3,[vals[2]/4.448,0,0]], q_ext),
    # Intermediate pose:
    Grasp([[0]*3,[0]*3,[0,vals[3]/4.448,0]], q_int),
    Grasp([[0]*3,[0]*3,[0,-vals[4]/4.448,0]], q_int),
    # Grasp([[0]*3,[0]*3,[vals[5]/4.448,0,0]], q_int),
    # Intermediate pose:
    Grasp([[0]*3,[0]*3,[0,vals[6]/4.448,0]], q_flx),
    Grasp([[0]*3,[0]*3,[0,-vals[7]/4.448,0]], q_flx),
    # Grasp([[0]*3,[0]*3,[vals[8]/4.448,0,0]], q_flx),
]

# ─── Configuration ───────────────────────────────────────────────────────────

# Choose your objective here:
#   FingerEvaluator.OBJECTIVE_WORST_CASE_OVL
#   FingerEvaluator.OBJECTIVE_WORST_CASE_TENSION
OBJECTIVE_MODE = FingerEvaluator.OBJECTIVE_WORST_CASE_TENSION

# Global optimizer settings
GLOBAL_METHOD = "differential_evolution"  # options: "differential_evolution", "dual_annealing", "shgo"
POLISH = False          # Run a local polish after global search?
WORKERS = -1           # -1 = all cores (only for differential_evolution)
MAX_ITER = 200        # Max generations/iterations for global search
TOL = 1e-3             # Tolerance for convergence
SEED = 42              # Random seed for reproducibility (None for random)
TENSION_LIMIT = 55.0   # Max allowable tension constraint
STALL_STEP_THRESHOLD = 0.001  # 1 thou — max change in any design variable
STALL_LIMIT = 50              # Stop after this many generations without movement
RELAX_THICKNESS = True       # Set true to allow the finger to get thicker to acheive tension requirements
DESIGN_LENGTHS = False        # Set true to allow the finger links to change lengths

numJoints = 3
numTendons = numJoints + 1
numFlexs = 3
numExts = 3
numJointParameters = numFlexs * 2 + numExts * 2
numConstraintParameters = 1 # Overall Thickness
numLengthParameters = 2 # Link Lengths

overallNumElements = numJointParameters
if RELAX_THICKNESS:
    overallNumElements += numConstraintParameters
if DESIGN_LENGTHS:
    overallNumElements += numLengthParameters

F = np.array([0, 5, 0])
lengths = [1.6, 1.3, 1]
overall_length = 4
initial_overall_thickness = 0.65
growth_factor = 1.2
R = secondaryDev.R
D = secondaryDev.D

def linear_constraints():
    # ─── Bounds ──────────────────────────────────────────────────────────────
    bounds = [(.125, .5)] * numJointParameters +  \
             [(initial_overall_thickness, initial_overall_thickness*growth_factor)]*(numConstraintParameters if RELAX_THICKNESS else 0) + \
             [(0.875, 2)]*(numLengthParameters if DESIGN_LENGTHS else 0)

    # ─── Linear Joint Parameter Constraints ──────────────────────────────────────────────────
    # Each max >= its associated min: (-max + min) > 0
    max_min_jacobian = np.array([
        np.roll(row, shift) for row, shift in
        zip(-np.eye(numJointParameters // 2, numJointParameters, k=1) +
            np.eye(numJointParameters // 2, numJointParameters, k=0),
            np.arange(numJointParameters))
    ])
    # add extra length to support a longer design vector if needed
    if DESIGN_LENGTHS or RELAX_THICKNESS:
        max_min_jacobian = np.hstack([max_min_jacobian, np.zeros((max_min_jacobian.shape[0], overallNumElements-max_min_jacobian.shape[1]))])

    # print(max_min_jacobian)
    # Each pair of max flexion and min extension <= established thickness
    # (This section adds max flexion and associated min extension together)
    overall_thickness_jacobian = np.zeros([numFlexs, numJointParameters])
    for i in range(numFlexs):
        overall_thickness_jacobian[i, 2 * i] = 1
        overall_thickness_jacobian[i, 2 * numFlexs + 2 * i + 1] = 1

    # Max extension constrained by overall finger thickness
    second_thickness_jacobian = np.hstack([
        np.zeros([numFlexs * 2, numFlexs * 2]),
        np.eye(numExts * 2)
    ])

    # IN THE CASE that established thickness is a design variable now:
    if RELAX_THICKNESS:
        # Each existing row of constraints for overall thickness
        overall_thickness_jacobian = np.hstack([overall_thickness_jacobian,
                                    # Gets a corresponding negative 1 associated
                                    # with the thickness design variable
                                   -np.ones((overall_thickness_jacobian.shape[0],
                                            numConstraintParameters))])
        # SO in this mode, Ax < 0 satisfies the constraint (established overall
        # thickness limit greater than the elements that make it up)

        # Update the other thickness constraint in the same way
        second_thickness_jacobian = np.hstack([second_thickness_jacobian,
                                    # Gets a corresponding negative 1 associated
                                    # with the thickness design variable
                                   -np.ones((second_thickness_jacobian.shape[0],
                                            numConstraintParameters))])
    if DESIGN_LENGTHS:
        # Also add extra columns of 0 for the lengths on the thickness jacobians,
        # since they don't matter here:
        overall_thickness_jacobian = np.hstack([overall_thickness_jacobian,
                                   np.zeros((overall_thickness_jacobian.shape[0],
                                            numLengthParameters))])
        second_thickness_jacobian = np.hstack([second_thickness_jacobian,
                                   np.zeros((second_thickness_jacobian.shape[0],
                                            numLengthParameters))])
        # And make a constraint array for the overall length
        # overall_length_jacobian = np.array([0]*(overallNumElements-numLengthParameters)+[1]*(numLengthParameters))
        overall_length_jacobian = np.array([0]*(overallNumElements-numLengthParameters)+[1]*numLengthParameters)

    constraints_objects = [
        LinearConstraint(
            A=max_min_jacobian,
            lb=0,
            ub=np.inf,
            keep_feasible=True,
        ),
        LinearConstraint(
            A = overall_thickness_jacobian,
            lb = -np.inf if RELAX_THICKNESS else 0,
            ub =  0      if RELAX_THICKNESS else initial_overall_thickness,
            keep_feasible=True,
        ),
        LinearConstraint(
            A = second_thickness_jacobian,
            lb = -np.inf if RELAX_THICKNESS else 0,
            ub = (0      if RELAX_THICKNESS else initial_overall_thickness) - 2.25 / 25.4 - .125,
            keep_feasible=True
        ),
    ]
    if DESIGN_LENGTHS:
        constraints_objects.append(
            LinearConstraint(
                A=overall_length_jacobian,
                lb = 0,
                ub = overall_length-0.3,
                keep_feasible=True
            )
        )

    # Catch any issues in the surrounding code controlling the length of the
    # design vector
    for i, constraint in enumerate(constraints_objects):
        # assert constraint.A.shape[1] == overallNumElements
        if constraint.A.shape[1] != overallNumElements:
            print(constraint.A, overallNumElements, i)
            raise AssertionError("Oops ya length wrong")

    return bounds, constraints_objects

if __name__ == "__main__":

    replace = False

    if not replace:
        grasps = VC_Dir_Grasps
        # for grasp in grasps:
        #     print(grasp.F)

        # Instantiate evaluator object to keep track of different performance
        # metrics as the optimizer runs
        evaluator = FingerEvaluator(grasps, objective_mode=OBJECTIVE_MODE)

        # Get all the linear constraints together in a subroutine so this
        # function doesn't get too long
        bounds, constraints_objects = linear_constraints()

        # Nonlinear tension constraint
        tension_limit = NonlinearConstraint(
            fun=evaluator.vector_grasps_constraint,
            lb=0,
            ub=TENSION_LIMIT,
        )

        # Add to constraints
        constraints_objects.append(tension_limit)

        # Try to find feasible point
        # v0 = find_feasible_point(evaluator, bounds, constraints_objects, v0)

        # ─── Initial guess (used for polish seeding if desired) ──────────────────
        # v0 = [0.2818, 0.206, 0.2793, 0.1488, 0.3852, 0.2605,
        #       0.3769, 0.3645, 0.3494, 0.3494, 0.3958, 0.2114]
        # v0 = [0.3203, 0.2489, 0.2739, 0.3354, 0.4623, 0.327,
        #       0.4937, 0.4949, 0.4321, 0.3478, 0.2445, 0.2896]
        # try:
        #     v0 = np.loadtxt("global_attempt.smx")
        # except:
        v0 = [0.2355, 0.1996, 0.3609, 0.3273, 0.4339, 0.4081,
              0.4271, 0.3945, 0.4171, 0.2595, 0.282,  0.1704]+ \
             ([initial_overall_thickness+.05] if RELAX_THICKNESS else [1]*0) + \
             (lengths[:numLengthParameters] if DESIGN_LENGTHS else [1]*0)
        assert len(v0)==overallNumElements
        # Try to make it feasible
        # v0, viol = find_feasible_point(evaluator, bounds, constraints_objects, v0)
        # if viol >0.1:
        #     print("Not super feasible but we're going with it")
        # print("Got past feasibility")

        # ─── Run Global Optimization ─────────────────────────────────────────────
        print(f"\n{'═' * 60}")
        print(f"  Running global optimization")
        print(f"  Method:    {GLOBAL_METHOD}")
        print(f"  Objective: {OBJECTIVE_MODE}")
        print(f"  Workers:   {WORKERS}")
        print(f"  Polish:    {POLISH}")
        print(f"  Max iter:  {MAX_ITER}")
        print(f"  Seed:      {SEED}")
        print(f"{'═' * 60}\n")

        try:
            if GLOBAL_METHOD == "differential_evolution":
                # init_pop = np.random.uniform(0.125, 0.5, size=(20, len(v0)))
                # init_pop[0] = v0
                # print(v0)
                result = optimize.differential_evolution(
                    evaluator.objective,
                    bounds=bounds,
                    constraints=constraints_objects,
                    callback=evaluator.global_callback,
                    workers=WORKERS,
                    maxiter=MAX_ITER,
                    tol=TOL,
                    seed=SEED,
                    polish=POLISH,
                    init='sobol',       # Better space-filling than 'latinhypercube'
                    # init = init_pop,
                    x0=v0,
                    mutation=(0.5, 1.5),
                    recombination=0.7,
                    popsize=20,         # 20 * numElements individuals per generation
                    updating='deferred' if WORKERS != 1 else 'immediate',
                    disp=False,
                )

            # elif GLOBAL_METHOD == "dual_annealing":
            #     # dual_annealing doesn't natively support constraints,
            #     # so we add a penalty wrapper
            #     def penalized_objective(v):
            #         obj = evaluator.objective(v)
            #         # Check linear constraints manually
            #         max_min_vals = max_min_jacobian @ v
            #         if np.any(max_min_vals < 0):
            #             obj += 1e6 * np.sum(np.abs(np.minimum(max_min_vals, 0)))
            #         thick_vals = overall_thickness_jacobian @ v
            #         if np.any(thick_vals > 0.65):
            #             obj += 1e6 * np.sum(np.maximum(thick_vals - 0.65, 0))
            #         sec_thick_vals = second_thickness_jacobian @ v
            #         sec_limit = 0.65 - 2.25 / 25.4 - .125
            #         if np.any(sec_thick_vals > sec_limit):
            #             obj += 1e6 * np.sum(np.maximum(sec_thick_vals - sec_limit, 0))
            #         # Nonlinear tension constraint
            #         tensions = evaluator.all_grasps_constraint(v)
            #         if np.any(tensions > TENSION_LIMIT):
            #             obj += 1e6 * np.sum(np.maximum(tensions - TENSION_LIMIT, 0))
            #         return obj

            #     result = optimize.dual_annealing(
            #         penalized_objective,
            #         bounds=bounds,
            #         maxiter=MAX_ITER,
            #         seed=SEED,
            #         x0=v0,
            #         callback=lambda x, f, ctx: evaluator.global_callback(x, 0.0),
            #         minimizer_kwargs={
            #             "method": "trust-constr",
            #             "constraints": constraints_objects,
            #             "options": {
            #                 "maxiter": 50,
            #                 "gtol": 1e-3,
            #             },
            #         } if POLISH else None,
            #     )

            elif GLOBAL_METHOD == "shgo":
                result = optimize.shgo(
                    evaluator.objective,
                    bounds=bounds,
                    # constraints=[
                    #     {"type": "ineq", "fun": lambda v: (max_min_jacobian @ v)},
                    #     {"type": "ineq", "fun": lambda v: 0.65 - (overall_thickness_jacobian @ v)},
                    #     {"type": "ineq", "fun": lambda v: (0.65 - 2.25 / 25.4 - .125) - (second_thickness_jacobian @ v)},
                    #     {"type": "ineq", "fun": lambda v: TENSION_LIMIT - evaluator.all_grasps_constraint(v)},
                    # ],
                    options={"maxiter": MAX_ITER, "disp": True},
                    minimizer_kwargs={
                        "method": "trust-constr",
                        "constraints": constraints_objects,
                        "options": {"maxiter": 50, "gtol": 1e-3},
                    },
                )

            else:
                raise ValueError(f"Unknown global method: {GLOBAL_METHOD}")

        except KeyboardInterrupt:
            print("\n\nOptimization terminated by user.")
            if evaluator.best_x is not None:
                # Create a minimal result-like object
                result = OptimizeResult(
                    x=evaluator.best_x,
                    fun=evaluator.best_objective,
                    success=False,
                    message="Terminated by user",
                )
            else:
                print("No valid result found before termination.")
                exit(1)
        except stallException as e:
            print(f"\n\n{e}")
            result = OptimizeResult(
                x=evaluator.best_x if evaluator.best_x is not None else v0,
                fun=evaluator.best_objective if evaluator.best_objective < np.inf else np.nan,
                success=False,
                message=str(e),
            )

        # ─── Results Summary ─────────────────────────────────────────────────────
        print("\n\n")
        print(f"{'═' * 60}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"{'═' * 60}")
        print(f"  Success:    {result.success if hasattr(result, 'success') else 'N/A'}")
        print(f"  Message:    {result.message if hasattr(result, 'message') else 'N/A'}")
        print(f"  Obj value:  {result.fun}")
        print(f"  Evals:      {evaluator._eval_count}")
        print(f"  Solution:   {np.array2string(result.x, precision=4, suppress_small=True)}")
        print(f"{'═' * 60}\n")

        v_result = result.x
        name = "VCSAC_"+("RT" if RELAX_THICKNESS else "")+("DL" if DESIGN_LENGTHS else "")
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
            plt.ylabel("Objective value")
            plt.title(f"Global Optimization Convergence ({OBJECTIVE_MODE})")
            plt.grid(True)

    else:
        v_result = np.loadtxt("global_attempt.smx")
        np.savetxt("result_without_distal_forces")

    # ─── Post-processing ─────────────────────────────────────────────────────────
    for i in range(5):
        print("")
    resultFinger = createFingerFromVector(v_result)

    resultFinger.structure.minFactor = 1 / optimize.minimize_scalar(
        lambda theta: -resultFinger.structure.controllability([theta] * resultFinger.numJoints),
        bracket=(0, np.pi / 2), bounds=(0, np.pi / 2)
    ).fun

    q = 0
    print(resultFinger.structure([q] * resultFinger.numJoints))
    q = np.pi / 2
    print(resultFinger.structure([q] * resultFinger.numJoints))

    for function in resultFinger.structure.effortFunctions:
        if isinstance(function, VariableStrucMatrix.convergent_circles_extension_joint):
            print("type: extension")
        elif isinstance(function, VariableStrucMatrix.convergent_circles_joint_with_limit):
            print("type: flexion")
        print(f"idx: {function.idx}")
        print(f"c: {function.c}")
        print(f"r: {function.r}")

    qs = np.linspace(0, np.pi / 2, 75)
    tvecs = []
    tvecs2 = []
    conditions = []
    magnitudes = []
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
        condition = resultFinger.structure.controllability([q] * resultFinger.numJoints)
        magnitude = resultFinger.structure.get_magnitude([q] * resultFinger.numJoints)

        tvecs.append(tensions)
        tvecs2.append(tensions2)
        conditions.append(condition)
        magnitudes.append(magnitude)

    print("\n\n\n\n\n")
    print(f"Objective mode: {OBJECTIVE_MODE}")
    if OBJECTIVE_MODE == FingerEvaluator.OBJECTIVE_WORST_CASE_OVL:
        print("lowest OVL:")
        print(evaluator.worst_case_OVL(v_result))
        print(f"happens at: {evaluator.worst_pose_for_strength}")
        print("confirming:")
        print(f"{resultFinger.structure.get_magnitude(evaluator.worst_pose_for_strength)}")
    elif OBJECTIVE_MODE == FingerEvaluator.OBJECTIVE_WORST_CASE_TENSION:
        print("worst case tension (objective):")
        print(result.fun)

    print("worst case tension (from constraint):")
    wct = evaluator.worst_case_tension(v_result)
    print(wct)

    plt.figure("conditions")
    plt.plot(qs, conditions)
    plt.xlabel("Joint angle (rad)")
    plt.ylabel("Controllability")
    plt.grid(True)

    plt.figure("magnitudes")
    plt.plot(qs, np.array(magnitudes) / np.min(magnitudes))
    plt.xlabel("Joint angle (rad)")
    plt.ylabel("Normalized magnitude")
    plt.grid(True)

    resultFinger.structure.plotCapability([0] * resultFinger.numJoints, enforcePosTension=True)
    resultFinger.structure.plotCapability([np.pi / 2] * resultFinger.numJoints, enforcePosTension=True)
    plt.show()