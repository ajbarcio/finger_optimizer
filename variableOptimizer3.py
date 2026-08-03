from strucMatrices import VariableStrucMatrix, secondaryDev
from utils import *
from finger import Finger, StructureKineMismatch
from scipy import optimize
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult
import inspect
from feasibility import *

import warnings
# warnings.filterwarnings('error', category=RuntimeWarning)

# from multiprocessing import Pool
# from concurrent.futures import ProcessPoolExecutor

class stallException(Exception):
    pass

def createFingerFromVector(v) -> Finger:
    v = np.asarray(v)
    # Only accept valid vectors
    # if np.any(np.isnan(v)) or np.any(np.isinf(v)) or np.any(v < 0.0625) or np.any(v > 0.5):
    #     # Option 1: Error-out (recommended for debugging)
    #     raise ValueError(f"Received invalid design vector: {v}")

    if not hasattr(createFingerFromVector, "called"):
        createFingerFromVector.called = 1
    else:
        createFingerFromVector.called+=1

    fs = []
    es = []
    for i in range(numFlexs+numExts):
        if i < numFlexs:
            minv, maxv, reliefv = v[3*i+2], v[3*i], v[3*i+1]
            fs.append((v[3*i+2], v[3*i], v[3*i+1]))
            # if not (maxv >= reliefv >= minv):
            #     raise ValueError(f"Invalid variable order at flex segment {i}: max={maxv}, relief={reliefv}, min={minv}")
        else:
            j = i - numFlexs
            k = numFlexs * 3
            es.append((v[k+2*j+1], v[k+2*j]))
    ranges = es[0:1]+fs[0:1]*3+es[1:2]+fs[1:2]*2+es[2:3]+fs[2:3]
    # print(ranges)

    R_use = R.copy()
    np.fill_diagonal(R_use, v[-numFixed:])

    VSM = VariableStrucMatrix(R_use, D, F=[50]*numTendons, ranges = ranges,
                                                        types = [VariableStrucMatrix.convergent_circles_joint_with_limit]*6,
                                                         name = f"VSM{createFingerFromVector.called}")
    # worstCondition = optimize.minimize_scalar(lambda theta: -VSM.controllability([theta]*Fing.numJoints),
    #                                 bracket=(0,np.pi/2),bounds=(0,np.pi/2))
    # minFactor = 1/worstCondition
    # VSM.minFactor = minFactor
    Fing = Finger(VSM, lengths, tensionLimit=np.max(VSM.F))
    # print(createFingerFromVector.called, v)
    return Fing

class FingerEvaluator:
    def __init__(self):
        self._v_save = None
        self._v_prev = None
        self.curr_res = None
        self._finger = None
        self._cb_last_lines = 0
        self._threshold_times = 0
        self.optimalities = []
        self._curr_wct = None

    def _get_finger(self, v):
        v = np.asarray(v)
        if self._v_save is None or not np.array_equal(v, self._v_save):
            self._v_save = v.copy()
            self._finger = createFingerFromVector(v)
            # print(createFingerFromVector.called, v)
        return self._finger

    # def callback(self, intermediate_result: OptimizeResult):
    #     print("CALLED~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     self.optimalities.append(intermediate_result.optimality)

    def global_callback(self, xk, convergence=None):
        print("Current best:", xk)
        if convergence is not None:
            print("Convergence:", convergence)

    # def callback(self, intermediate_result: OptimizeResult):
        self.curr_res = intermediate_result
        # Clear previous output
        print("\r" + "\033[F" * self._cb_last_lines, end="")

        self.optimalities.append(intermediate_result.optimality)
        lines = [
            f"iter: {intermediate_result.niter}",
            f"optimality: {intermediate_result.optimality:.3e}",
            # f"worst case tension: {intermediate_result.constr_violation+50.0}",
            f"worst case tension: {self._curr_wct}",
            "x: " + np.array2string(
                intermediate_result.x,
                precision=4,
                suppress_small=True
            )+"                                                                  ",
            # f"worst constraint violation (note, not all constraints have same units): {intermediate_result.}"
        ]
        if self._v_prev is not None:
            step = np.linalg.norm(intermediate_result.x-self._v_prev, ord=np.inf)
            if step < 1e-3:
                # print("\n")
                self._threshold_times+=1
                lines.append(f"step lower than 1 thou {self._threshold_times} times in a row")
            else:
                self._threshold_times=0
        ## THIS IS THE IMPORTANT PART ##
        if self._threshold_times > 10:
            raise stallException(f"Optimizer stalled out at low step value ater {self._threshold_times} in a row with a change less than 1 thou")
        ## OVER ##
        # Print fresh block
        print("\r\033[2K" + "\n".join(lines), end="", flush=True)
        self._cb_last_lines = len(lines)

        self._v_prev = intermediate_result.x

    def callback(self, intermediate_result: OptimizeResult):
        self.curr_res = intermediate_result
        self.optimalities.append(intermediate_result.optimality)

        lines = [
            f"iter: {intermediate_result.niter}",
            f"optimality: {intermediate_result.optimality:.3e}",
            f"worst case tension: {self._curr_wct}",
            "x: " + np.array2string(
                intermediate_result.x,
                precision=4,
                suppress_small=True
            ),
        ]
        if self._v_prev is not None:
            step = np.linalg.norm(intermediate_result.x - self._v_prev, ord=np.inf)
            if step < 1e-3:
                self._threshold_times += 1
                lines.append(f"step lower than 1 thou {self._threshold_times} times in a row")
            else:
                self._threshold_times = 0
        if self._threshold_times > 20:
            print("Optimizer stalled out at low step value after", self._threshold_times, "in a row with a change less than 1 thou")
            raise stallException()
        for line in lines:
            print(line)
        self._v_prev = intermediate_result.x

    def condition(self, v):
        Fing = self._get_finger(v)
        # Fing.structure.controllability()
        res = optimize.minimize_scalar(lambda theta: -Fing.structure.controllability([theta]*Fing.numJoints),
                                    bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        return abs(res.fun)

    def magnitude_scale(self, v):
        Fing = self._get_finger(v)
        return Fing.structure.get_magnitude([0]*Fing.numJoints)/Fing.structure.get_magnitude([np.pi/2]*Fing.numJoints)

    def ultimate_magnitude(self, v):
        Fing = self._get_finger(v)
        res = optimize.minimize_scalar(lambda theta: -Fing.structure.get_magnitude([theta]*Fing.numJoints),bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        magnitude = res.fun
        # magnitude = -Fing.structure.get_magnitude([np.pi/2]*Fing.numJoints)
        # print(f"successfully evaluated magnitude as {magnitude}")
        return magnitude

    def strength_increase(self, v):
        Fing = self._get_finger(v)
        strength_ratio = (np.linalg.norm(
                              Fing.grip_to_tensions([np.pi/2]*Fing.numJoints, Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [np.pi/2]*Fing.numJoints, frame="EE"))))
                              /
                          np.linalg.norm(
                              Fing.grip_to_tensions([0]*Fing.numJoints, Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [0]*Fing.numJoints, frame="EE"))))
                         )
        check_valid(strength_ratio)
        return strength_ratio

    def worst_case_tension(self, v):
        Fing = self._get_finger(v)
        # Locate maximum on 1D function (cheap) for a) flex grip at rated (5) lb grip and b) 1lb extension tip force
        # Locate maximum on 1D function (cheap) for a) flex tip wrench at rated (12) lb grip and b) 1.2 lb extension tip force
        try:
            res1 = optimize.minimize_scalar(lambda theta:
                                        # -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,   Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [theta]*Fing.numJoints, frame="EE"))))
                                        -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,   Fing.tip_wrench_at_pose_to_grip([theta]*Fing.numJoints, F, frame="EE")))
                                        ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
            res2 = optimize.minimize_scalar(lambda theta:
                                        -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,  Fing.tip_wrench_at_pose_to_grip([theta]*Fing.numJoints, -F*0.1, frame="EE")))
                                        ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
            objectiveRet = np.max([-res1.fun, -res2.fun])
        except Exception as e:
            print(f'error in tension calcs {e}')
        # print(f"'successfully' evaluated worst case tension as: {objectiveRet}")
        # print(objectiveRet)
        # WHOA NOT SURE THIS IS WISE
        self._curr_wct = objectiveRet
        check_valid(objectiveRet)

        return objectiveRet

    def worst_case_bend_radius(self, v):
        Fing = self._get_finger(v)
        try:
            rs = []
            for function in Fing.structure.effortFunctions:
                rs.append(function.r)
        except Exception as e:
            print(f"error in bend radius calcs: {e}")
        # worst case (minimum) bend radius
        return np.min(rs)

def check_valid(x):
    if np.isnan(x) or np.isinf(x):
        caller = inspect.stack()[1][3]
        print(f"BAD VALUE in {caller}")

if __name__=="__main__":
    numJoints = 3
    numTendons = numJoints+1
    numFlexs = 3
    numExts = 0
    numFixed = 3
    numElements = numFlexs*3+numExts*2+numFixed

    F = np.array([0,8,0])
    lengths = [1.4,1.4,1.2]
    # R = secondaryDev.R
    R = np.array([[0.25, np.nan, np.nan, np.nan],
                  [0,    0.25,   np.nan, np.nan],
                  [0,    0,      0.25,   np.nan]])
    # print(R)
    D = secondaryDev.D
    # print(D)
    replace = False
# EDITING TO VERSION WHERE V IS (max, (DISTANCE TO MIN))
    # initialize evaluator
    if not replace:
        evaluator = FingerEvaluator()

        # Constrain each slider so that the finger remains dimensionally feasible (keep maxs maxs, mins mins, etc)
        max_relief_jacobian = np.array([[0]*3*i+[1,-1,0]+[0]*(numElements-3*(i+1)) for i in range(numFlexs)]+
                                       [[0]*numFlexs*3+[0]*2*i+[1,-1]+[0]*2*(numExts-i-1)+[0]*numFixed for i in range(numExts)])
        relief_min_jacobian = np.array([[0]*3*i+[0,1,-1]+[0]*(numElements-3*(i+1)) for i in range(numFlexs)])
        # print(max_min_jacobian)

        adaptive_bounds_jacobian = np.vstack([max_relief_jacobian, relief_min_jacobian])
        # print(adaptive_bounds_jacobian)
        
        # Constraints such that the finger maintains an overall acceptable form factor
        overall_thickness_jacobian = np.zeros([numFlexs, numElements])

        for i in range(numFlexs):
            overall_thickness_jacobian[i, 3*i] = 1
            overall_thickness_jacobian[i, 3*numFlexs+i] = 1

        second_thickness_jacobian = np.zeros([numFlexs, numElements])

        for i in range(numFlexs):
            second_thickness_jacobian[i, 3*i] = np.sqrt(2)/2
            second_thickness_jacobian[i, 3*numFlexs+i] = 1

        thickness_jacobian = np.vstack([overall_thickness_jacobian, second_thickness_jacobian])
        # print(np.array2string(thickness_jacobian, suppress_small=True, max_line_width=150))

        constraints_objects = [
            # Constrain each relevant dimensional max greater than its associated min
            LinearConstraint(
                A=adaptive_bounds_jacobian,
                lb=0,
                ub=np.inf,
                keep_feasible=True,
                ),
            # Constrain each relief greater than its associated min
            # LinearConstraint(
            #     A=relief_min_jacobian,
            #     lb=0,
            #     ub=np.inf,
            #     keep_feasible=True,
            #     ),
            # Constrain each pair of max flexion and min extension to be less than the established thickness
            # Constrain each pair of max flexion (perpendicular to joint) and max extension to be less than the established thickness
            LinearConstraint(
                A=thickness_jacobian,
                lb=0,
                ub=0.65,
                keep_feasible=True,
            ),
            # Constrain the max extension to be such that the overall thickness of the finger remains good
            # LinearConstraint(
            #     A=second_thickness_jacobian,
            #     lb=0,
            #     ub=0.65,
            #     keep_feasible=True
            # ),
            # Constrain worst case tension less than 50 lb
            NonlinearConstraint(
                fun=evaluator.worst_case_tension,
                lb=0,
                ub=50.0,
                # finite_diff_rel_step=1e-4
            ),
            NonlinearConstraint(
                fun=evaluator.worst_case_bend_radius,
                lb=0.45,
                ub=0.9
            )
            ]

        # v0 = [(.5+.125)/2]*numElements
        # v0 = [.5,.4]*numFlexs+[.3,.25]*numExts
        # v0 = [.35,.2]*numFlexs+[0.37,0.25]*numExts
        # Initialize with the best solution of previous version's results:
        # v0 = [0.428,.3153, .1711, .435, .263, .1196, .438, .272, .1217]+[.315, .1796, .333, .160, .3263, .158]
        # Initialize with some middle ground
        v0 = [.375, .25, .2125, .375, .25, .2125, .375, .25, .2125]+[.25]*3
        # v0 = [.425,.25]*3+[.25]*6
        # objectivewheee = evaluator.worst_case_tension([.425,.25]*3+[.25]*6)
        # print(objectivewheee)

        bounds=[(.0625+2.25/2/25.4,.5)]*(numFlexs*3+numExts*2)+[(0.25,0.5)]*numFixed
        best_x, is_feasible, _ = find_feasible_initial_guess(constraints_objects, 
                                                             bounds, 
                                                             num_attempts = 50,
                                                             x0_guess=v0)
        if is_feasible:
            v0=best_x
        else:
            print(f"could not find feasible guess, using your dumb guess:\n{v0}")
        print(v0)
        initialFinger = createFingerFromVector(v0)

        # for constraint in constraints_objects:
        #     if isinstance(constraint, LinearConstraint):
        #         print(constraint.A @ (v0))
        #     else:
        #         print(constraint.fun(v0))

        print(initialFinger.structure([0]*3))
        print(initialFinger.structure([np.pi/2]*3))
        try:
            result = optimize.minimize(evaluator.strength_increase,
                                    v0,
                                    bounds=[(.0625+2.25/2/25.4,.5)]*(numFlexs*3+numExts*2)+[(0.25,0.5)]*numFixed,
                                    constraints=constraints_objects,
                                    options={
                                            # "maxiter": int(50),
                                                # "finite_diff_rel_step": 1e-4,
                                            "verbose": 1,
                                            "gtol": 1e-4,
                                            #  "keep_feasible": True,
                                                # "finite_diff_rel_step": None,
                                                # "finite_diff_abs_step": 1e-4
                                            #  "workers": -1
                                                },
                                    callback=evaluator.callback,
                                    method="trust-constr",
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
        # print("about to start optimization")
        # with ProcessPoolExecutor(max_workers=10) as executor:
        # with Pool(10) as pool:
        # result = optimize.differential_evolution(evaluator.ultimate_magnitude,
        #                     bounds=[(.0625,.5)]*numElements,
        #                     constraints=constraints_objects,
        #                     callback=evaluator.global_callback,
        #                     workers=-1,
                            #    minimizer_kwargs= {
                            #         "method": "trust-constr",
                            #         "constraints": constraints_objects,
                            #         "options": {
                            #             "maxiter": 50,
                            #             "verbose": 1,
                            #             "xtol": 1e-3
                            #         },
                            #         "callback": evaluator.callback
                                # }
        # )
        print(result)
        print(result.fun)
        v_result = result.x
        np.savetxt("prev3.smx", v_result)
        plt.plot(evaluator.optimalities)
    else:
        v_result = np.loadtxt("prev3.smx")
    # print(evaluator.optimalities)
    # if replace:
    #     resultFinger = createFingerFromVector(v_replace)
    # else:
    for i in range(5):
        print("")
    resultFinger = createFingerFromVector(v_result)

    resultFinger.structure.minFactor = 1/optimize.minimize_scalar(
                                                  lambda theta: -resultFinger.structure.controllability([theta]*resultFinger.numJoints),
                                                                 bracket=(0,np.pi/2),bounds=(0,np.pi/2)).fun

    q = 0
    print(resultFinger.structure([q]*resultFinger.numJoints))
    q = np.pi/2
    print(resultFinger.structure([q]*resultFinger.numJoints))

    for function in resultFinger.structure.effortFunctions:
        if isinstance(function, VariableStrucMatrix.convergent_circles_extension_joint):
            print("type: extension")
        elif isinstance(function, VariableStrucMatrix.convergent_circles_joint_with_limit):
            print("type: flexion")
        print(f"idx: {function.idx}")
        print(f"c: {function.c}")
        print(f"r: {function.r}")

    qs = np.linspace(0,np.pi/2,75)
    tvecs = []
    tvecs2 = []
    conditions = []
    magnitudes = []
    for q in qs:
        tensions  = resultFinger.grip_to_tensions([q]*resultFinger.numJoints,
                                                  resultFinger.grasp_to_grip(resultFinger.grasp(
                                                                                                [F]*resultFinger.numJoints,
                                                                                                [q]*resultFinger.numJoints,
                                                                                                frame="EE")))
        tensions2 = resultFinger.grip_to_tensions([q]*resultFinger.numJoints,
                                                  resultFinger.tip_wrench_at_pose_to_grip([q]*resultFinger.numJoints,
                                                                                          -F*0.1,
                                                                                          frame="EE"))

        condition = resultFinger.structure.controllability([q]*resultFinger.numJoints)
        magnitude = resultFinger.structure.get_magnitude([q]*resultFinger.numJoints)

        tvecs.append(tensions)
        tvecs2.append(tensions2)
        conditions.append(condition)
        magnitudes.append(magnitude)
    plt.figure("flex grasps")
    plt.plot(qs, tvecs)
    plt.figure("extn grasps")
    plt.plot(qs, tvecs2)
    plt.figure("conditions")
    plt.plot(qs, conditions)
    plt.figure("magnitudes")
    plt.plot(qs, np.array(magnitudes))
    # print(resultFinger.structure.npJoints)
    # resultFinger.structure.plotCapability([0]*resultFinger.numJoints)
    resultFinger.structure.plotCapability([0]*resultFinger.numJoints, enforcePosTension=True)
    # resultFinger.structure.plotCapability([np.pi/2]*resultFinger.numJoints)
    resultFinger.structure.plotCapability([np.pi/2]*resultFinger.numJoints, enforcePosTension=True)
    plt.show()

# np.savetxt("prev.smx", result.x)

    # result of optimizer with random middle ground beginning [0.3, 0.24329802, 0.3, 0.24474896, 0.3, 0.28859318, 0.36322518, 0.30384326, 0.36281905, 0.30108046, 0.3624088,  0.30436718], 67ish lbs
    # result of optimizer with est manual option initial guess [0.125      0.5        0.38767142 0.5        0.25888551 0.125 0.5        0.5        0.5        0.5        0.37405896 0.45347323], 51ish lbs
    # ^^ this one is broken
    # result of optimizer with low beginning
    # result of optimizer with high beginning
    # result of optimizer with idead splits?1000

    # result of optimizer with condition objective and medial initial guess [0.21350445 0.21209735 0.27698973 0.2526812  0.27707309 0.27206885 0.35411562 0.32669469 0.3279375  0.30864402 0.24131075 0.25118998]
    # result of optimizer with condition objective, remain feasible,
    #   and informed extreme initial guess: [0.43566228 0.38578858 0.36561368 0.29610095 0.3297116  0.2339563 0.36849157 0.34032965 0.38316362 0.35185861 0.4167588  0.3840339 ]

    # result of optimizer with strength increase objective and best-current-design initial guess:
    # 4.645934180950757608e-01
    # 2.772725194709753649e-01
    # 3.999286357141569326e-01
    # 2.143762366118264484e-01
    # 4.423210755608414368e-01
    # 2.622491727704909237e-01
    # 2.408983539795451767e-01
    # 1.510581096737557139e-01
    # 2.922009753471491167e-01
    # 1.996003209312147142e-01
    # 2.298057080774267569e-01
    # 1.634217687383705542e-01

    # result of optimizer with bend radius objective:
    # [0.3996 0.3808 0.1712 0.3793 0.3007 0.1576 0.4144 0.3427 0.183  0.2994
    #  0.2088 0.3162 0.1911 0.2608 0.165 ]