from strucMatrices import VariableStrucMatrix, secondaryDev
from utils import *
from finger import Finger, StructureKineMismatch
from scipy import optimize
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult

from multiprocessing import Pool
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
            fs.append((0, v[2*i], v[2*i+1]))
        else:
            es.append((v[2*i+1], v[2*i]))
    ranges = [es[0]]+[fs[0]]*3+[es[1]]+[fs[1]]*2+[es[2]]+[fs[2]]
    # print(ranges)
    
    VSM = VariableStrucMatrix(R, D, F=[50]*numTendons, ranges = [es[0]]+[fs[0]]*3
                                                              +[es[1]]+[fs[1]]*2
                                                              +[es[2]]+[fs[2]],
                                                       types = [VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit]*3
                                                              +[VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit]*2
                                                              +[VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit],
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

    def callback(self, intermediate_result: OptimizeResult):
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
        return -Fing.structure.get_magnitude([np.pi/2]*Fing.numJoints)

    def strength_increase(self, v):
        Fing = self._get_finger(v)
        return (np.linalg.norm(
                    Fing.grip_to_tensions([np.pi/2]*Fing.numJoints, Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [np.pi/2]*Fing.numJoints, frame="EE"))))
                    /
                np.linalg.norm(
                    Fing.grip_to_tensions([0]*Fing.numJoints, Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [0]*Fing.numJoints, frame="EE"))))
                )
    
    def worst_case_tension(self, v):
        Fing = self._get_finger(v)
        # Locate maximum on 1D function (cheap) for a) flex grip at rated (5) lb grip and b) 1lb extension tip force
        res1 = optimize.minimize_scalar(lambda theta:
                                    -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,   Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [theta]*Fing.numJoints, frame="EE"))))
                                    ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        res2 = optimize.minimize_scalar(lambda theta:
                                    -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,  Fing.tip_wrench_at_pose_to_grip([theta]*Fing.numJoints, -F*0.1, frame="EE")))
                                    ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        objectiveRet = np.max([-res1.fun, -res2.fun])
        # print(objectiveRet)
        # WHOA NOT SURE THIS IS WISE
        self._curr_wct = objectiveRet
        return objectiveRet

if __name__=="__main__":
    numJoints = 3
    numTendons = numJoints+1
    numFlexs = 3
    numExts = 3
    numElements = numFlexs*2+numExts*2

    F = np.array([0,5,0])
    lengths = [1.4,1.4,1.2]
    R = secondaryDev.R
    # print(R)
    D = secondaryDev.D
    # print(D)
    replace = False
# EDITING TO VERSION WHERE V IS (max, (DISTANCE TO MIN))
    # initialize evaluator
    if not replace:
        evaluator = FingerEvaluator()
        constraints_dicts = [
            {
            # Constrian worst case tension less than 50 lbs
            "type": "ineq",
            "fun":  lambda v: 50.0 - evaluator.worst_case_tension(v)
            },
            {
            # Constrain each max greater than its associated min
            "type": "ineq",
            "fun":  lambda v: v[1::2] - v[0::2]
            }]
        max_min_jacobian = np.array([np.roll(row, shift) for row, shift in
            zip(-np.eye(numElements//2, numElements, k=1)+
                np.eye(numElements//2, numElements, k=0),np.arange(numElements))])
        # print(max_min_jacobian)
        overall_thickness_jacobian = np.zeros([numFlexs, numElements])
        for i in range(numFlexs):
            overall_thickness_jacobian[i, 2*i] = 1
            overall_thickness_jacobian[i, 2*numFlexs+2*i+1] = 1
        second_thickness_jacobian = np.hstack([np.zeros([numFlexs*2, numFlexs*2]), np.eye(numExts*2)])
        constraints_objects = [
            # Constrain each max greater than its associated min
            LinearConstraint(
                A=max_min_jacobian,
                lb=0,
                ub=np.inf,
                keep_feasible=True,
                ),
            # Constrain each pair of max flexion and min extension to be less than the established thickness
            LinearConstraint(
                A=overall_thickness_jacobian,
                lb=0,
                ub=0.65,
                keep_feasible=True,
            ),
            # Constrain the max extension to be such that the overall thickness of the finger remains good
            LinearConstraint(
                A=second_thickness_jacobian,
                lb=0,
                ub=0.65-2.25/25.4-.125,
                keep_feasible=True
            ),
            # Constrain worst case tension less than 50 lb
            NonlinearConstraint(
                fun=evaluator.worst_case_tension,
                lb=0,
                ub=50.0,
                # finite_diff_rel_step=1e-4
            )
            ]

        # v0 = [(.5+.125)/2]*numElements
        # v0 = [.5,.4]*numFlexs+[.3,.25]*numExts
        v0 = [.35,.2]*numFlexs+[0.37,0.25]*numExts
        # v0 = [.425,.25]*3+[.25]*6
        # objectivewheee = evaluator.worst_case_tension([.425,.25]*3+[.25]*6)
        # print(objectivewheee)
        try:
            result = optimize.minimize(evaluator.ultimate_magnitude,
                                    v0,
                                    # bounds=[(.125,.5)]*numElements,
                                    constraints=constraints_objects,
                                    options={
                                            # "maxiter": int(50),
                                                # "finite_diff_rel_step": 1e-4,
                                            "verbose": 1,
                                            "gtol": 1e-3,
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
        print("about to start optimization")
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

        print(result.fun)
        v_result = result.x
        np.savetxt("prev.smx", v_result)
        plt.plot(evaluator.optimalities)
    else:
        v_result = np.loadtxt("prev.smx")
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
    plt.plot(qs, np.array(magnitudes)/np.min(magnitudes))
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