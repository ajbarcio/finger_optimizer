from strucMatrices import VariableStrucMatrix, secondaryDev
from utils import *
from finger import Finger, StructureKineMismatch
from scipy import optimize
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult

numJoints = 3
numFlexs = 3
numExts = 3
numElements = numFlexs*2+numExts*2

F = np.array([0,5,0])
lengths = [1.4,1.4,1.2]
R = secondaryDev.R
# print(R)
D = secondaryDev.D
# print(D)

def createFingerFromVector(v) -> Finger:
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
    VSM = VariableStrucMatrix(R, D, F=[50]*numJoints, ranges = [es[0]]+[fs[0]]*3
                                                              +[es[1]]+[fs[1]]*2
                                                              +[es[2]]+[fs[2]],
                                                       types = [VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit]*3
                                                              +[VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit]*2
                                                              +[VariableStrucMatrix.convergent_circles_extension_joint]+[VariableStrucMatrix.convergent_circles_joint_with_limit],
                                                        name = f"VSM{createFingerFromVector.called}")
    Fing = Finger(VSM, lengths, tensionLimit=np.max(VSM.F))
    # print(createFingerFromVector.called, v)
    return Fing

class FingerEvaluator:
    def __init__(self):
        self._v_prev = None
        self._finger = None
        self._cb_last_lines = 0

        self.optimalities = []

    def _get_finger(self, v):
        v = np.asarray(v)
        if self._v_prev is None or not np.array_equal(v, self._v_prev):
            self._v_prev = v.copy()
            self._finger = createFingerFromVector(v)
            # print(createFingerFromVector.called, v)
        return self._finger

    # def callback(self, intermediate_result: OptimizeResult):
    #     print("CALLED~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     self.optimalities.append(intermediate_result.optimality)

    def callback(self, intermediate_result: OptimizeResult):
        # Clear previous output
        print("\r" + "\033[F" * self._cb_last_lines, end="")

        self.optimalities.append(intermediate_result.optimality)

        lines = [
            f"iter: {intermediate_result.niter}",
            f"optimality: {intermediate_result.optimality:.3e}",
            f"worst case tension: {intermediate_result.constr_violation+50.0}",
            "x: " + np.array2string(
                intermediate_result.x,
                precision=4,
                suppress_small=True
            )+"                                                                  "
        ]

        # Print fresh block
        print("\r\033[2K" + "\n".join(lines), end="", flush=True)
        self._cb_last_lines = len(lines)

    def condition(self, v):
        Fing = self._get_finger(v)
        # Fing.structure.controllability()
        res = optimize.minimize_scalar(lambda theta: -Fing.structure.controllability([theta]*Fing.numJoints),
                                    bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        return abs(res.fun)

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
        return objectiveRet

if __name__=="__main__":
    replace = True
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
        constraints_objects = [
            # Constrain each max greater than its associated min
            LinearConstraint(
                A=max_min_jacobian,
                lb=0,
                ub=np.inf,
                keep_feasible=True,
                ),
            NonlinearConstraint(
                fun=evaluator.worst_case_tension,
                lb=0,
                ub=50.0,
                # finite_diff_rel_step=1e-4
            )
            ]
        
        # v0 = [(.5+.125)/2]*numElements
        v0 = [.5,.4]*numFlexs+[.3,.25]*numExts
        
        # v0 = [.425,.25]*3+[.25]*6
        # objectivewheee = evaluator.worst_case_tension([.425,.25]*3+[.25]*6)
        # print(objectivewheee)

        result = optimize.minimize(evaluator.condition,
                                v0,
                                bounds=[(.125,.5)]*numElements,
                                constraints=constraints_objects, 
                                options={"maxiter": int(100),
                                            # "finite_diff_rel_step": 1e-4,
                                         "verbose": 1,
                                        #  "keep_feasible": True,
                                            # "finite_diff_rel_step": None,
                                            # "finite_diff_abs_step": 1e-4
                                            },
                                callback=evaluator.callback,
                                method="trust-constr",
                                )
        print(result.fun)
        v_result = result.x
        np.savetxt("prev.smx", v_result)
        plt.plot(evaluator.optimalities)
        plt.figure()
    else:
        v_result = np.loadtxt("prev.smx")
    # print(evaluator.optimalities)
    # if replace:
    #     resultFinger = createFingerFromVector(v_replace)
    # else:
    resultFinger = createFingerFromVector(v_result)
    
    q = 0
    print(resultFinger.structure([q]*resultFinger.numJoints))
    q = np.pi/2
    print(resultFinger.structure([q]*resultFinger.numJoints))
    
    qs = np.linspace(0,np.pi/2,75)
    tvecs = []
    tvecs2 = []
    conditions = []
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

        tvecs.append(tensions)
        tvecs2.append(tensions2)
        conditions.append(condition)
    plt.plot(qs, tvecs)
    plt.figure()
    plt.plot(qs, tvecs2)
    plt.figure()
    plt.plot(qs, conditions)
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