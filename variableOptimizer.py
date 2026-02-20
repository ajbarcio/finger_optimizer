from strucMatrices import VariableStrucMatrix, secondaryDev
from utils import *
from finger import Finger, StructureKineMismatch
from scipy import optimize
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult
from combinatorics import create_decoupling_matrix

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
    VSM = VariableStrucMatrix(R, D, F=[50]*numTendons, ranges = [es[0]]+[fs[0]]*3
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

    def range(self, v):
        Fing = self._get_finger(v)
        res = optimize.minimize_scalar(lambda theta: -(np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,   
                                                                  Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, 
                                                                                                [theta]*Fing.numJoints, 
                                                                                                frame="EE")))) - 
                                                       np.min(Fing.grip_to_tensions([theta]*Fing.numJoints,   
                                                                  Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, 
                                                                                                [theta]*Fing.numJoints, 
                                                                                                frame="EE")))))
                                       ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        return abs(res.fun)

    def worst_case_tension(self, v):
        Fing = self._get_finger(v)
        # Locate maximum on 1D function (cheap) for a) flex grip at rated (5) lb grip and b) 1lb extension tip force
        res1 = optimize.minimize_scalar(lambda theta: 
                                    -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,   
                                                                  Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, 
                                                                                                [theta]*Fing.numJoints, 
                                                                                                frame="EE"))))
                                    ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        res2 = optimize.minimize_scalar(lambda theta: 
                                    -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,  
                                                                  Fing.tip_wrench_at_pose_to_grip([theta]*Fing.numJoints,
                                                                                                   -F*0.1,
                                                                                                   frame="EE")))
                                    ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
        objectiveRet = np.max([-res1.fun, -res2.fun])
        # print(objectiveRet)
        return objectiveRet

    def decouplability(self, v):
        Fing = self._get_finger(v)
        
        def decouplability_eval(theta, Fing):
            M = create_decoupling_matrix(Fing.structure([theta]*3))
            success = identify_strict_central(M)
            if success:
                return 0.0
            else:
                stiffness_dir = null_space(M)
                # scale to unit minimum: this just generally results in rounder numbers
                scale = 1.0/np.min(stiffness_dir) if np.min(stiffness_dir) != 0 else 1.0
                # scale = 1.0
                stiffnesses = stiffness_dir * scale
                K_A = np.diag(stiffnesses)
                K_J = Fing.structure().dot(K_A).dot(Fing.structure().T)
                decoupledness = np.linalg.norm(K_J-np.diag(np.diag(K_J)))
                return decoupledness
        
        M = create_decoupling_matrix(Fing.structure([0]*3))
        success = identify_strict_sign_central(M)
        if success:
            print("You have an inherently decouplable structure")
            return 0.0
        else:
            worst_case_decoupled = optimize.minimize_scalar(lambda theta: -decouplability_eval(theta, Fing),
                                                            bracket=(0,np.pi/2),bounds=(0,np.pi/2))
            return np.abs(worst_case_decoupled.fun)

    def 

    # def actuator_to_cartesian_ellipse(self, v):


if __name__=="__main__":
    replace = False
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

        result = optimize.minimize(evaluator.decouplability,
                                v0,
                                bounds=[(.125,.5)]*numElements,
                                constraints=constraints_objects, 
                                options={"maxiter": int(100),
                                            # "finite_diff_rel_step": 1e-4,
                                         "verbose": 1,
                                            # "keep_feasible": True,
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
    print("result vector:", v_result)
    q = 0
    print(resultFinger.structure([q]*resultFinger.numJoints))
    q = np.pi/2
    print(resultFinger.structure([q]*resultFinger.numJoints))
    
    qs = np.linspace(0,np.pi/2,75)
    tvecs = []
    tvecs2 = []
    conditions = []

    q = 0
    resultFinger.structure.plotCapability([q]*resultFinger.numJoints)
    resultFinger.structure.plotCapability([q]*resultFinger.numJoints, enforcePosTension=True)
    q = np.pi/2
    resultFinger.structure.plotCapability([q]*resultFinger.numJoints)
    resultFinger.structure.plotCapability([q]*resultFinger.numJoints, enforcePosTension=True)

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
    # print(tvecs)
    # print(tvecs2)
    plt.figure("Flexion Grasp Tensions")
    plt.plot(qs, tvecs)
    plt.figure("Extension Tip Wrench Tensions")
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
    # result of optimizer with condition objective, remain feasible,
    #   and informed initial guess: [0.37186058 0.33161087 0.33068164 0.27605197 0.33462263 0.29978758 0.35823917 0.34383309 0.38030427 0.35522027 0.34556798 0.32352671]
    # result of optimizer with range objective, remain feasible,
    #   and informed initial guess: [0.40526359 0.32743508 0.46607661 0.41733753 0.49212554 0.4864513 0.4991715  0.48895549 0.49701881 0.42181808 0.42260185 0.3690979 ]
    # result of optimizing for decouplability:
    #   [0.44647548 0.34287771 0.47558069 0.29465648 0.52269942 0.3068709 0.4377329  0.43223791 0.40167322 0.36529423 0.38384292 0.33848131]