from strucMatrices import VariableStrucMatrix, secondaryDev
from utils import *
from finger import Finger, StructureKineMismatch
from scipy import optimize

numJoints = 3
numFlexs = 3
numExts = 3
numElements = numFlexs*2+numExts*2

F = [0,5,0]
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
    print(createFingerFromVector.called, v)
    return Fing

def objective(v):
    Fing = createFingerFromVector(v)
    
    res1 = optimize.minimize_scalar(lambda theta: 
                                   -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,   Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [theta]*Fing.numJoints, frame="EE"))))
                                 ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
    res2 = optimize.minimize_scalar(lambda theta: 
                                   -np.max(Fing.grip_to_tensions([theta]*Fing.numJoints,  -Fing.grasp_to_grip(Fing.grasp([F]*Fing.numJoints, [theta]*Fing.numJoints, frame="EE"))*0.25))
                                 ,bracket=(0,np.pi/2),bounds=(0,np.pi/2))
    objectiveRet = np.max([-res1.fun, -res2.fun])
    print(objectiveRet)
    return objectiveRet

if __name__=="__main__":
    
# EDITING TO VERSION WHERE V IS (max, (DISTANCE TO MIN))

    #v0 = [0.3]*numElements
    v0 = [.425,.25]*3+[.25]*6
    objectivewheee = objective([.425,.25]*3+[.25]*6)
    print(objectivewheee)
    result = optimize.minimize(objective,v0, bounds=[(.125,.5)]*numElements, options={"maxiter": int(1000/12)})
    print(result.fun)
    
    resultFinger = createFingerFromVector(result.x)
    q = 0
    print(resultFinger.structure([q]*resultFinger.numJoints))
    q = np.pi/2
    print(resultFinger.structure([q]*resultFinger.numJoints))
    
    qs = np.linspace(0,np.pi/2,75)
    tvecs = []
    tvecs2 = []
    for q in qs:
        tensions  = resultFinger.grip_to_tensions([q]*resultFinger.numJoints,  resultFinger.grasp_to_grip(resultFinger.grasp([F]*resultFinger.numJoints, [q]*resultFinger.numJoints, frame="EE")))
        tensions2 = resultFinger.grip_to_tensions([q]*resultFinger.numJoints, -resultFinger.grasp_to_grip(resultFinger.grasp([F]*resultFinger.numJoints, [q]*resultFinger.numJoints, frame="EE"))*0.25)
        
        tvecs.append(tensions)
        tvecs2.append(tensions2)
    plt.plot(qs, tvecs)
    plt.figure()
    plt.plot(qs, tvecs2)
    plt.show()

    # result of optimizer with random middle ground beginning [0.3, 0.24329802, 0.3, 0.24474896, 0.3, 0.28859318, 0.36322518, 0.30384326, 0.36281905, 0.30108046, 0.3624088,  0.30436718], 67ish lbs
    # result of optimizer with est manual option initial guess [0.125      0.5        0.38767142 0.5        0.25888551 0.125 0.5        0.5        0.5        0.5        0.37405896 0.45347323], 51ish lbs
    # ^^ this one is broken
    # result of optimizer with low beginning
    # result of optimizer with high beginning
    # result of optimizer with idead splits?