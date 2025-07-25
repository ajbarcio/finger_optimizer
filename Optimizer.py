import numpy as np
from matplotlib import pyplot as plt
from strucMatrices import StrucMatrix, centeredType1, centeredType2, centeredType3, naiiveAmbrose, quasiHollow

def test3dofn1():
    # S = centeredType1
    # S = centeredType2
    # S = centeredType3
    # S = naiiveAmbrose
    S = quasiHollow
    print(S())
    print(S.validity)
    print(S.nullSpace)
    S.plotCapability()
    # S.plotGrasp([0.5,0,0])
    # S.plotGrasp([2,0,0])

    # print(S.contains([0.5,0,0]))
    # print(S.contains([2,0,0]))

    plt.show()

def optimize_standard(numJoints, numTendons):
    pass

class Finger():
    def __init__(self, lengths=None, numJoints=None, numTendons=None, S=None, grasps=None) -> None:
        if S is None:
            R = np.array([[1,0.5,1,0.5],
                        [0, 1 ,0.5,0.5],
                        [0, 0 ,1 ,1]])
            D = np.array([[1,1,-1,-1],
                        [0,1,-1,-1],
                        [0,0,-1,1]])
            self.S = StrucMatrix(R,D)
        else:
            self.S = S

        if grasps is not None:
            self.grasps = grasps
        else:
            self.grasps = []

def main():
    test3dofn1()

if __name__ == "__main__":
    main()