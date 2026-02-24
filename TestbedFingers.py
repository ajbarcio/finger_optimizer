from strucMatrices import StrucMatrix, VariableStrucMatrix
import numpy as np

R = np.array([[0.400,0.400,0.19375,0.400],
              [0.0  ,0.400,0.400  ,0.300],
              [0.0  ,0.0  ,0.400  ,0.1472]])
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])

FixedControllable = StrucMatrix(R, D, name="Fixed Controllable")

D = np.array([[-1,1,1,1],
              [0,-1,1,1],
              [0,0,-1,1]])

R = np.array([[np.nan,np.nan,np.nan,np.nan],
              [0,     np.nan,np.nan,np.nan],
              [0,     0     ,np.nan,np.nan]])

# [(min, max, minim), (), etc...]
flexure_extents = [(0.0,0.35,0.2),(0.0,0.35,0.2),(0.0,0.35,0.2)]
# [(min, max), (), etc...]
extensure_extents = [(.25, .367),(.25, .367),(.25, .367)]

VaraibleArbitrary = StrucMatrix(R, D, ranges = [extensure_extents[0]]+[flexure_extents[0]]*3
                                              +[extensure_extents[1]]+[flexure_extents[1]]*2
                                              +[extensure_extents[2]]+[flexure_extents[2]],
                                       types = [VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit]*3
                                              +[VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit]*2
                                              +[VariableStrucMatrix.convergent_circles_extension_joint] + [VariableStrucMatrix.convergent_circles_joint_with_limit],
                                           F = np.array([50]*5),
                                      name="Arbitrary Variable")

