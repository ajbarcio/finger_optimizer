# TODO: need to convert cuevas spatial jacobian to correct coordinate frame (how to adjust moment arm matrix accordingly??)

import numpy as np
from scipy.differentiate import jacobian
import sympy as sp

def get_bifurcation_angles(target_prox_t2, target_prox_t3, target_term_t2, target_term_t3, prox_slip, term_slip, prop_prox):
    # target_prox_t2 = -2.174503356588717; target_prox_t3 = -2.418878165370875
    # target_term_t2 = -0.562503356588717; target_term_t3 = -1.650078165370875
    # prox_slip = -3.479205370541947; term_slip = -1.5000089509032453
    # prop_prox = 0.625
    
    
    ## Finding Correct Bifurcation Angle ##
    a = target_prox_t3 / prox_slip
    b = target_term_t3 / term_slip

    S = np.arccos((1 - a*a - b*b) / (2*a*b))

    angle_bot = np.arcsin(a * np.sin(S))
    angle_top = S - angle_bot

    t3_diag = np.sin(angle_top)/np.sin(angle_top+angle_bot)
    t3_lat  = np.sin(angle_bot)/np.sin(angle_top+angle_bot)

    return t3_diag, t3_lat

######### UTILITIES ##########
def trans(dx,dy,dz): # 3D translation matrix
    trans=np.eye(4) # create identity matrix
    trans[:,3] = np.array([dx,dy,dz,1]) # add translation vector to last column
    return trans

"""[WARNING] The coordinate frame for RAD finger differs from Valero Cuevas where RAD: (adab=Ry, fe=Rz) and Cuevas: (adab=Rz, fe=Rx). Comments and descriptions are relative to RAD 
        coordinate frame but current code is for Valero Cuevas.
    When switching between RAD and Cuevas coordinate frames, you must also reflect the length change in dx, dy, and dz where RAD: (dx,dy,dz=L,0,0) and Cuevas: (dx,dy,dz=0,L,0)"""

def fe_trans(Q,L): # rotation and translation matrix about y-axis (ad-abduction angle) 
    c=np.cos(Q); s=np.sin(Q)
    dx,dy,dz=0,L,0 # translation vector across length of phalange (x)
    Rx=np.array([  # rotation matrix about y-axis
        [1,0,0,0], 
        [0,c,-s,0],
        [0,s,c,0],
        [0,0,0,1]], dtype=float)
    return Rx @ trans(dx,dy,dz) # returns transformation matrix
def adab_trans(Q,L): # rotation and translation matrix about z-axis (flexion-extension angle)
    c=np.cos(Q); s=np.sin(Q)
    dx,dy,dz=0,L,0 # translation vector across length of phalange (x)
    Rz=np.array([  # rotation matrix about z-axis
        [c,-s,0,0], 
        [s, c,0,0],
        [0, 0,1,0],
        [0, 0,0,1]], dtype=float)
    return Rz @ trans(dx,dy,dz) # returns transformation matrix
def transform(Q,L): # transform the world frame to end effector frame given joint angles and link lengths
    # pass num_joints? check if number of link lengths are equal to number of joints?
    Q = np.asarray(Q)
    if len(Q) != len(L):
        raise ValueError("Each input Q must have a respective length L for all joints")
    origin = np.transpose(np.array([0,0,0,1])) # world/global coordinate (x,y,z,1)=(0,0,0,1)
    trans = np.eye(4) # identity matrix to compile all matrix transformations

    if len(Q) == 4: # if there are more DOF than phalange lengths, assume first joint is ad-abduction and remaining are flexion-extension
        T = adab_trans(Q[0],0) # MCP ad-abduction transformation matrix (0 length relative to origin) 
        Q = Q[1:]; L = L[1:] # remove first element of Q and L
        trans = trans @ T # transform origin to new coordinate frame

    for idx in range(len(Q)): # for each joint angle and length, find the transformation matrix
        T = fe_trans(Q[idx],L[idx]) # rotation and translation matrix
        
        trans = trans @ T
    pos = (trans @ origin)[:3] # apply full transform to origin

    if len(np.asarray(Q)) == len(L): # 3 DOF: no ad-adbuction
        return np.append(pos[1:],np.sum(Q)) # [x, y, sum of angles] (2x2)
    else:
        return np.append(pos, np.sum(Q)) # [x, y, z, sum of angles] (3x3)

########### FINGER #############
def get_jacobian_sympy():
    
    t1, t2, t3, t4 = sp.symbols('t1 t2 t3 t4')
    L1, L2, L3 = sp.symbols('L1 L2 L3')

    def rot_y_s(q):
        c, s = sp.cos(q), sp.sin(q)
        return sp.Matrix([[c,0,s,0],
                          [0,1,0,0],
                          [-s,0,c,0],
                          [0,0,0,1]])
    def rot_z_s(q):
        c, s = sp.cos(q), sp.sin(q)
        return sp.Matrix([[c,-s,0,0],
                          [s,c,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])
    def rot_x_s(q):
        c, s = sp.cos(q), sp.sin(q)
        return sp.Matrix([[1,0,0,0],
                          [0,c,-s,0],
                          [0,s,c,0],
                          [0,0,0,1]])
    def trans_s(dx, dy, dz):
        T = sp.eye(4)
        T[0,3]=dx; T[1,3]=dy; T[2,3]=dz
        return T

    T = rot_z_s(t1) * rot_x_s(t2)*trans_s(0,L1,0) * rot_x_s(t3)*trans_s(0,L2,0) * rot_x_s(t4)*trans_s(0,L3,0)
    sp.pprint(T)

    p = T * sp.Matrix([0,0,0,1])  # end effector position
    p[-1] = t2+t3+t4
    sp.pprint(sp.simplify(p))

    # Differentiate to get Jacobian
    theta = [t1, t2, t3, t4]
    J = sp.Matrix([[sp.diff(p[i], q) for q in theta] for i in range(len(p))])
    # print(J)
    J = sp.simplify(J)
    return(J)

def get_jacobian_at_pose(Q,L):
    # if singularity is detected??? (maybe) collapse the jacobian matrix (aka remove a DOF) and return the new jacobian matrix
    # figure out how to get the function to work with 2d and 3d inputs (1x3 and 1x4 arrays)
    # attribute of joint type....?
    def end_effector(Q): # input to scipy jacobian must ALWAYS be a function
        Q = np.asarray(Q)

        if Q.ndim > 1: # scipy jacobian reshapes Q and is very evil, code below uses witchcraft to fix the issue ¯\_ (ツ)_/¯
            Q_flat = Q.reshape(Q.shape[0], -1)  # collapse all batch dims → (4, k)
            return np.stack([transform(Q_flat[:, i], L) for i in range(Q_flat.shape[1])], axis=1)
        
        return transform(Q, L) # get the end effector position [x,y,z] without trailing 1
    J = jacobian(end_effector,Q) # get the jacobian matrix of the end effector position with respect to the joint angles
    return J.df # return the jacobian matrix

def get_jacobian_at_pose_3(THETA, lengths):
    J = np.array([
        [
            -1*np.cos(THETA[0])*(np.cos(THETA[1])*lengths[0] + np.cos(np.sum(THETA[1:3]))*lengths[1] + np.cos(np.sum(THETA[1:4]))*lengths[2]),
            np.sin(THETA[0])*(lengths[0]*np.sin(THETA[1]) + lengths[1]*np.sin(np.sum(THETA[1:3])) + lengths[2]*np.sin(np.sum(THETA[1:4]))),
            np.sin(THETA[0])*(lengths[1]*np.sin(np.sum(THETA[1:3])) + lengths[2]*np.sin(np.sum(THETA[1:4]))),
            lengths[2]*np.sin(THETA[0])*np.sin(np.sum(THETA[1:4]))
        ],
        [
            -1*(np.cos(THETA[1])*lengths[0] + np.cos(np.sum(THETA[1:3]))*lengths[1] + np.cos(np.sum(THETA[1:4]))*lengths[2])*np.sin(THETA[0]),
            -1*np.cos(THETA[0])*(lengths[0]*np.sin(THETA[1]) + lengths[1]*np.sin(np.sum(THETA[1:3])) + lengths[2]*np.sin(np.sum(THETA[1:4]))),
            -1*np.cos(THETA[0])*(lengths[1]*np.sin(np.sum(THETA[1:3])) + lengths[2]*np.sin(np.sum(THETA[1:4]))),
            -1*np.cos(THETA[0])*lengths[2]*np.sin(np.sum(THETA[1:4]))
        ],
        [
            0,
            np.cos(THETA[1])*lengths[0] + np.cos(np.sum(THETA[1:3]))*lengths[1] + np.cos(np.sum(THETA[1:4]))*lengths[2],
            np.cos(np.sum(THETA[1:3]))*lengths[1] + np.cos(np.sum(THETA[1:4]))*lengths[2],
            np.cos(np.sum(THETA[1:4]))*lengths[2]
        ],
        [0, 1, 1, 1]
    ], dtype=float)
    return J

#### VALERO CUERVAS FLEXION PARAMETERS ####
L = np.array([0, 50e-3, 31e-3, 16e-3]) # phalange lenghts (m)
PCSA = np.array([4.10, 7.3, 4.16, 4.32, 0.784, 0.72, 3.058]) # (cm^2) adjusted values
Fo = np.diag(PCSA*30) # Fo=diag(fo) where fo=PCSAxσ     (cm^2*N/cm^2) = N

R = np.array([ # moment arm of each tendon across each joint (m)
    # (FDP,            FDS,             DI,              PI,               EIP,             LUM,            EDC) (m)
    [ 2.91270673e-03,  1.45619048e-03,  6.79881327e-03,  6.96495580e-03,   3.01304379e-04,  4.62918718e-03, 1.19524896e-03],  # MCP ad-abd
    [ 9.03962540e-03,  9.99561738e-03,  2.00817179e-03,  4.01227521e-03,   9.37992146e-03,  7.02453290e-03, 9.37992146e-03],  # MCP
    [ 5.09361601e-03,  4.63740968e-03,  1.14638637e-08, -2.41887817e-03,   2.17450336e-03,  2.41887817e-03, 2.17450336e-03],  # PIP
    [ 3.64002601e-03,  1.90320646e-07,  1.14638637e-08,  1.65006832e-03,   5.62500000e-04,  1.65006832e-03, 5.62500000e-04]]) #DIP
D = np.array([ # direction matrix
    [1, 1,-1, 1, 1,-1,-1],
    [1, 1, 1, 1,-1, 1,-1],
    [1, 1,-1,-1,-1,-1,-1],
    [1,-1,-1,-1,-1,-1,-1]])

## dependent parameter implemented cuevas model
# R = np.array([ # moment arm of each tendon across each joint (m)
#         # (FDP,            FDS,             DI,              PI,               EIP,             LUM,            EDC) (m)
#         [ 2.91270673e-03,  1.45619048e-03,  6.79881327e-03,  6.96495580e-03,   3.01304379e-04,  4.62918718e-03, 1.19524896e-03],  # MCP ad-abd
#         [ np.nan,          np.nan,          np.nan,          np.nan,           9.37992146e-03,  7.02453290e-03, 9.37992146e-03],  # MCP
#         [ np.nan,          np.nan,          1.14638637e-08,  np.nan,           np.nan,          np.nan,         np.nan],  # PIP
#         [ 3.64002601e-03,  1.90320646e-07,  1.14638637e-08,  np.nan,           np.nan,          np.nan,         np.nan]]) #DIP
# D = np.array([ # direction matrix
    # [1, 1,-1, 1, 1,-1,-1],
    # [1, 1, 1, 1,-1, 1,-1],
    # [1, 1,-1,-1,-1,-1,-1],
    # [1,-1,-1,-1,-1,-1,-1]])

np.set_printoptions(precision=10, formatter={'float_kind':'{:.5f}'.format})

if __name__ == "__main__":
    q_flx = np.radians(np.array([0,45,45,10])); q_int = np.radians(np.array([0,45,10,10])); q_ext = np.radians(np.array([0,10,10,10]))
    # Q = np.array([45,45,10])
    Q = np.radians(np.array([45,45,10]))
    print(f"########## CUEVAS MODEL #############\n{get_jacobian_at_pose_3(q_flx, L)} \n\n")
    # print(f"########## CUEVAS MODEL #############\n{get_jacobian_at_pose_3(q_flx)[1:,1:]} \n\n")
    print(f"################ TEST MODEL ############## \n {get_jacobian_at_pose(Q,L)}")
    # sp.pprint(get_jacobian_sympy())