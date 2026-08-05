import numpy as np
from scipy.differentiate import jacobian

def get_bifurcation_angles(target_prox_T1, target_prox_T2, target_term_T1, target_term_T2, prox_slip, term_slip, prop_prox):
    # target_prox_T1 = -2.174503356588717; target_prox_T2 = -2.418878165370875
    # target_term_T1 = -0.562503356588717; target_term_T2 = -1.650078165370875
    # prox_slip = -3.479205370541947; term_slip = -1.5000089509032453
    # prop_prox = 0.625
    
    
    ## Finding Correct Bifurcation Angle ##
    a = target_prox_T2 / prox_slip
    b = target_term_T2 / term_slip

    S = np.arccos((1 - a*a - b*b) / (2*a*b))

    angle_bot = np.arcsin(a * np.sin(S))
    angle_top = S - angle_bot

    T2_diag = np.sin(angle_top)/np.sin(angle_top+angle_bot)
    T2_lat  = np.sin(angle_bot)/np.sin(angle_top+angle_bot)

    return T2_diag, T2_lat

######### UTILITIES ##########
def trans(dx,dy,dz): # 3D translation matrix
    trans=np.eye(4) # create identity matrix
    trans[:,3] = np.array([dx,dy,dz,1]) # add translation vector to last column
    return trans
def rot_y(Q,L): # rotation and translation matrix about y-axis (ad-abduction angle) 
    c=np.cos(Q); s=np.sin(Q)
    dx,dy,dz=L,0,0 # translation vector across length of phalange (x)
    Ry=np.array([  # rotation matrix about y-axis
        [c,0,s,0], 
        [0,1,0,0],
        [-s,0,c,0],
        [0,0,0,1]], dtype=float)
    return Ry@trans(dx,dy,dz) # returns transformation matrix
def rot_z(Q,L): # rotation and translation matrix about z-axis (flexion-extension angle)
    c=np.cos(Q); s=np.sin(Q)
    dx,dy,dz=L,0,0 # translation vector across length of phalange (x)
    Rz=np.array([  # rotation matrix about z-axis
        [c,-s,0,0], 
        [s, c,0,0],
        [0, 0,1,0],
        [0, 0,0,1]], dtype=float)
    return Rz@trans(dx,dy,dz) # returns transformation matrix
def transform(Q,L): # transform the world frame to end effector frame given joint angles and link lengths
    # pass num_joints? check if number of link lengths are equal to number of joints?
    Q = np.asarray(Q)
    origin = np.transpose(np.array([0,0,0,1])) # world/global coordinate (x,y,z,1)=(0,0,0,1)
    # all_trans = [] # list of transformation matrices
    if len(Q) > len(L): # if there are more DOF than phalange lengths, assume first joint is ad-abduction and remaining are flexion-extension
        T = rot_y(Q[0],0) # MCP ad-abduction transformation matrix (0 length relative to origin) 
        Q = Q[1:] # remove first element of Q

        # all_trans.append(T) # add to list of transformation matrices
        origin = T @ origin # transform origin to new coordinate frame
    for idx in range(len(Q)): # for each joint angle and length, find the transformation matrix
        T = rot_z(Q[idx],L[idx]) # rotation and translation matrix
        # all_trans.append(T) # add to list of transformation matrices
        origin = T @ origin # transform origin (world) to end effector coordinate frame
    return(origin[:3]) # return final transformation matrix and list of transformation matrices for each joint angle and length

########### FINGER #############
def get_jacobian_at_pose(Q,L):
    # if singularity is detected??? (maybe) collapse the jacobian matrix (aka remove a DOF) and return the new jacobian matrix
    # figure out how to get the function to work with 2d and 3d inputs (1x3 and 1x4 arrays)
    # attribute of joint type....?
    def end_effector(Q): # input to scipy jacobian must ALWAYS be a function
           return (transform(Q,L)) # get the end effector position [x,y,z] without trailing 1
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
L = np.array([50e-3, 31e-3, 16e-3]) # phalange lenghts (m)
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

if __name__ == "__main__":
    q_flx = np.radians(np.array([0,45,45,10])); q_int = np.radians(np.array([0,45,10,10])); q_ext = np.radians(np.array([0,10,10,10]))
    Q = q_flx
    # print(f"########## CUEVAS MODEL #############\n{get_jacobian_at_pose_3(q_flx, L)} \n\n")
    print(f"################ TEST MODEL ############## \n {get_jacobian_at_pose(q_flx, L)}")
    # print(transform(Q,L))