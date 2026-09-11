import numpy as np
import csv
from scipy.linalg import null_space

## finger.py, strucMatricies.py, TestbedFingers.py

"""Note: bifurcation angle adjusted from approximated model for greater precision
Same method used for proximal slip moment arm. Calculations found in Calibration.py

Changes: 
    * prox_slip [-3.44 >> -3.479205370541947]
    * angle_top [79 >> 78.66646801]
    * angle_bot [39 >> 38.29311372]"""

## Constant Variables (Nominal/Flexion) ##
l1=50e-3; l2=31e-3; l3=16e-3    # phalange lengths
# MCP ad-abduction, MCP Flexion, PIP, DIP (stays 10° for all poses)
q_flx = np.radians(np.array([0,45,45,10])); q_int = np.radians(np.array([0,45,10,10])); q_ext = np.radians(np.array([0,10,10,10]))

pose_dict = {"Flexion (45.0°, 45.0°, 10.0°)": q_flx,
             "Intermediate (45.0°, 10.0°, 10.0°)": q_int,
             "Extension (10.0°, 10.0°, 10.0°)": q_ext}

e = np.array([
    # Extension
    [0.2613, 0, 0.9356, 1, 1, 1, 1],       # 1.  pt. 40
    [0.2346, 0, 1, 1, 1, 1, 0.4141],       # 2.  pt. 42
    [0.2199, 0, 1, 1, 0, 1, 0.3014],       # 3.  pt. 45
    [0.201,  0, 1, 0.9709, 0, 1, 0.],       # 4.  pt. 10
    # Intermediate
    [1, 0.6842, 1, 0.4121, 0, 1, 0],       # 5.  pt. 7
    [1, 0.7245, 1, 0.2872, 0, 0, 0],       # 6.  pt. 8
    [1, 0.7629, 0.7089, 0, 0, 0, 0],       # 7.  pt. 31
    [1, 0.7281, 0.5666, 0, 0, 0, 1],       # 8.  pt. 35
    [1, 0.7208, 0.5722, 0, 1, 0, 1],       # 9.  pt. 37
    # Flexion
    [0.1928, 0.105, 0, 0, 1, 0, 1],        # 10. pt. 16
    [0.07022, 0, 0, 0.08573, 1, 0, 1],     # 11. pt. 18
    [0.1173, 0, 0, 0.1778, 1, 1, 1]        # 12. pt. 17
])
f = np.array([
    # Extension
    [8.663e-8,   8.235,  24.31,  7.424e-11],    # 1.  pt. 40
    [3.823e-9,   5.595,  32.72, -9.07e-11],     # 2.  pt. 42
    [-4.748e-9,  3.919,  37.74,  1.481e-10],    # 3.  pt. 45
    [-2.281e-9,  2.496,  41,   -2.087e-11],    # 4.  pt. 10
    # Intermediate
    [-2.273e-8, -26.32,  32.69, -6.838e-10],    # 5.  pt. 7
    [5.919e-10, -28.58,  23.05, -1.276e-10],    # 6.  pt. 8
    [-1.878e-8, -30.76,  13.31, -1.109e-9],     # 7.  pt. 31
    [-1.542e-9, -24.85,  -1.636,-5.066e-10],    # 8.  pt. 35
    [-1.406e-8, -23.39,  -5.169,-7.885e-10],    # 9.  pt. 37
    # Flexion
    [-1.387e-9,  1.442, -15.91, -6.661e-11],    # 10. pt. 16
    [-4.84e-9,   5.864, -14.64, -2.276e-11],    # 11. pt. 18
    [4.649e-9,   6.423,  -5.492, 2.496e-11]     # 12. pt. 17
])
M_paper=np.array([
    # fp         fs          di              pi          ei           lum          ec
    [-0.08941,  -0.04470,    0.20870,       -0.21380,   -0.009249,   0.14210,     0.03669],  # Fx
    [-0.04689,  -0.14960,    1.456e-17,      0.02480,    0.05200,    0.02480,     0.05200],  # Fy
    [ 0.06472,   0.001953,   0.05680,        0.20670,   -0.15180,    0.29190,    -0.15180],  # Fz
    [ 0.003081, -0.002352,   0.0001578,     -0.000685,  -0.0001649, -0.0004483,  -0.0001649] # Tx
])

## Jacobian for Index-Finger (4x4 Matrix) ##
def J_at_pos(q):
    if len(q) != 4:
        raise ValueError("Input q must be a 4-element array representing joint angles [q1, q2, q3, q4].")
    q1, q2, q3, q4 = q
    return np.array([
    [ #row sx
        -1*np.cos(q1)*(np.cos(q2)*l1+np.cos(q2+q3)*l2+np.cos(q2+q3+q4)*l3), #sq1
        np.sin(q1)*(l1*np.sin(q2)+l2*np.sin(q2+q3)+l3*np.sin(q2+q3+q4)), #sq2
        np.sin(q1)*(l2*np.sin(q2+q3)+l3*np.sin(q2+q3+q4)), #sq3
        l3*np.sin(q1)*np.sin(q2+q3+q4)], 
    [ #row sy
        -1*(np.cos(q2)*l1+np.cos(q2+q3)*l2+np.cos(q2+q3+q4)*l3)*np.sin(q1), #sq1
        -1*np.cos(q1)*(l1*np.sin(q2)+l2*np.sin(q2+q3)+l3*np.sin(q2+q3+q4)), #sq2
        -1*np.cos(q1)*(l2*np.sin(q2+q3)+l3*np.sin(q2+q3+q4)), #sq3
        -1*np.cos(q1)*l3*np.sin(q2+q3+q4)], #sq4
    [ #row sz
        0, #sq1
        np.cos(q2)*l1+np.cos(q2+q3)*l2+np.cos(q2+q3+q4)*l3, #sq2
        np.cos(q2+q3)*l2+np.cos(q2+q3+q4)*l3, #sq3
        np.cos(q2+q3+q4)*l3], #sq4
    #row stheta
    [0,1,1,1]])
def R_at_pos(q):
    if len(q) != 4:
        raise ValueError("Input q must be a 4-element array representing joint angles [q1, q2, q3, q4].")
    q1, q2, q3, q4 = q

    # Model Parameters (Nominal Values, modified by pose)
    MCP_FDP, MCP_DI, MCP_PI, PIP_FDP, PIP_FDS = 9.03962540, 2.00817179, 4.01227521, 5.09361601, 0.9104356651336974

    ## Winslow's Tendinous Rhombous (Variables) ##
    # Adjusted extensor mechanism parameters (Table I.1)
    prox_slip = -3.479205370541947; term_slip = -1.50; prop_prox = 0.625 # proportion to proximal slip (%) adjusted from 0.50
    angle_top = np.radians(78.66646801); angle_bot = np.radians(38.29311372)  # 79° and 39° adjusted from 10° and 30°
    T2_diag = np.sin(angle_top)/np.sin(angle_top+angle_bot); T2_lat  = np.sin(angle_bot)/np.sin(angle_top+angle_bot)                          # T2 group tension split (LUM, PI) from Figure 1.5 (diagonal/lateral band)

    # Initial assumption: assume model nominal is at flexed position
    if (q==q_flx).all(): # find pose to reflect percent change (Table I.6)
        pass # do nothing
    # If we are not flexed, we are either in intermediate
    elif (q==q_int).all() or (q==q_ext).all(): # flex >> int % change
        # Intermediate changes:
        PIP_FDP*=.90      # PIP FDP -10% change
        prox_slip*=.80    # proximal slip -20% change
        angle_top*=.77    # top bifurcation angle -23% change
        angle_bot*=1.10   # bottom bifurcation angle +10% change

        # Or we are in extension
        if (q==q_ext).all(): # int >> ext % change
            MCP_FDP*=.80    # MCP FDP -20% change
            MCP_DI*=1.80    # MCP DI +80% change
            MCP_PI*=.40     # MCP PI -60% change
            prop_prox*=1.20 # prop. to prox slip +20% change
            PIP_FDS = 0.8
    else: 
        raise ValueError("Input q must be flexion, intermediate, or extension")

    # Proximal slip component for each tendon group (I dont think this math is correct)
    prox_T2 = prox_slip*T2_lat # T2 Group (PI, LUM)
    term_T2 = term_slip*T2_diag # ^^

    prox_T1 = prox_slip*prop_prox # T1 group (EIP, EDC)
    term_T1 = term_slip*(1-prop_prox) # ^^

    R=np.array([ 
        # (FDP,          FDS,                            DI,                PI,             EIP,             LUM,            EDC) (mm)
        # MCP adduction/abduction DOF no.1
        [ 2.91270673,    (0.5*2.91238096),               -6.79881327,       6.96495580,     0.301304379,    -4.62918718,     -1.19524896], # -4.61 and 6.94 adjusted from -3.84 and 4.08
        # MCP flexion/extension DOF no.2
        [ MCP_FDP,       (1.105755707531863*MCP_FDP),    MCP_DI,            MCP_PI,        -9.37992146,      7.02453290,    (-9.37992146)], # 9 and -9.32 adjusted from 12 and -7.77 
        # PIP DOF no. 3
        [ PIP_FDP,       (PIP_FDS*PIP_FDP),              -1.14638637e-05,   prox_T2,        prox_T1,         prox_T2,        prox_T1],
        # DIP DOF no. 3
        [ 3.64002601,    -1.90320646e-04,                -1.14638637e-05,   term_T2,        term_T1,         term_T2,        term_T1]    
    ]) * 1e-3 # (mm to m) conversion
    return R
def Fo_at_pos(q):
    ## F_o (7x7 Matrix Diagonalized) (FDP, FDS, DI, PI, EIP, LUM, EDC) ##
    # reference: pg. 13 & Table I.1
    PCSA = np.array([4.10, 7.3, 4.16, 4.32, 0.784, 0.72, 3.058]) # (cm^2) adjusted values
    if (q==q_flx).all(): # find pose to reflect percent change (Table I.6)
        pass # do nothing
    elif (q==q_int).all() or (q==q_ext).all(): # flex >> int % change
        PCSA*=np.array([1.00, 1.25, 1.10, 0.67, 1.43, 1.00, 1.14]) # FDS +25%, DI +10%, PI -33%, EIP +43%, EDC +14% change
        if (q==q_ext).all(): # int >> ext % change
            PCSA*=np.array([1.00, 1.00, 1.00, 1.50, 1.00, 1.00, 1.00]) # PI +50% change
    Fo = np.diag(PCSA*30) # Fo=diag(fo) where fo=PCSAxσ     (cm^2*N/cm^2) = N
    return(Fo)
def M_at_pos(q):
    M = np.linalg.inv(J_at_pos(q)).T @ R_at_pos(q) # Only J^-T@R (to compare to M (pg. 87)), needs Fo for actual model
    return M
def Mo_at_pos(q):
    Mo = np.linalg.inv(J_at_pos(q)).T @ R_at_pos(q)@Fo_at_pos(q) # Only J^-T@R (to compare to M (pg. 87)), needs Fo for actual model
    return Mo
def null_model():
    # Find all nullspaces of M=J^-TRFo and RFo for each pose (flexion, intermediate, extension)
    for name, pose in pose_dict.items():
        M = Mo_at_pos(pose)
        RFo = R_at_pos(pose)@Fo_at_pos(pose)
        print(f"\n\033[1m#################  {name}  #################")
        print(f"\033[1mNullspace of M=J^-TRFo:\n\033[0m{null_space(M)}"); print(f"\033[1mNullspace of RFo:\n\033[0m{null_space(RFo)}")
        print(f"\n\n\033[1mM=J^-TRFo:\n\033[0m{M}\n\033[1mRFo:\n\033[0m{RFo}")
def output_csv():
# ── Open CSV once, write header, then all rows ────────────────────────────────
    with open("solver_results.csv", "w", newline="") as csv.out:
        writer = csv.writer(csv.out)
        writer.writerow([
            "Pose", "Point",
            "Model_Fy", "Model_Fz",
            "Paper_Fy", "Paper_Fz",
            "Diff_Fy",  "Diff_Fz",
            "Overall_Percentage_Error"
        ])

        for q_pose, pose_label in pose_dict.items():
            q = q_pose
            print(f"\n\n\n\n\033[1m#################{np.degrees(q)}#################")

            for i in np.arange(1,13):
                point = i
                            
                R_paper = (J_at_pos(q_flx).T @ M_paper)
                R = R_at_pos(q)
                # print(f"\tDI\tPI\tEIP\tLUM\tEDC");print(f"R_paper:\n{R_paper[-2:, -4:]*1e3}");print(f"R:\n{R[-2:, -4:]*1e3}");print(prox_T2, term_T2, prox_T1, term_T1)
                # print(f"R_paper:\n{R_paper*1e3}"); print(f"R:\n{R*1e3}"); print(f"Difference:\n{(R-R_paper)*1e3}")
                # print(R-R_paper)

                Fo=Fo_at_pos(q)
                M = np.linalg.inv(J_at_pos(q)).T @ R # Only J^-T@R (to compare to M (pg. 87)), needs Fo for actual model
                M_percent_error = (abs(M - M_paper) / (M_paper)) * 100
                # print(M_percent_error)
                # print(M-M_paper)

                f_test=M@Fo@e[point-1] # adding Fo for actual model calculation

                diff=abs(f_test-f[point-1])/((f_test + f[point-1]))*100

                print(f"\033[1mPoint: {point}       ({np.degrees(q[1])}°, {np.degrees(q[2])}°, {np.degrees(q[3])}°)")
                print(f"\033[0mModel:{f_test[1:3]} Paper:{f[point-1,1:3]}");print(f"Percentage Difference:{diff[1:3]}")
                print(f"Overall Percentage Error: {np.linalg.norm(diff[1:3])}\n")

                writer.writerow([
                    pose_label,point,
                    round(f_test[1], 5), round(f_test[2], 5),
                    round(f[point-1, 1], 5),    round(f[point-1, 2], 5),
                    round(diff[1], 5),   round(diff[2], 5),
                    round(np.linalg.norm(diff[1:3]), 5)
                ])

# np.set_printoptions(precision=10, formatter={'float_kind':'{:.5f}'.format})


if __name__ == "__main__":    
    # for i in np.arange(1,13):
        # for q_opt in [q_ext, q_int, q_flx]:
        q, point = q_flx, 5

        R_paper = (J_at_pos(q_flx).T @ M_paper)
        R = R_at_pos(q)
        # print(f"\tDI\tPI\tEIP\tLUM\tEDC");
        # print(f"R_paper:\n{R_paper[-2:, -2:]*1e3}");print(f"R:\n{R[-2:, -2:]*1e3}") ## Compare Dependent Parameters with Paper Moment Arms (R)
        # print(f"R_paper:\n{R_paper*1e3}"); print(f"R:\n{R*1e3}"); print(f"Difference:\n{(R-R_paper)*1e3}") ## Compare R with R_paper

        Fo=Fo_at_pos(q)
        M = np.linalg.inv(J_at_pos(q)).T @ R # Only J^-T@R (to compare to M (pg. 87)), needs Fo for actual model
        M_percent_error = (abs(M - M_paper) / (M_paper)) * 100
        # print(M_percent_error)
        # print(M-M_paper)
        # print(f"\n\n\n\n\033[1m#################{np.degrees(q)}#################")
        # print(f"\033[1mModel:\n\033[0m{M}\n\033[1mPaper:\n\033[0m{M_paper}\n\033[1mDifference:\n\033[0m{M-M_paper}\n")

        f_test=M@Fo@e[point-1] # adding Fo for actual model calculation

        diff=abs(f_test-f[point-1])/((f_test + f[point-1]))*100
        # print(f"point: {point}    ({np.degrees(q[1])}°, {np.degrees(q[2])}°, {np.degrees(q[3])}°)")
        # print(f"Model:{f_test[1:3]} Paper:{f[point-1,1:3]}");print(f"Percentage Difference:{diff[1:3]}")
        
        # print(f"point: {point}    ({np.degrees(q[1])}°, {np.degrees(q[2])}°, {np.degrees(q[3])}°)")
        # print(f"Model:{f_test[1:3]} Paper:{f[point-1,1:3]}");print(f"Percentage Difference:{diff[1:3]}")
        # print(f"Overall Percentage Error: {np.linalg.norm(diff[1:3])}")

        # output_csv()
        # null_model()
        print(J_at_pos(q_flx))
        
    # print(f"M_paper:\n{M_paper}");print(f"M:\n{M}");print(f"Difference:\n\n{M-M_paper}")