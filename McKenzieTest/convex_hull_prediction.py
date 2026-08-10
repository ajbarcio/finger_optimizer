# TODO: We now have 2-constraint plotting for feasible force and torque regions, need to make definition for 1-constraint
# TODO: finish feasible force and torque polytopes
import numpy as np
import itertools as it
import scipy.optimize as opt
import scipy.spatial as spa
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .Solver import Mo_at_pos, q_flx, q_int, q_ext, R_at_pos, Fo_at_pos

num_excitations = 7
num_constraints = 5 # CONSTRAINT(1): num_constraints = 6

def generate_binary_lists(length):
    # Generates a cartesian product of [0, 1] repeated 'length' times
    for combo in it.product([0, 1], repeat=length):
        yield list(combo)

def solve_for_feasible_excitations(A, b):
    c = np.zeros(len(b))
    res = opt.linprog(c, A_eq=A, b_eq=b, bounds=(0,1))
    # if res.success:
    #     print(np.linalg.norm(res.con))
    return res.x, res.success

def get_convex_capability_at_pos(pose,name,num_constraints): # finds feasible force and torque regions
    # print(f"-----------# Trying Pose {name} {pose} #-----------")
    num_sols_for_pose = 0
    unique_sol_for_pose = set()
    unique_forces_for_pose = []
    unique_torques_for_pose = []

    for idx in it.combinations(np.arange(0,num_excitations), num_constraints):
        idx = np.array([int(idx_i) for idx_i in idx])
        # print(idx)
        for edge in generate_binary_lists(num_constraints):
            edge = np.array(edge)
            # print(edge)
            b = edge
            A = np.zeros([num_constraints,num_excitations])
            for i in range(num_constraints):
                A[i,idx[i]]=1
            if (num_excitations-num_constraints)==2: # TWO CONSTRAINTS (CUEVAS MODEL)
                b = np.concatenate([[0,0], b])
                A = np.vstack((Mo_at_pos(pose)[0,:],Mo_at_pos(pose)[-1,:],A))
            elif (num_excitations-num_constraints)==1: # ONE CONSTRAINT (TORQUE ALLOWANCE)
                b = np.concatenate([[0], b])
                A = np.vstack((Mo_at_pos(pose)[0,:], A))
                        
            excitation, success = solve_for_feasible_excitations(A, b)
            if not success:
                pass
            else:
                # print(f"for pose {pose} found excitation {excitation}")
                # print(np.linalg.norm(con))
                num_sols_for_pose += 1
                unique_sol_for_pose.add(tuple(excitation))
    # print("--")
            
    # print(f"found {num_sols_for_pose} intersections at pose {pose}")
    # print(f"found {len(unique_sol_for_pose)} unique intersections at pose {pose}")
    for u_ext in list(unique_sol_for_pose):
        # print(f"For the exitation: {np.array(list(u_ext))}")

        force = Mo_at_pos(pose) @ u_ext
        # print(f"This force is produced: {force}")

        torques = R_at_pos(pose) @ Fo_at_pos(pose) @ u_ext
        # print(f"These joint torques are produced: {torques}")

        unique_forces_for_pose.append(force[1:3]) # CONSTRAINT(1): unique_forces_for_pose.append(force[1:])
        unique_torques_for_pose.append(torques[1:])
        if (pose == q_ext).all():
            unique_forces_q_ext.append(force)
        elif (pose == q_int).all():
            unique_forces_q_int.append(force)
        elif (pose == q_flx).all():
            unique_forces_q_flx.append(force)
    # print(f"")
    convex_forces_for_pose = spa.ConvexHull(np.array(unique_forces_for_pose))
    return convex_forces_for_pose, np.array(unique_torques_for_pose)

def plot_ffr_at_pose(pose, name): # ffr: feasible force region (2d)
    font=16
    convex_forces_for_pose = get_convex_capability_at_pos(pose, name)[0]
    spa.convex_hull_plot_2d(convex_forces_for_pose, ax=plt.gca())
    plt.title(F"Feasible Force Region for {name} Pose (N)", fontsize=font); plt.xlabel("Fz", fontsize=font); plt.ylabel("Fy", fontsize=font)

def plot_ftr_at_pose(pose, name): # ftr: feasible torque region (3d)
    font=16
    unique_torques_for_pose = np.array(get_convex_capability_at_pos(pose, name)[1])
    plt.axes(projection='3d')
    plt.gca().scatter(unique_torques_for_pose[:, 0], unique_torques_for_pose[:, 1], unique_torques_for_pose[:, 2], color='blue')
    plt.title(f"Feasible Torque Region for {name} Pose", fontsize=font)
    plt.gca().set_xlabel('τ₁',fontsize=font); plt.gca().set_ylabel('τ₂',fontsize=font); plt.gca().set_zlabel('τ₃',fontsize=font)

def feasible_force_polytope(pose, name): # single constraint version of plot_ffr_at_pose
    return None
def feasible_torque_polytope(pose, name): # single constraint version of plot_ftr_at_pose
    return None
unique_forces_q_ext = []
unique_forces_q_int = []
unique_forces_q_flx = []

pose_dict = {"Extended": q_ext, 
             "Intermediate": q_int,
             "Flexed": q_flx}

# for name, pose in pose_dict.items():
    # if name != "bah":
    #     plt.figure()
    #     plot_ffr_at_pose(pose, name)

        ########### ONE CONSTRAINT MODEL ###############
        # plt.figure() # FORCES
        # plt.axes(projection='3d')
        # vertices = [convex_forces_for_pose.points[simplex] for simplex in convex_forces_for_pose.simplices]
        # hull_surface = Poly3DCollection(vertices, alpha=0.3, facecolor='cyan', edgecolor='red', linewidths=0.5)
        # plt.gca().add_collection3d(hull_surface)
        # plt.tight_layout()
        # xlim = plt.gca().get_xlim()
        # ylim = plt.gca().get_ylim()
        # zlim = plt.gca().get_zlim()
        # plt.gca().plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
        # plt.gca().plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
        # plt.gca().plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
        # plt.gca().set_xlabel('Fy',fontsize=font)
        # plt.gca().set_ylabel('Fz',fontsize=font)
        # plt.gca().set_zlabel('τx',fontsize=font)
        # plt.title(f"Feasible Force Region for {name} pose",fontsize=font)

        # convex_torques_for_pose = spa.ConvexHull(np.array(unique_torques_for_pose))
        # plt.figure() # TORQUES
        # plt.axes(projection='3d')
        # unique_torques_for_pose = np.array(unique_torques_for_pose)

        # vertices = [convex_torques_for_pose.points[simplex] for simplex in convex_torques_for_pose.simplices]
        # hull_surface = Poly3DCollection(vertices, alpha=0.3, facecolor='cyan', edgecolor='red', linewidths=0.5)
        # plt.gca().add_collection3d(hull_surface)
        # plt.tight_layout()
        # xlim = plt.gca().get_xlim()
        # ylim = plt.gca().get_ylim()
        # zlim = plt.gca().get_zlim()
        # plt.gca().plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
        # plt.gca().plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
        # plt.gca().plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
        # plt.gca().set_xlabel('τ₁',fontsize=font)
        # plt.gca().set_ylabel('τ₂',fontsize=font)
        # plt.gca().set_zlabel('τ₃',fontsize=font)
        # plt.title(f"Feasible Torque Region for {name} pose",fontsize=font)
plt.show()

