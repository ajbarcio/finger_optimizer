import numpy as np
import itertools as it
import scipy.optimize as opt
import scipy.spatial as spa
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Solver import Mo_at_pos, q_flx, q_int, q_ext, R_at_pos, Fo_at_pos

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

print("--")

unique_forces_q_ext = []
unique_forces_q_int = []
unique_forces_q_flx = []

pose_dict = {"Extended": q_ext, 
             "Intermediate": q_int,
             "Flexed": q_flx}

for name, pose in pose_dict.items():
    # unique_froces_for_pose = []
    print(f"-----------# Trying Pose {name} {pose} #-----------")
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
            b = np.concatenate([[0,0], b]) # CONSTRAINT(1): b = np.concatenate([[0], b])
            A = np.vstack((Mo_at_pos(pose)[0,:],Mo_at_pos(pose)[-1,:],A)) # CONSTRAINT(1): A = np.vstack((Mo_at_pos(pose)[0,:], A))
                        
            excitation, success = solve_for_feasible_excitations(A, b)
            if not success:
                pass
            else:
                # print(f"for pose {pose} found excitation {excitation}")
                # print(np.linalg.norm(con))
                num_sols_for_pose += 1
                unique_sol_for_pose.add(tuple(excitation))
    print("--")
            
    print(f"found {num_sols_for_pose} intersections at pose {pose}")
    print(f"found {len(unique_sol_for_pose)} unique intersections at pose {pose}")
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
    print(f"")

    if name != "bah":
        convex_forces_for_pose = spa.ConvexHull(np.array(unique_forces_for_pose))
        font=16

        ########## TWO CONSTRAINT MODEL #################
        plt.figure() # forces
        spa.convex_hull_plot_2d(convex_forces_for_pose, ax=plt.gca())
        plt.title(F"Feasible Force Region for {name} Pose (N)", fontsize=font); plt.xlabel("Fz", fontsize=font); plt.ylabel("Fy", fontsize=font)

        plt.figure() # torques
        plt.axes(projection='3d')
        unique_torques_for_pose = np.array(unique_torques_for_pose)
        plt.gca().scatter(unique_torques_for_pose[:, 0], unique_torques_for_pose[:, 1], unique_torques_for_pose[:, 2], color='blue')
        plt.title(f"Feasible Torque Region for {name} Pose", fontsize=font)
        plt.gca().set_xlabel('τ₁',fontsize=font); plt.gca().set_ylabel('τ₂',fontsize=font); plt.gca().set_zlabel('τ₃',fontsize=font)

        # ########### ONE CONSTRAINT MODEL ###############
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

# TODO: Since we now have 3-d force vectors, update the feasible force region plotting to 3-d as well
# TODO: FIgure out why extension pose is still null space