import numpy as np
import itertools as it
import scipy.optimize as opt
from scipy.spatial import ConvexHull, QhullError, convex_hull_plot_2d
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ValeroCuervasModel import Mo_at_pos, q_flx, q_int, q_ext, R_at_pos, Fo_at_pos

num_excitations = 7
num_constraints = 5

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

pose_dict = {"extended": q_ext, 
             "intermediate": q_int,
             "flexed": q_flx}

for name, pose in pose_dict.items():
    # uniqueForces = []
    # print(f"-----------# Trying Pose {pose} #-----------")
    num_sols_for_pose = 0
    unique_sol_for_pose = set()
    uniqueForces = []
    uniqueTorques = []
    # if (pose == q_ext).all():
    #     print(np.vstack((Mo_at_pos(pose)[0,:],Mo_at_pos(pose)[-1,:])))
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
            b = np.concatenate([[0,0],b])
            A = np.vstack((Mo_at_pos(pose)[0,:],Mo_at_pos(pose)[-1,:],A))
                        
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

        uniqueForces.append(force[1:3])
        uniqueTorques.append(torques[1:])
        if (pose == q_ext).all():
            unique_forces_q_ext.append(force)
        elif (pose == q_int).all():
            unique_forces_q_int.append(force)
        elif (pose == q_flx).all():
            unique_forces_q_flx.append(force)
    # print(f"")

    if name == 'flexed':
        hullForces = ConvexHull(np.array(uniqueForces))
        try: 
            hullTorques = ConvexHull(np.array(uniqueTorques))
        except QhullError:
            hullTorques = ConvexHull(np.array(uniqueTorques), qhull_options='QJ')
        
        plt.figure()
        convex_hull_plot_2d(hullForces, ax=plt.gca())
        plt.title(f"Feasible Force Region for {name} pose")
        
        plt.figure()
        plt.axes(projection='3d')
        uniqueTorques = np.array(uniqueTorques)
        plt.gca().scatter(uniqueTorques[:, 0], uniqueTorques[:, 1], uniqueTorques[:, 2], color='blue')
        
        for simplex in hullTorques.simplices:
            # units for plotting polytope
            triangle = hullTorques.points[simplex]
            plt.gca().add_collection3d(Poly3DCollection([triangle]))

        vertices = [hullTorques.points[simplex] for simplex in hullTorques.simplices]
        hull_surface = Poly3DCollection(vertices, alpha=0.3, facecolor='cyan', edgecolor='red', linewidths=0.5)
        plt.gca().add_collection3d(hull_surface)
        plt.title(f"Feasible Torque Region for {name} pose")
        # plt.plot(convex_forces.points)
        # plt.plot(convex_forces.vertices)

plt.show()
        # print(b)
        # print(A)
        # print(idx)