import numpy as np
import warnings
from strucMatrices import VariableStrucMatrix, StrucMatrix, secondaryDev, primaryDev
from utils import trans, jac, clean_array, hArray, generate_binary_lists
from scipy.optimize import nnls, lsq_linear, linprog
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import itertools

from matplotlib import pyplot as plt

from McKenzieTest.convex_hull_prediction import plot_ffr_at_pose, pose_dict

import matplotlib.animation as animation

class StructureKineMismatch(Warning):
    def __init__(self, message='WARNING: number of link lengths does not match \
                                         number of joints'):
        super().__init__(message)

# def grasp(self, F, q, l=None, frame="World"):
#     return Finger.Grasp(self, F, q, l, frame)

class Grasp:
    """
    A grasp is a defined by reaction forces at each joint (F)
                                                and joint angles (q)
    """
    def __init__(self, F, q, l=None, frame="World"):
        self.l = l if l is not None else [1] * len(q)
        self.F = F
        self.q = q
        self.frame=frame

class Finger():

    # // a grip is a vector of numJoints torques

    def __init__(self, structure: VariableStrucMatrix | StrucMatrix, lengths, tensionLimit=50):
        self.structure = structure
        self.lengths = lengths
        if self.structure.numJoints != len(self.lengths):
            warnings.warn(StructureKineMismatch())

        self.numJoints = self.structure.numJoints
        self.numTendons = self.structure.numTendons
        # print(self.numJoints, self.numTendons)
        self.tensionLimit = tensionLimit

    def get_jacobian_at_pose(self, THETA, lengths=None):
        # F = trans(THETA, self.lengths)
        if lengths is None:
            lengths=self.lengths
        J = jac(THETA, lengths)
        return J

    def get_jacobian_at_pose_2(self, THETA, lengths=None):
        if lengths==None:
            lengths=self.lengths
        ## This is a test 32x3 array for a 3-joint finger that is only being
        ## used to test get_jacobian_at_pose to make
        ## sure the frame used here matches the frame in the paper
        J = np.array([
            [-(lengths[0]*np.sin(THETA[0])+lengths[1]*np.sin(np.sum(THETA[0:2]))+lengths[2]*np.sin(np.sum(THETA))),
             -(lengths[1]*np.sin(np.sum(THETA[0:2]))+lengths[2]*np.sin(np.sum(THETA))),
              -lengths[2]*np.sin(np.sum(THETA))],
            [  lengths[0]*np.cos(THETA[0])+lengths[1]*np.cos(np.sum(THETA[0:2]))+lengths[2]*np.cos(np.sum(THETA)),
               lengths[1]*np.cos(np.sum(THETA[0:2]))+lengths[2]*np.cos(np.sum(THETA)),
               lengths[2]*np.cos(np.sum(THETA))],
            [  1,
               1,
               1]])
        return J
        # -
    def get_jacobian_at_pose_3(self, THETA, lengths=None):
        if lengths==None:
            lengths=self.lengths
        J = np.array([
            [
                -1*np.cos(THETA[0])*(np.cos(THETA[1])*self.lengths[0] + np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2]),
                np.sin(THETA[0])*(self.lengths[0]*np.sin(THETA[1]) + self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
                np.sin(THETA[0])*(self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
                self.lengths[2]*np.sin(THETA[0])*np.sin(np.sum(THETA[1:4]))
            ],
            [
                -1*(np.cos(THETA[1])*self.lengths[0] + np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2])*np.sin(THETA[0]),
                -1*np.cos(THETA[0])*(self.lengths[0]*np.sin(THETA[1]) + self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
                -1*np.cos(THETA[0])*(self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
                -1*np.cos(THETA[0])*self.lengths[2]*np.sin(np.sum(THETA[1:4]))
            ],
            [
                0,
                np.cos(THETA[1])*self.lengths[0] + np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2],
                np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2],
                np.cos(np.sum(THETA[1:4]))*self.lengths[2]
            ],
            [0, 1, 1, 1]
        ], dtype=float)
        return J

    def tip_wrench_at_pose_to_grip(self, THETA, F, lengths=None, frame="world"):
        '''
        takes a tip wrench in the EE frame given a pose and generates a set of
        joint torques to satisfy
        '''
        if lengths is None:
            lengths=self.lengths
        if frame=="EE":
            q = np.asarray(THETA)
            T = trans(q, lengths)
            R = T[:3,:3]
            F = R @ F
        Taus = (self.get_jacobian_at_pose(THETA, lengths).T @ F).flatten()
        Taus = clean_array(Taus)
        return Taus

    def grasp_to_grip(self, grasp: Grasp):
        Taus = np.zeros(3)
        # print(grasp.F)
        for i, f in reversed(list(enumerate(grasp.F))):
            lengths = self.lengths[:i+1]
            # print(lengths)
            # print(grasp.l)
            lengths[-1] = lengths[-1]*grasp.l[i]
            THETA = grasp.q[:i+1]
            taus = self.tip_wrench_at_pose_to_grip(THETA, grasp.F[i], lengths, frame=grasp.frame)
            # print(f"torques contributed from force at index {i}: {taus}")
            while len(taus) < self.numJoints:
                taus = np.append(taus, 0)
            Taus+=taus
        # print(f"resultant torques: {Taus}")
        return Taus

    def tensions_to_tip_wrench(self, THETA, T):
        if self.numTendons != len(T):
            warnings.warn(StructureKineMismatch(
                message=f'passed tension vector of length {len(T)}, \
                          expected {self.numTendons}'))
        if isinstance(self.structure, VariableStrucMatrix):
            Taus = self.structure.grip_from_tensions(THETA, T)
        elif isinstance(self.structure, StrucMatrix):
            Taus = self.structure().dot(T)
        print(Taus)
        wrench = jac(THETA, self.lengths) @ Taus
        wrench = clean_array(wrench)
        return wrench

    def grip_to_tensions(self, THETA, Taus):

        if isinstance(self.structure, VariableStrucMatrix):
            A = self.structure(THETA)
        elif isinstance(self.structure, StrucMatrix):
            A = self.structure()
        b = Taus
        best = np.inf
        bestRes = None
        for i in range(self.numTendons):
            # c = M[i,:] #negative sign to maximize
            c = np.zeros(self.numTendons)
            c[i] = 1
            # print("c", c)
            if isinstance(self.structure, VariableStrucMatrix):
                self.structure.controllability(THETA)
                minFactor = 1/self.structure.controllability.biasForceCondition*0.1
            elif isinstance(self.structure, StrucMatrix):
                minFactor = 1/self.structure.biasCondition()*0.1
            opt = linprog(c, A_eq=A, b_eq=b, bounds=(self.tensionLimit*minFactor, None), method='interior-point')
            # print(opt.fun, opt.x, c)
            if (opt.fun < best):
                best = opt.fun
                bestRes = opt.x
        return bestRes
        # self.structure.controllability(THETA)
        # controllability = self.structure.controllability
        # S = self.structure(THETA)
        # res = lsq_linear(S, Taus, bounds=(0,self.tensionLimit))
        # T = res.x
        # rnorm = np.linalg.norm(S @ res.x - Taus)
        # print(rnorm)
        # print("solution:", T)
        # if 0 in clean_array(T):
        #     max_biases = []
        #     min_biases = []
        #     for i in range(len(T)):
        #         if controllability.biasForceSpace[i] > 0:
        #             min_bias = (biasForce-T[i])/controllability.biasForceSpace[i]
        #             max_bias = (self.tensionLimit - T[i])/controllability.biasForceSpace[i]
        #         elif controllability.biasForceSpace[i] < 0:
        #             min_bias = (self.tensionLimit - T[i])/controllability.biasForceSpace[i]
        #             max_bias = (biasForce-T[i])/controllability.biasForceSpace[i]
        #         else:
        #             min_bias = biasForce
        #             max_bias = self.tensionLimit
        #         max_biases.append(max_bias)
        #         min_biases.append(min_bias)
        #     if np.max(min_biases) < np.min(max_biases):
        #         biasScale = np.min(max_biases)
        #     else:
        #         biasScale = 0
        #     # biasScale = np.min(max_biases)
        #     print(f"used calculated scale {biasScale}")
        #     T = T + (controllability.biasForceSpace*biasScale).flatten()
        # else:
        #     print("blind scale")
        #     T = T + (controllability.biasForceSpace/np.max(controllability.biasForceSpace)*biasForce).flatten()
        # # print("normalized bias:", controllability.biasForceSpace.flatten())
        # # print("scaled bias", (controllability.biasForceSpace/np.max(controllability.biasForceSpace)*biasForce).flatten())
        # print("solution with bias:", T)
        # confirm = S @ T
        # # print(confirm)
        # # print(Taus)
        # confirm = clean_array(confirm)
        # Taus = clean_array(Taus)
        # if  np.allclose(confirm, Taus):
        #     return T, "exact", confirm
        # else:
        #     return T, "best-case", confirm

    def get_planar_force_capability_at_pose(self, THETA, f_max=None):
        J = self.get_jacobian_at_pose_2(THETA)
        S = self.structure(THETA)
        # F0 = self.tensionLimit
        if f_max is None:
            f_max = self.tensionLimit
        F0 = np.diag([f_max]*self.numTendons)
        try:
            M = np.linalg.inv(J).T @ S @ F0
        except np.linalg.LinAlgError:
            return False
        # print(M.shape)
        num_excitations = self.numTendons
        num_constraints = 3 # For now, this works for 3 and 4 dof fingers in n+1
                            # I think
        unique_excitations = set()
        unique_forces = []
        for idx in itertools.combinations(np.arange(0,num_excitations), num_constraints):
            # integers for slicing
            # print(idx)
            idx = np.array([int(idx_i) for idx_i in idx])
            for edge in generate_binary_lists(num_constraints):
                edge = np.array(edge)
                # print(idx, edge)
                b = edge
                A = np.zeros([num_constraints,num_excitations])
                for i in range(num_constraints):
                    A[i, idx[i]]=1
                b = np.concatenate([[0]*(num_excitations-num_constraints), b])
                # THIS LINE ONLY WORKS FOR IN-PLANE FINGERS
                A = np.vstack((M[-1:], A))
                # print(A)
                c = np.zeros(len(b))
                res = linprog(c, A_eq = A, b_eq = b, bounds=(0,1))
                if not res.success:
                    pass
                else:
                    unique_excitations.add(tuple(res.x))
                # unique_forces.add(tuple(M @ res.x))
        for excitation in list(unique_excitations):
            unique_forces.append((M @ excitation)[:2])
        unique_forces = np.array(unique_forces)*4.44822162 # convert unique forces from lbs to N
        convex_forces = ConvexHull(np.array(unique_forces))
        return convex_forces

def planar_force_demo():
    # from strucMatrices import secondaryDev
    # from strucMatrices import secondaryDev
    testFinger = Finger(secondaryDev, [1.4,1.4,1.2])
    # pose = np.array([10,10,10])*np.pi/180
    angles = np.linspace(5*np.pi/180,np.pi/2,86)
    
    # plt.figure()
    fig, ax = plt.subplots()

    # --- Phase 1: Pre-calculate to find the absolute max/min bounds ---
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    for angle in angles:
        pose = [angle] * 3
        convex_hull = testFinger.get_planar_force_capability_at_pose(pose)
        if convex_hull != False:
            # Assumes convex_hull.points holds the 2D coordinates
            pts = convex_hull.points
            x_min = min(x_min, pts[:, 0].min())
            x_max = max(x_max, pts[:, 0].max()) # Replace index if coordinates are stored differently
            y_min = min(y_min, pts[:, 1].min())
            y_max = max(y_max, pts[:, 1].max())

    pause_frames = 20
    frame_sequence = ([0] * pause_frames) + list(range(len(angles))) + ([len(angles) - 1] * pause_frames)

    def update(frame):
        ax.clear()
        # Lock the axis limits to the biggest hull's boundaries so it doesn't jitter

        angle = angles[frame]
        pose = [angle]*3
        convex_hull = testFinger.get_planar_force_capability_at_pose(pose)
        if convex_hull != False:
            convex_hull_plot_2d(convex_hull, ax=ax)
            ax.set_title(f"Feasible Force Region @ theta = {np.array(pose)*180/np.pi}")
        ax.set_xlim(x_min * 1.1, x_max * 1.1) # Added 10% padding for look
        ax.set_ylim(y_min * 1.1, y_max * 1.1)
        # ax.autoscale()

    # for angle in angles:
    #     pose = [angle]*3
    #     convex_hull = testFinger.get_planar_force_capability_at_pose(pose)
    #     if convex_hull == False:
    #         continue
    #     convex_hull_plot_2d(convex_hull, ax=plt.gca())
    #     plt.title(f"Feasible Force Region @ theta = {pose}")
    #     # pose = np.array([45,10,10])*np.pi/180
    #     # convex_hull = testFinger.get_planar_force_capability_at_pose(pose)
    # plt.gca().autoscale()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_sequence,
        interval=(50),
        repeat=True
    )

    ani.save("force_capability_of_fake_finger.gif", writer="pillow", fps=10)

    plt.show()

if __name__=="__main__":
    lengths=[0,1.4, 1.4, 1.2]
    testFinger = Finger(primaryDev, lengths=lengths)
    testFinger = Finger(secondaryDev, [1.4,1.4,1.2]) # Turn this into a finger class definition?

    for name, pose in pose_dict.items():
        if name != "bah":
            font=16
            plt.figure() # forces
            plot_ffr_at_pose(pose, name) # Cuevas Anatomical Feasible Force Region
            convex_hull = testFinger.get_planar_force_capability_at_pose(pose[1:])
            convex_hull_plot_2d(convex_hull, ax=plt.gca())
    plt.show()
