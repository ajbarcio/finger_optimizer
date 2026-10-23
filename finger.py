import numpy as np
import warnings
from strucMatrices import VariableStrucMatrix, StrucMatrix, DiscreteStrucMatrix, secondaryDev, primaryDev, valero
from utils import trans, jac, clean_array, hArray, generate_binary_lists, solve_for_feasible_excitations#, plot3D_ConvexHull, plot2D_ConvexHull#, get_jacobian
from scipy.optimize import nnls, lsq_linear, linprog
from scipy.spatial import ConvexHull, convex_hull_plot_2d, QhullError
import itertools as it

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
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

    def __init__(self, structure: VariableStrucMatrix | StrucMatrix | DiscreteStrucMatrix, lengths, tensionLimit=50, units='lbf', name=None):
        self.structure = structure
        self.lengths = lengths
        if self.structure.numJoints != len(self.lengths):
            warnings.warn(StructureKineMismatch())

        self.numJoints = self.structure.numJoints
        self.numTendons = self.structure.numTendons
        # print(self.numJoints, self.numTendons)
        self.tensionLimit = tensionLimit

        if units not in ('lbf', 'N'):
            raise ValueError("units must be 'lbf' or 'N'.")
        self.units = units
        self.name = name if name is not None else getattr(structure, 'name', 'Finger') # FIXME: don't entirely understand what i did here or if its necessary

    def S(self, THETA):
        """Structure matrix at THETA for any structure type (StrucMatrix is pose-independent)"""
        if isinstance(self.structure, StrucMatrix):
            return self.structure()
        return self.structure(THETA)

    def get_jacobian_at_pose(self, THETA, lengths=None):
        # F = trans(THETA, self.lengths)
        if lengths is None:
            lengths=self.lengths
        J = jac(THETA, lengths)
        return J

    def M(self, THETA, f_max=None):
        if f_max is None:
            f_max = self.tensionLimit
        if isinstance(f_max, (np.ndarray, list)):
            if np.asarray(f_max).ndim >1:
                Fo = f_max
            else: Fo = np.diag(f_max)
        else: Fo = np.diag([f_max]*self.numTendons)

        J = self.get_jacobian_at_pose(THETA)
        S = self.S(THETA)

        try: return np.linalg.inv(J).T @ S @ Fo
        except np.linalg.LinAlgError: raise ValueError(f"{self.name}: Jacobian is singular at THETA={THETA}")

    # def get_jacobian_at_pose(self, THETA, lengths=None):
    #     if lengths is None:
    #         lengths=self.lengths
    #     J = get_jacobian(THETA, lengths)
    #     return J

    # region Jacobians
    # def get_jacobian_at_pose_2(self, THETA, lengths=None):
    #     if lengths==None:
    #         lengths=self.lengths
    #     ## This is a test 32x3 array for a 3-joint finger that is only being
    #     ## used to test get_jacobian_at_pose to make
    #     ## sure the frame used here matches the frame in the paper
    #     J = np.array([
    #         [-(lengths[0]*np.sin(THETA[0])+lengths[1]*np.sin(np.sum(THETA[0:2]))+lengths[2]*np.sin(np.sum(THETA))),
    #          -(lengths[1]*np.sin(np.sum(THETA[0:2]))+lengths[2]*np.sin(np.sum(THETA))),
    #           -lengths[2]*np.sin(np.sum(THETA))],
    #         [  lengths[0]*np.cos(THETA[0])+lengths[1]*np.cos(np.sum(THETA[0:2]))+lengths[2]*np.cos(np.sum(THETA)),
    #            lengths[1]*np.cos(np.sum(THETA[0:2]))+lengths[2]*np.cos(np.sum(THETA)),
    #            lengths[2]*np.cos(np.sum(THETA))],
    #         [  1,
    #            1,
    #            1]])
    #     return J
    #     # -
    # def get_jacobian_at_pose_3(self, THETA, lengths=None):
    #     if lengths==None:
    #         lengths=self.lengths
    #     J = np.array([
    #         [
    #             -1*np.cos(THETA[0])*(np.cos(THETA[1])*self.lengths[0] + np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2]),
    #             np.sin(THETA[0])*(self.lengths[0]*np.sin(THETA[1]) + self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
    #             np.sin(THETA[0])*(self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
    #             self.lengths[2]*np.sin(THETA[0])*np.sin(np.sum(THETA[1:4]))
    #         ],
    #         [
    #             -1*(np.cos(THETA[1])*self.lengths[0] + np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2])*np.sin(THETA[0]),
    #             -1*np.cos(THETA[0])*(self.lengths[0]*np.sin(THETA[1]) + self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
    #             -1*np.cos(THETA[0])*(self.lengths[1]*np.sin(np.sum(THETA[1:3])) + self.lengths[2]*np.sin(np.sum(THETA[1:4]))),
    #             -1*np.cos(THETA[0])*self.lengths[2]*np.sin(np.sum(THETA[1:4]))
    #         ],
    #         [
    #             0,
    #             np.cos(THETA[1])*self.lengths[0] + np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2],
    #             np.cos(np.sum(THETA[1:3]))*self.lengths[1] + np.cos(np.sum(THETA[1:4]))*self.lengths[2],
    #             np.cos(np.sum(THETA[1:4]))*self.lengths[2]
    #         ],
    #         [0, 1, 1, 1]
    #     ], dtype=float)
    #     return J
    # endregion

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
        # Fo = self.tensionLimit
        if f_max is None:
            f_max = self.tensionLimit
        Fo = np.diag([f_max]*self.numTendons)
        try:
            M = np.linalg.inv(J).T @ S @ Fo
        except np.linalg.LinAlgError:
            return False
        # print(M.shape)
        num_excitations = self.numTendons
        num_constraints = 3 # For now, this works for 3 and 4 dof fingers in n+1
                            # I think
        unique_excitations = set()
        unique_forces = []
        for idx in it.combinations(np.arange(0,num_excitations), num_constraints):
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
        if self.units=='N':
            unique_forces = np.array(unique_forces)/4.44822162 # convert unique forces from N to lbf
        convex_forces = ConvexHull(np.array(unique_forces))
        return convex_forces

    def get_convex_capability(self, THETA, f_max=None, constrained_dof=None):
        """
        FIXME: define what function does
        
        Parameters
        ----------
        variable : variable type
            variable description

        Returns
        -------
        convexForces : numpy array
            Data structure containing the parameters of the rover with updated telemetry data after simulation
        uniqueTorques : numpy array #FIXME out of date.
        """
        
        if not isinstance(constrained_dof, (list, np.ndarray)):
            raise ValueError("constrained_rows must be a list or numpy array of integers.")

        # initialize variables
        num_excitations = self.numTendons

        if f_max is None:
            f_max = self.tensionLimit
        if isinstance(f_max, (np.ndarray,list)): # FIXME: convert to where()?
            if (f_max).ndim>1:
                Fo = f_max  # FIXME: does structmatrix or finger.py need the % change of F?
            else: Fo = np.diag(f_max)
        else: Fo = np.diag([f_max]*self.numTendons)

        S = self.S(THETA)
        M = self.M(THETA, f_max)

        constrained_dof = np.asarray(constrained_dof, dtype=int)
        num_constraints = len(constrained_dof)
        num_unconstrained = num_excitations - num_constraints
        num_vertex_exts_for_pose = 0
        unique_vertex_exts_for_pose, uniqueTaskEfforts, uniqueJointEfforts = set(), [], []

        for idx in it.combinations(np.arange(0,num_excitations), num_unconstrained):
            idx = np.array([int(idx_i) for idx_i in idx])
            for edge in generate_binary_lists(num_unconstrained):
                edge = np.array(edge)
                # print(edge)
                b = edge
                A = np.zeros([num_unconstrained,num_excitations])
                for i in range(num_unconstrained):
                    A[i,idx[i]]=1 # NOTE: what this do
                b = np.concatenate([[0]*(num_constraints), b])

                A = np.vstack((M[constrained_dof,:], A))
                excitation, success = solve_for_feasible_excitations(A, b)

                if not success: pass
                else:
                    num_vertex_exts_for_pose += 1
                    unique_vertex_exts_for_pose.add(tuple(excitation))

        for u_ext in list(unique_vertex_exts_for_pose): # unique excitations for pose
            u_ext = np.array(u_ext)
            taskEffort = M @ u_ext # J^-T*S*Fo * e
            jointEffort = S @ Fo @ u_ext # S*Fo * e

            # FIXME: not sure if forces and torques are indexed correctly
            uniqueTaskEfforts.append(np.delete(taskEffort, constrained_dof)) # keep only the unconstrained rows
            uniqueJointEfforts.append(np.delete(jointEffort, constrained_dof)) # FIXME: need to change index?? to what tho (currently only removes out of plane torques)
        uniqueTaskEfforts, uniqueJointEfforts = np.array(uniqueTaskEfforts), np.array(uniqueJointEfforts)
        
        if self.units=='N':
            uniqueTaskEfforts = uniqueTaskEfforts/4.44822162 # convert N to lbf
            uniqueJointEfforts = uniqueJointEfforts*8.8507457916 # convert N*m to lbf-in

        def make_hull(pts):
            try: return ConvexHull(np.array(pts))
            except QhullError: return ConvexHull(np.array(pts), qhull_options='QJ')

        hullForces, hullTorques = (make_hull(pts) for pts in (uniqueTaskEfforts, uniqueJointEfforts))
        return hullForces, hullTorques

    def plot2D_ConvexHull(self, hull : ConvexHull, ax=None, ax_labels=None, title=None, label=None, color='cyan', font=16):
        # print('Plotting 2D Convex Hull') # FIXME bug check
        if ax is None:
            ax = plt.figure().add_subplot()

        pts = hull.points
        ax.scatter(*pts.T, color=color, s=10, label=label) # all points (incl. interior), label = legend entry

        hull_surface = PolyCollection([pts[hull.vertices]], facecolor=(color, 0.2), edgecolor=color, linewidths=1.5) # 2D vertices are CCW-ordered
        ax.add_collection(hull_surface)

        if ax_labels is not None:
            ax.set_xlabel(ax_labels[0], fontsize=font); ax.set_ylabel(ax_labels[1], fontsize=font)
        if title is not None:
            ax.set_title(title, fontsize=font)

        # formatting stuff (draw "axes" through origin, etc.)
        ax.axhline(0, color='black', linewidth=1); ax.axvline(0, color='black', linewidth=1)
        return ax
    
    def plot3D_ConvexHull(self, hull : ConvexHull, ax=None, ax_labels=None, title=None, label=None, color='cyan', font=16):
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')

        pts = hull.points
        ax.scatter(*pts[hull.vertices].T, color=color, s=10, label=label) # label = legend entry; scatter autoscales the axes (Poly3DCollection doesn't)

        vertices = [hull.points[simplex] for simplex in hull.simplices]
        hull_surface = Poly3DCollection(vertices, facecolor=(color, 0.2), edgecolor=color, linewidths=0.3)
        ax.add_collection3d(hull_surface)

        if ax_labels is not None:
            ax.set_xlabel(ax_labels[0], fontsize=font); ax.set_ylabel(ax_labels[1], fontsize=font); ax.set_zlabel(ax_labels[2], fontsize=font)
        if title is not None:
            ax.set_title(title, fontsize=font)

        # formatting stuff (draw "axes" through origin, etc.), redrawn each call to span every hull on this ax
        # add_artist (not ax.plot) so these lines don't count as data and push the limits out
        for line in getattr(ax, '_origin_lines', []): line.remove()
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        ax._origin_lines = [ax.add_artist(Line3D(xlim, [0, 0], [0, 0], color='black', linewidth=1)),
                            ax.add_artist(Line3D([0, 0], ylim, [0, 0], color='black', linewidth=1)),
                            ax.add_artist(Line3D([0, 0], [0, 0], zlim, color='black', linewidth=1))]
        return ax

    def plot_Capability(self, convex_hull: ConvexHull, THETA, ax=None, f_max=None, constrained_dof=None, title=None, labels=None):
        """
        helper function to simplify task and joint capability plotting (utilizes plot2D and 3D convex hull)
        """
        _, num_efforts = (convex_hull.points).shape
        if num_efforts not in (2,3):
            raise ValueError(f'Hull values must only have two or three effort values. {num_efforts}) were inputted.')
        elif num_efforts==3:
            ax = self.plot3D_ConvexHull(convex_hull, ax, labels, title)
        else: ax = self.plot2D_ConvexHull(convex_hull, ax, labels, title)
        ax.figure.tight_layout(); ax.grid(True, alpha=0.8) # general formatting for ALL plots
        return ax
    
    def plot_taskCapability(self, THETA, ax=None, f_max=None, constrained_dof=None, title=None, labels=None):
        convex_hull = self.get_convex_capability(THETA, f_max, constrained_dof)[0]
        return self.plot_Capability(convex_hull, THETA, ax, f_max, constrained_dof, title, labels)
    def plot_jointCapability(self, THETA, ax=None, f_max=None, constrained_dof=None, title=None, labels=None):
        convex_hull = self.get_convex_capability(THETA, f_max, constrained_dof)[1]
        return self.plot_Capability(convex_hull, THETA, ax, f_max, constrained_dof, title, labels)

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

def compare_capabilities(cases, title=None):
    if title is not None:
        fig.suptitle(title, fontsize=10)
# region Finger Instances
# testbed
# valero
# endregion

if __name__=="__main__":
    np.set_printoptions(precision=10, formatter={'float_kind':'{:.5f}'.format})
    lengths=[0,1.4, 1.4, 1.2]
    testFinger = Finger(primaryDev, lengths=lengths) # 4DOF
    testFinger = Finger(secondaryDev, lengths=lengths[1:]) # 3DOF

    valeroFinger, THETA = Finger(valero, [0, 50e-3, 31e-3, 16e-3 ], units='N', name='Valero (discrete)'), np.radians(np.array([0,45,45,10])) #4DOF (metric units)
    valero_model_dict = valero.model
    J = valeroFinger.get_jacobian_at_pose(THETA)
    Fo = valero.Fo(THETA)

    valeroFinger.plot_taskCapability(THETA, constrained_dof=[0,3], f_max=Fo, 
                                     title='Feasbile Joint Capabilities for Flexion', labels=['F_y', 'F_z']) # constrain F_x and τ_x
    valeroFinger.plot_jointCapability(THETA, constrained_dof=[0], f_max=Fo, 
                                      title='Feasbile Joint Capabilities for Flexion', labels=r'$\tau_{1}$') # only constrain F_x
    plt.show()
