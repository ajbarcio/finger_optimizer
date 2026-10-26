import numpy as np
import warnings
from strucMatrices import colors, VariableStrucMatrix, StrucMatrix, DiscreteStrucMatrix, secondaryDev, primaryDev, valero, canonB
from utils import trans, jac, clean_array, hArray, generate_binary_lists, solve_for_feasible_excitations, compare_hulls, plot3D_ConvexHull, plot2D_ConvexHull#, get_jacobian
from scipy.optimize import nnls, lsq_linear, linprog
from scipy.spatial import ConvexHull, convex_hull_plot_2d, QhullError
import itertools as it

from matplotlib import pyplot as plt
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
        self.name = name if name is not None else getattr(structure, 'name')

    def S(self, THETA):
        """Structure matrix at THETA for any structure type (StrucMatrix is pose-independent)"""
        if isinstance(self.structure, StrucMatrix):
            return self.structure()
        return self.structure(THETA)

    def get_jacobian_at_pose(self, THETA, lengths=None):
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

    # region Anthony functions

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
    # endregion

    def get_convex_capability(self, THETA, f_max=None, constrained_dof=None, space='task'):
        """
        Feasible effort region at a pose, in task space or joint space, as a convex hull.

        Parameters
        ----------
        THETA : numpy array
            joint angles [rad]
        f_max : float, 1D numpy array, or 2D numpy array (optional)
            max tendon tension(s); scalar -> same for all tendons, 1D -> diagonal of Fo, 2D -> Fo. Defaults to self.tensionLimit
        constrained_dof : list or numpy array of int
            rows of the effort map forced to zero and removed from the output
        space : {'task', 'joint'}
            'task' -> efforts are tip wrenches (M = J^-T*S*Fo); 'joint' -> efforts are joint torques (S*Fo)

        Returns
        -------
        hull : scipy.spatial.ConvexHull
            hull of the unconstrained efforts, in lbf (task) or lbf-in (joint)
        """

        if not isinstance(constrained_dof, (list, np.ndarray)):
            raise ValueError("constrained_dof must be a list or numpy array of integers.")
        if space not in ('task', 'joint'):
            raise ValueError("space must be 'task' or 'joint'.")

        # initialize variables
        num_excitations = self.numTendons

        if f_max is None:
            f_max = self.tensionLimit
        if isinstance(f_max, (np.ndarray, list)):
            if np.asarray(f_max).ndim >1:
                Fo = f_max
            else: Fo = np.diag(f_max)
        else: Fo = np.diag([f_max]*self.numTendons)

        S = self.S(THETA)
        M = self.M(THETA, f_max)
        effort_map = M if space=='task' else S @ Fo # e -> tip wrench, or e -> joint torques

        constrained_dof = np.asarray(constrained_dof, dtype=int)
        num_constraints = len(constrained_dof)
        num_unconstrained = num_excitations - num_constraints
        num_vertex_exts_for_pose = 0
        unique_vertex_exts_for_pose, uniqueEfforts = set(), []
        for idx in it.combinations(np.arange(0,num_excitations), num_unconstrained):
            idx = np.array([int(idx_i) for idx_i in idx])
            for edge in generate_binary_lists(num_unconstrained):
                edge = np.array(edge)
                # print(edge)
                b = edge
                A = np.zeros([num_unconstrained,num_excitations])
                for i in range(num_unconstrained):
                    A[i,idx[i]]=1
                b = np.concatenate([[0]*(num_constraints), b])

                A = np.vstack((effort_map[constrained_dof,:], A))
                excitation, success = solve_for_feasible_excitations(A, b)

                if not success: pass
                else:
                    num_vertex_exts_for_pose += 1
                    unique_vertex_exts_for_pose.add(tuple(excitation))

        for u_ext in list(unique_vertex_exts_for_pose): # unique excitations for pose
            effort = effort_map @ np.array(u_ext) # J^-T*S*Fo * e (task) or S*Fo * e (joint)
            uniqueEfforts.append(np.delete(effort, constrained_dof)) # keep only the unconstrained rows
        uniqueEfforts = np.array(uniqueEfforts)

        if self.units=='N':
            uniqueEfforts = uniqueEfforts/4.44822162 if space=='task' else uniqueEfforts*8.8507457916 # N -> lbf, N*m -> lbf-in

        try: return ConvexHull(uniqueEfforts)
        except QhullError: return ConvexHull(uniqueEfforts, qhull_options='QJ')

    def plot_Capability(self, convex_hull: ConvexHull, ax=None, title=None, labels=None, label=None, color='cyan'):
        """
        Helper function to simplify task and joint capability plotting (utilizes plot2D and 3D convex hull)

        Parameters
        ----------
        convex_hull : scipy.spatial.ConvexHull
            2D or 3D hull to plot
        ax : matplotlib axes (optional)
            axes to draw on; new figure if None
        title : str (optional)
            axes title
        labels : list of str (optional)
            axis labels
        label : str (optional)
            legend entry
        color : str
            matplotlib color

        Returns
        -------
        ax : matplotlib axes
            axes the hull was drawn on
        """
        _, num_efforts = (convex_hull.points).shape
        if num_efforts not in (2,3):
            raise ValueError(f'Hull values must only have two or three effort values. {num_efforts}) were inputted.')
        elif num_efforts==3:
            ax = plot3D_ConvexHull(convex_hull, ax, labels, title, label, color)
        else: ax = plot2D_ConvexHull(convex_hull, ax, labels, title, label, color)
        ax.figure.tight_layout(); ax.grid(True, alpha=0.8) # general formatting for ALL plots
        return ax
    
    def plot_taskCapability(self, THETA, ax=None, f_max=None, constrained_dof=None, title=None, labels=None, label=None, color='cyan'):
        convex_hull = self.get_convex_capability(THETA, f_max, constrained_dof, space='task')
        return self.plot_Capability(convex_hull, ax, title, labels, label, color)
    def plot_jointCapability(self, THETA, ax=None, f_max=None, constrained_dof=None, title=None, labels=None, label=None, color='cyan'):
        convex_hull = self.get_convex_capability(THETA, f_max, constrained_dof, space='joint')
        return self.plot_Capability(convex_hull, ax, title, labels, label, color)

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

def compare_capabilities(cases, title=None, task_labels=None, joint_labels=None):
    """
    Plot the task (left) and joint (right) capabilities of several fingers on one figure.

    Parameters
    ----------
    cases : list of dict
        one per finger: finger, THETA, task_dof, joint_dof, f_max (optional), color (optional)
    title : str (optional)
        figure title
    task_labels : list of str (optional)
        task plot axis labels
    joint_labels : list of str (optional)
        joint plot axis labels

    Returns
    -------
    axTask, axJoint : matplotlib axes
        task and joint plot axes
    """
    n, first = cases[0]['finger'].numJoints, cases[0]
    fig = plt.figure(figsize=(14, 6))
    axTask  = fig.add_subplot(1, 2, 1, projection='3d' if n-len(first['task_dof'])==3 else None) # 3 unconstrained efforts -> 3D, 2 -> 2D
    axJoint = fig.add_subplot(1, 2, 2, projection='3d' if n-len(first['joint_dof'])==3 else None)

    for i, case in enumerate(cases):
        color = case.get('color', colors[i % len(colors)]) # optional per-case override, else cycle (wraps if more fingers than colors)
        finger, THETA, f_max = case['finger'], case['THETA'], case.get('f_max')
        finger.plot_taskCapability(THETA, axTask, f_max, case['task_dof'], 'Feasible Task Capabilities', task_labels, finger.name, color)
        finger.plot_jointCapability(THETA, axJoint, f_max, case['joint_dof'], 'Feasible Joint Capabilities', joint_labels, finger.name, color)

    axTask.legend(); axJoint.legend()
    if title is not None:
        fig.suptitle(title, fontsize=18); fig.subplots_adjust(top=0.85) # leave room for the suptitle
    return axTask, axJoint

# region Finger instances
valeroFinger    = Finger(valero, [0, 50e-3, 31e-3, 16e-3], units='N', name='Valero (discrete)')
# valeroStatic    = Finger(StrucMatrix(S=valero.S(list(valero.poses.values())[0])), valeroFinger.lengths, units='N', name='Valero (static)')
primaryFinger   = Finger(primaryDev,   [0, 1.4, 1.4, 1.2], name='primaryDev (variable)') # 4DOF
secondaryFinger = Finger(secondaryDev, [1.4, 1.4, 1.2],    name='secondaryDev (variable)') # 3DOF
constantFinger =  Finger(canonB, [1.4, 1.4, 1.2], name='canonB (static)')
# endregion

if __name__=="__main__":
    # define compare_capabilities parameters
    dof4, dof3 = dict(task_dof=[0,3], joint_dof=[0]), dict(task_dof=[2], joint_dof=[])

    THETA = np.radians(np.array([0,45,45,10])) # Flexion pose
    title = title='Flexion Pose'; task_labels=['$F_y$ (lbf)', '$F_z$ (lbf)']
    joint_labels = joint_labels=[r'$\tau_{1}$ (lbf-in)', r'$\tau_{2}$ (lbf-in)', r'$\tau_{3}$ (lbf-in)']

    # define cases for comparison
    cases = [
        dict(finger=valeroFinger,    THETA=THETA,     f_max=valero.Fo(THETA), **dof4, color=colors[1]),
        dict(finger=primaryFinger,   THETA=THETA,                             **dof4, color=colors[2]),
        dict(finger=secondaryFinger, THETA=THETA[1:],                         **dof3, color=colors[7]), # 3DOF: drop ad/ab angle
        dict(finger=constantFinger,  THETA=THETA[1:],                         **dof3, color=colors[0])
    ]
    compare_capabilities(cases, title=title, task_labels=task_labels, joint_labels=joint_labels)

    #-------------------- Valero Finger Single Plots (task and joint space) --------------------
    # valeroFinger.plot_taskCapability(THETA, constrained_dof=[0,3], f_max=Fo, 
    #                                  title='Feasible Joint Capabilities for Flexion', labels=['F_y', 'F_z']) # constrain F_x and τ_x
    # valeroFinger.plot_jointCapability(THETA, constrained_dof=[0], f_max=Fo, 
    #                                   title='Feasible Joint Capabilities for Flexion', labels=r'$\tau_{1}$') # only constrain F_x
    plt.show()
