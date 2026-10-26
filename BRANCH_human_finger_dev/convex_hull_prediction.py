import numpy as np
import itertools as it
import scipy.optimize as opt
import scipy.linalg as la
from scipy.spatial import ConvexHull, convex_hull_plot_2d, QhullError
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .ValeroCuervasModel import Mo_at_pos, q_flx, q_int, q_ext, R_at_pos, F_at_pos
from utils import generate_binary_lists, solve_for_feasible_excitations

# region ConvexHull (added to finger.py)

pose = q_flx
global M, R, F 
M,R,F = Mo_at_pos(pose), R_at_pos(pose), F_at_pos(pose)

num_excitations = 7
num_constraints = 5 # CONSTRAINT(1): num_constraints = 6

def get_convex_capability(self, THETA, f_max=None, constrained_rows=None, units='lbf'):
    if not isinstance(constrained_rows, (list, np.ndarray)):
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

    S = self.structure.get_S(THETA)
    M = self.get_M(THETA, f_max)

    constrained_rows = np.asarray(constrained_rows)
    num_constraints = len(constrained_rows)
    num_unconstrained = num_excitations - num_constraints
    num_sols_for_pose = 0
    unique_sol_for_pose, uniqueForces, uniqueTorques = set(), [], []

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

            A = np.vstack((M[constrained_rows,:], A))
            excitation, success = solve_for_feasible_excitations(A, b)

            if not success: pass
            else:
                num_sols_for_pose += 1
                unique_sol_for_pose.add(tuple(excitation))

    for u_ext in list(unique_sol_for_pose): # unique excitations for pose
        u_ext = np.array(u_ext)
        force = M @ u_ext # J^-T*S*Fo * e
        torques = S @ Fo @ u_ext # S*Fo * e

        # FIXME: not sure if forces and torques are indexed correctly
        uniqueForces.append(np.delete(force, constrained_rows)) # keep only the unconstrained rows
        uniqueTorques.append(torques[1:]) # FIXME: need to change index?? to what tho
    uniqueForces, uniqueTorques = np.array(uniqueForces), np.array(uniqueTorques)
    
    if units=='lbf':
        uniqueForces = uniqueForces/4.44822162 # convert N to lbf
        uniqueTorques = uniqueTorques*8.8507457916 # convert N*m to lbf-in
    hullForces = ConvexHull(uniqueForces)
    return hullForces, uniqueTorques

def plot3D_ConvexHull(hull: ConvexHull, ax=None, labels=None, title=None, color='cyan', font=16):
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    pts = hull.points
    hull_surface = Poly3DCollection(pts[hull.simplices], alpha=0.3, facecolor=color, edgecolor='k', linewidths=0.3)
    ax.add_collection3d(hull_surface)
    ax.scatter(*pts[hull.vertices].T, color='red', s=10)

    # collections don't autoscale, so set limits from the points (padded, and always including the origin)
    lo = np.minimum(pts.min(axis=0), 0); hi = np.maximum(pts.max(axis=0), 0)
    pad = 0.1*(hi - lo)
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo) # true proportions

    ax.plot([lo[0], hi[0]], [0, 0], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [lo[1], hi[1]], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [0, 0], [lo[2], hi[2]], color='black', linewidth=1)

    if labels is not None:
        ax.set_xlabel(labels[0], fontsize=font); ax.set_ylabel(labels[1], fontsize=font); ax.set_zlabel(labels[2], fontsize=font)
    if title is not None:
        ax.set_title(title, fontsize=font)
    return ax

def plot2D_ConvexHull(hull: ConvexHull, ax=None, labels=None, title=None, color='cyan', font=16):
    if ax is None:
        ax = plt.figure().add_subplot()
    pts = hull.points
    ax.fill(*pts[hull.vertices].T, alpha=0.3, facecolor=color, edgecolor='k') # 2D hull.vertices are in counterclockwise order
    ax.scatter(*pts[hull.vertices].T, color='red', s=10)
    ax.axhline(0, color='black', linewidth=1); ax.axvline(0, color='black', linewidth=1)
    ax.set_aspect('equal', adjustable='datalim')

    if labels is not None:
        ax.set_xlabel(labels[0], fontsize=font); ax.set_ylabel(labels[1], fontsize=font)
    if title is not None:
        ax.set_title(title, fontsize=font)
    return ax

# endregion

def closest_in_subspace(subspace, vector):
    projector = subspace @ np.linalg.inv(subspace.T @ subspace) @ subspace.T
    closest = projector @ vector
    return closest

def balancable_bias_force(bias):
    balancable = False
    b = np.ones((bias.shape[0],1))
    b = b / np.linalg.norm(b)
    combo = np.hstack([bias, -b])
    coeffs = la.null_space(combo)[:bias.shape[0],:]
    if coeffs.shape[1] > 0:
        balancable = True
        basis = bias @ coeffs
        intersection = np.linalg.svd(basis, full_matrices=False).U
        return balancable, 1.0
    else:
        projector = bias @ np.linalg.inv(bias.T @ bias) @ bias.T
        closest = projector @ b
        # print(closest)
        return balancable, np.max(closest)/np.min(closest)
 
if __name__ == "__main__":
    pass

