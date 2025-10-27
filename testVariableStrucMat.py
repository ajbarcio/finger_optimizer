import numpy as np
from matplotlib import pyplot as plt
from strucMatrices import VariableStrucMatrix
from grasps import generateGrasps

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import proj3d


D = np.array([[-1,1,1,1],
              [0,-1,1,1],
              [0,0,-1,1]])
c = 1
r = 1.9/2/25.4*4 # 4 times tendon radius (in inches)
r_c = 0.5

R = np.array([[r,np.nan,np.nan,np.nan],
              [0,r     ,np.nan,np.nan],
              [0,0     ,r     ,np.nan]])

inherentVariable = VariableStrucMatrix(R, D, ranges = [(c*np.sqrt(2)/2-r_c, c-r_c)],
                                             types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
                                             F = np.array([50]*4),
                                             name='Inherently Controllable w/ Variable Effort')

D = np.array([[1,1,1,-1],
              [1,1,1,-1],
              [1,1,1,-1]])
r = .1625
R = np.array([[np.nan,r     ,r     ,r],
              [r     ,np.nan,r     ,r],
              [r     ,r     ,np.nan,r]])

r_1 = .261281
r_2 = .190271
r_3 = .307475

c_1 = .42378
c_2 = .35277
c_3 = .46997

skinnyLegend = VariableStrucMatrix(R, D, ranges=[(c_1*np.sqrt(2)/2-r_1,c_1-r_1),
                                                 (c_2*np.sqrt(2)/2-r_2,c_2-r_2),
                                                 (c_3*np.sqrt(2)/2-r_3,c_3-r_3),],
                                         types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
                                         F = np.array([50]*4),
                                         name='Skinny Legend')
c = 1
r = 1.9/2/25.4*4 # 4 times tendon radius (in inches)
r_c = 0.5

R = np.array([[np.nan,np.nan,np.nan,r],
              [0     ,np.nan,np.nan,r],
              [0,     0     ,np.nan,r]])
D = np.array([[1,1,1,-1],
              [0,1,1,-1],
              [0,0,1,-1]])
# R = 
# R = np.array([[.203125,.203125,.203125,.171875],
#               [0      ,np.nan ,np.nan ,.125   ],
#               [0      ,0      ,np.nan ,.101103]])
# c1 = .9541575
# c2 = .9505297
# r = .5625
# conceptualAmbrose = VariableStrucMatrix(R, D, ranges=[(c1*np.sqrt(2)/2-r,c1-r)]*2+
#                                                       [(c2*np.sqrt(2)/2-r,c2-r)],
#                                                types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
#                                                F = np.array([50]*4),
#                                                name='The Ambrose')

conceptualAmbrose = VariableStrucMatrix(R, D, ranges = [(c*np.sqrt(2)/2-r_c, c-r_c)],
                                              types=[VariableStrucMatrix.convergent_circles_joint]*np.sum(np.isnan(R)),
                                               F = np.array([50]*4),
                                               name='The Ambrose')

for effortFunction in conceptualAmbrose.effortFunctions:
    print(effortFunction(0))
    print(effortFunction.min)
    print(effortFunction(np.pi/2))
    print(effortFunction.max)
    print("--")

print('---------------------------------------')
for effortFunction in skinnyLegend.effortFunctions:
    print(effortFunction(0))
    print(effortFunction.min)
    print(effortFunction(np.pi/2))
    print(effortFunction.max)
    print("--")

print('---------------------------------------')
for effortFunction in inherentVariable.effortFunctions:
    print(effortFunction(0))
    print(effortFunction.min)
    print(effortFunction(np.pi/2))
    print(effortFunction.max)
    print("--")

# print(skinnyLegend)
# print(dimensionalAmbrose)

# print(skinnyLegend.constraints)
# print(dimensionalAmbrose.constraints)
# # add a grasp at full extension:
# skinnyLegend.add_grasp([0,0,0], [1,0.5,0.25], type='ineq')
# # add a grasp at full flexion:
# skinnyLegend.add_grasp([np.pi/2,np.pi/2,np.pi/2], [1,0.5,0.25], type='ineq')

print(skinnyLegend.constraints)
print(conceptualAmbrose.constraints)
print(inherentVariable.constraints)
# add human-like boundary grasps:

# l = np.array([1.375,1.4375,1.23])
# for grasp in generateGrasps(l):
#     skinnyLegend.add_grasp(grasp[0], grasp[1], type='ineq')
#     dimensionalAmbrose.add_grasp(grasp[0], grasp[1], type='ineq')

# add a grasp at full extension:
# skinnyLegend.add_grasp([0,0,0], [1,0.5,0.25], type='ineq')
# theAmbrose.add_grasp([0,0,0], [1,0.5,0.25], type='ineq')
# # add a grasp at full flexion:
# skinnyLegend.add_grasp([np.pi/2,np.pi/2,np.pi/2], [1,0.5,0.25], type='ineq')
# theAmbrose.add_grasp([np.pi/2,np.pi/2,np.pi/2], [1,0.5,0.25], type='ineq')

# rtestSL = skinnyLegend.flatten_r_matrix()

# for constraintContainer in skinnyLegend.constraints:
#     result = constraintContainer.constraint.fun(rtestSL)
#     print("Constraint value at x_test:", result)
#     print("Lower bound:", constraintContainer.constraint.lb, "Upper bound:", constraintContainer.constraint.ub)

# print(skinnyLegend.ffNorm(np.inf))
# newr = skinnyLegend.flatten_r_matrix()/2
# print(newr)
# skinnyLegend.reinit(newr)
# print(skinnyLegend.ffNorm(np.inf))

# Test that the NonlinearConstraints work properly


# print(theAmbrose.S([0]*theAmbrose.numJoints))
# print(theAmbrose.S([np.pi/2]*theAmbrose.numJoints))

# # # print(np.array(test.variablePulleys))
# # # print(test.j0t0r(0))
# # print(skinnyLegend.extS)
# # # print(skinnyLegend([np.pi/2,np.pi/2,np.pi/2]))
# # print(skinnyLegend.flatten_r_matrix())
# # print(skinnyLegend.reinit(skinnyLegend.flatten_r_matrix()/2))

# # print(skinnyLegend.extS)
print("--------------------------------------------------------------------------------------")
print(f"Skinny Legend overall valid at THETA={[0]*skinnyLegend.numJoints}? {skinnyLegend.controllability([0]*skinnyLegend.numJoints)}")
print(skinnyLegend([0]*skinnyLegend.numJoints))
print(f"rank valid? {skinnyLegend.controllability.rankCriterion}")
print(f"null space valid? {skinnyLegend.controllability.nullSpaceCriterion}, with condition {skinnyLegend.controllability.biasForceCondition}")
print(f"bias force direction: {skinnyLegend.controllability.biasForceSpace.T}")
print(f"Skinny Legend overall valid at THETA={[np.pi/2]*skinnyLegend.numJoints}? {skinnyLegend.controllability([np.pi/2]*skinnyLegend.numJoints)}")
print(skinnyLegend([np.pi/2]*skinnyLegend.numJoints))
print(f"rank valid? {skinnyLegend.controllability.rankCriterion}")
print(f"null space valid? {skinnyLegend.controllability.nullSpaceCriterion}, with condition {skinnyLegend.controllability.biasForceCondition}")
print(f"bias force direction: {skinnyLegend.controllability.biasForceSpace.T}")
print("--------------------------------------------------------------------------------------")
print(f"Ambrose overall valid at THETA={[0]*conceptualAmbrose.numJoints}? {conceptualAmbrose.controllability([0]*conceptualAmbrose.numJoints)}")
print(conceptualAmbrose([0]*conceptualAmbrose.numJoints))
print(f"rank valid? {conceptualAmbrose.controllability.rankCriterion}")
print(f"null space valid? {conceptualAmbrose.controllability.nullSpaceCriterion}, with condition {conceptualAmbrose.controllability.biasForceCondition}")
print(f"bias force direction: {conceptualAmbrose.controllability.biasForceSpace.T}")
print(f"Ambrose overall valid at THETA={[np.pi/2]*conceptualAmbrose.numJoints}? {conceptualAmbrose.controllability([np.pi/2]*conceptualAmbrose.numJoints)}")
print(conceptualAmbrose([np.pi/2]*conceptualAmbrose.numJoints))
print(f"rank valid? {conceptualAmbrose.controllability.rankCriterion}")
print(f"null space valid? {conceptualAmbrose.controllability.nullSpaceCriterion}, with condition {conceptualAmbrose.controllability.biasForceCondition}")
print(f"bias force direction: {conceptualAmbrose.controllability.biasForceSpace.T}")
print("--------------------------------------------------------------------------------------")
print(f"Inherent overall valid at THETA={[0]*inherentVariable.numJoints}? {inherentVariable.controllability([0]*inherentVariable.numJoints)}")
print(inherentVariable([0]*inherentVariable.numJoints))
print(f"rank valid? {inherentVariable.controllability.rankCriterion}")
print(f"null space valid? {inherentVariable.controllability.nullSpaceCriterion}, with condition {inherentVariable.controllability.biasForceCondition}")
print(f"bias force direction: {inherentVariable.controllability.biasForceSpace.T}")
print(f"Inherent overall valid at THETA={[np.pi/2]*inherentVariable.numJoints}? {inherentVariable.controllability([np.pi/2]*inherentVariable.numJoints)}")
print(inherentVariable([np.pi/2]*inherentVariable.numJoints))
print(f"rank valid? {inherentVariable.controllability.rankCriterion}")
print(f"null space valid? {inherentVariable.controllability.nullSpaceCriterion}, with condition {inherentVariable.controllability.biasForceCondition}")
print(f"bias force direction: {inherentVariable.controllability.biasForceSpace.T}")

skinnyLegend.plotCapability([0]*skinnyLegend.numJoints)
skinnyLegend.plotCapability([np.pi/2]*skinnyLegend.numJoints)
# skinnyLegend.plotCapabilityAcrossAllGrasps()
conceptualAmbrose.plotCapability([0]*conceptualAmbrose.numJoints)
conceptualAmbrose.plotCapability([np.pi/2]*conceptualAmbrose.numJoints)
# dimensionalAmbrose.plotCapabilityAcrossAllGrasps()
inherentVariable.plotCapability([0]*conceptualAmbrose.numJoints)
inherentVariable.plotCapability([np.pi/2]*conceptualAmbrose.numJoints)
# inherentVariable.plotCapabilityAcrossAllGrasps()

jointAngles = np.linspace(0,np.pi/2,40)
ABObjectives = np.zeros((len(jointAngles),len(jointAngles),len(jointAngles)))
SLObjectives = np.zeros((len(jointAngles),len(jointAngles),len(jointAngles)))
IVObjectives = np.zeros((len(jointAngles),len(jointAngles),len(jointAngles)))
ABValids = np.zeros((len(jointAngles),len(jointAngles),len(jointAngles)), dtype=bool)
SLValids = np.zeros((len(jointAngles),len(jointAngles),len(jointAngles)), dtype=bool)
IVValids = np.zeros((len(jointAngles),len(jointAngles),len(jointAngles)), dtype=bool)

n = len(jointAngles)
l = 0
for i in range(len(jointAngles)):
    for j in range(len(jointAngles)):
        for k in range(len(jointAngles)):
            q = [jointAngles[i],jointAngles[j],jointAngles[k]]
            # print(q)
            ABValids[i,j,k] = conceptualAmbrose.controllability(q)
            SLValids[i,j,k] = skinnyLegend.controllability(q)
            IVValids[i,j,k] = skinnyLegend.controllability(q)
            ABObjectives[i,j,k] = conceptualAmbrose.controllability.biasForceCondition
            SLObjectives[i,j,k] = skinnyLegend.controllability.biasForceCondition
            IVObjectives[i,j,k] = skinnyLegend.controllability.biasForceCondition
            # ABObjectives[i,j,k] = dimensionalAmbrose.maxGrip(q)
            # SLObjectives[i,j,k] = skinnyLegend.maxGrip(q)
            if l == n-1 or not l%100:
                print(f"{l/n**3*100:3.2f}% done (calculating controllability at each position ...)", end = '\r')
            l+=1
def plot_conditions_depth_transparent(
        Conditions, Valids, jointAngles,
        title="Conditions (depth-transparent)",
        s=6,
        near_alpha=0.15,   # alpha for points nearest to camera (more transparent)
        far_alpha=1.0,     # alpha for points farthest from camera (more opaque)
        inf_multiplier=1000,
        mode="power",      # "power", "exp", or "sigmoid"
        gamma=0.3,         # shape parameter for the mapping
        auto_flip_depth=True  # Automatically flip normalized depth so far->opaque
    ):
    # Validate args (catch the 'passed the title instead of jointAngles' mistake)
    if not isinstance(jointAngles, (list, np.ndarray)):
        raise TypeError("jointAngles must be an array-like of numeric joint values. "
                        "Call: plot_conditions_depth_transparent(Conditions, Valids, jointAngles, title='...')")

    # Build point arrays
    n = len(jointAngles)
    xs, ys, zs, cs = [], [], [], []
    for ii in range(n):
        for jj in range(n):
            for kk in range(n):
                if Valids[ii, jj, kk]:
                    xs.append(jointAngles[ii])
                    ys.append(jointAngles[jj])
                    zs.append(jointAngles[kk])
                    cs.append(Conditions[ii, jj, kk])

    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs); cs = np.array(cs)
    if xs.size == 0:
        raise ValueError("No valid points to plot (Valids is all False).")

    # Handle infinite condition values for the log colormap
    finite_vals = cs[np.isfinite(cs)]
    vmax = finite_vals.max() * inf_multiplier if finite_vals.size > 0 else 1e6
    # vmax = finite_vals.max() if finite_vals.size > 0 else 1e6
    vmin = finite_vals.min() if finite_vals.size > 0 else 1
    cs_plot = np.where(np.isinf(cs), vmax, cs)

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["viridis_r"]  # modern API

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(xs, ys, zs, c=cs_plot, cmap=cmap, norm=norm, s=s)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="Bias Force Condition (log scale)")

    ax.set_xlabel("Joint 1"); ax.set_ylabel("Joint 2"); ax.set_zlabel("Joint 3")
    ax.set_title(title)

    base_colors = cmap(norm(cs_plot))

    def depth_to_alpha(dnorm):
        """Map normalized depth (0=nearest, 1=farthest) to alpha using chosen mode."""
        if mode == "power":
            dscaled = np.power(dnorm, gamma)
        elif mode == "exp":
            k = max(1e-6, gamma)
            dscaled = 1 - np.exp(-k * dnorm)
            dscaled = (dscaled - dscaled.min()) / (dscaled.max() - dscaled.min() + 1e-12)
        elif mode == "sigmoid":
            k = max(1e-6, gamma * 12)
            dscaled = 1.0 / (1.0 + np.exp(-k * (dnorm - 0.5)))
            dscaled = (dscaled - dscaled.min()) / (dscaled.max() - dscaled.min() + 1e-12)
        else:
            dscaled = dnorm
        return near_alpha + (far_alpha - near_alpha) * dscaled

    def update_alpha(event=None):
        """
        Compute a depth value for each point using the axes projection,
        normalize it, optionally flip polarity, then map to alpha.
        """
        # Use mpl projection to get consistent display depths
        _, _, proj_z = proj3d.proj_transform(xs, ys, zs, ax.get_proj())
        depths = np.array(proj_z)

        dmin, dmax = depths.min(), depths.max()
        denom = (dmax - dmin) if (dmax - dmin) != 0 else 1.0
        dnorm = (depths - dmin) / denom  # 0..1 but polarity depends on projection convention

        # Flip so that dnorm==0 => nearest and dnorm==1 => farthest, if requested.
        # (Different Matplotlib versions / view setups sometimes invert this.)
        if auto_flip_depth:
            # Heuristic: prefer mapping where far points are more opaque.
            # If current dnorm already gives far more opaque when used, don't flip.
            # We'll test by mapping to alphas both ways and pick the one where the median depth
            # yields expected relation (this is cheap).
            alphas_direct = depth_to_alpha(dnorm)
            alphas_flipped = depth_to_alpha(1.0 - dnorm)

            # Heuristic to choose which orientation makes the far-most points have larger alpha:
            # Compare mean alpha of the top-10% depths for direct vs flipped.
            threshold = 0.9
            mask_far = dnorm >= threshold
            if mask_far.sum() > 0:
                mean_direct_far = np.mean(alphas_direct[mask_far])
                mean_flipped_far = np.mean(alphas_flipped[mask_far])
                # Choose orientation where mean alpha of far points is larger
                use_flipped = (mean_flipped_far > mean_direct_far)
            else:
                use_flipped = True  # fallback

            chosen_alphas = alphas_flipped if use_flipped else alphas_direct
            alphas = chosen_alphas
        else:
            alphas = depth_to_alpha(dnorm)

        colors = base_colors.copy()
        colors[:, -1] = alphas
        sc.set_facecolors(colors)
        try:
            sc.set_edgecolors(colors)
        except Exception:
            pass

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("draw_event", update_alpha)
    fig.canvas.mpl_connect("button_release_event", update_alpha)
    fig.canvas.mpl_connect("scroll_event", update_alpha)

    update_alpha()# Make the two plots

plot_conditions_depth_transparent(ABObjectives, ABValids, jointAngles,
    title="AB Conditions", s=6, mode="sigmoid", gamma=0.25)
plot_conditions_depth_transparent(SLObjectives, SLValids, jointAngles,
    title="SL Conditions", s=6, mode="exp", gamma=0.25)
plot_conditions_depth_transparent(IVObjectives, IVValids, jointAngles,
    title="IV Conditions", s=6, mode="exp", gamma=0.25)


# # resls = np.logspace(0,3,5)
# # resls = np.round(resls).astype(int)
# # print(resls)
# # pSkinnys = []
# # pAmbroses = []
# # for resl in resls:
# #     print(resl)
# #     percentSkinny = skinnyLegend.plotCapabilityAcrossAllGrasps(resl=resl, showBool=False)
# #     percentAmbrose = theAmbrose.plotCapabilityAcrossAllGrasps(resl=resl, showBool=False)
# #     pSkinnys.append(percentSkinny)
# #     pAmbroses.append(percentAmbrose)

# # plt.close('all')
# # plt.plot(resls, pSkinnys)
# # plt.plot(resls, pAmbroses)

plt.show()