import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError, HalfspaceIntersection

from .utils import find_axis_extent_lp, ee_func, rot, trans, jac

F_global = 6.12


def generateAllVertices(L):
    print("generating grasps")
    jointResl=25
    forceResl=25

    l = L
    jointAngles = np.linspace(0,np.pi/2,jointResl)
    forceAngles = np.linspace(0,np.pi/2,forceResl)

    F = F_global
    F_t = np.array([[0],[F]])
    Tau = np.zeros([len(jointAngles), len(forceAngles), 3])

    # print(trans(np.array([1,2,3]), l))
    # print(jac(np.array([0,0,0]), l))

    j=0
    for phi in forceAngles:
        i = 0
        for theta in jointAngles:
            # print(i,j)
            Q = np.array([theta, theta, theta])
            F = trans(Q, l)[:-1,:-1] @ (np.array([[np.cos(phi),-np.sin(phi)],
                                                [np.sin(phi), np.cos(phi)]]) @ F_t)
            F = np.vstack((F,0))
            Tau[i,j,:] = (jac(Q,l).T @ F).flatten()
            i+=1
        j+=1

    list_of_torque_vectors = np.array([Tau[i, j, :] for i in range(Tau.shape[0]) for j in range(Tau.shape[1])])

    torqueHull = ConvexHull(list_of_torque_vectors)
    # print(torqueHull.vertices)
    return np.array(torqueHull.points[torqueHull.vertices])

def generateNecessaryVertices(L):
    print("generating grasps")
    jointResl=100
    forceResl=100 # Critical grasps will only occur at full hook and full push

    l = L
    # Generate list of joint and force angles
    jointAngles = np.linspace(0,np.pi/2,jointResl)
    forceAngles = np.linspace(0,np.pi/2,forceResl)
    # Magintude of force, initialize torque-space result array
    F = F_global
    F_t = np.array([[0],[F]])
    Tau = np.zeros([len(jointAngles), len(forceAngles), 3])
    # Solve IK for each torque-space grasp
    predictedBoundaries=[]
    predictedAngles =[]
    j=0
    for phi in forceAngles:
        i = 0
        for theta in jointAngles:
            # print(i,j)
            Q = np.array([theta, theta, theta])
            F = trans(Q, l)[:-1,:-1] @ (np.array([[np.cos(phi),-np.sin(phi)],
                                                [np.sin(phi), np.cos(phi)]]) @ F_t)
            F = np.vstack((F,0))
            Tau[i,j,:] = (jac(Q,l).T @ F).flatten()
            if (theta, phi) in {(0, np.pi/2), (np.pi/2, 0), (0, 0), (np.pi/2, np.pi/2)}:
                predictedBoundaries.append(Tau[i,j,:])
                predictedAngles.append((theta, phi))
            i+=1
        j+=1

    def index_to_angles(idx):
        i = idx // forceResl
        j = idx % forceResl
        return (jointAngles[i], forceAngles[j])
    # Flatten Torque Result
    Tau_flat = Tau.reshape(-1, 3)

    idx_list = [
        np.argmax(Tau_flat[:, 0]),
        np.argmin(Tau_flat[:, 0]),
        np.argmax(Tau_flat[:, 1]),
        np.argmin(Tau_flat[:, 1]),
        # np.argmax(Tau_flat[:, 2]),
        # np.argmin(Tau_flat[:, 2])
    ]
    boundaryPoints = [Tau_flat[idx] for idx in idx_list]
    boundaryAngles = [index_to_angles(idx) for idx in idx_list]

    boundaryPoints.extend(predictedBoundaries)
    boundaryAngles.extend(predictedAngles)

    # boundaryPoints = np.array([(Tau_flat[np.argmax(Tau_flat[:, 0])]),
    #                            (Tau_flat[np.argmin(Tau_flat[:, 0])]),
    #                            (Tau_flat[np.argmax(Tau_flat[:, 1])]),
    #                            (Tau_flat[np.argmin(Tau_flat[:, 1])]),
    #                            (Tau_flat[np.argmax(Tau_flat[:, 2])]),
    #                            (Tau_flat[np.argmin(Tau_flat[:, 2])])])
    # boundaryPoints = np.vstack([boundaryPoints, predictedBoundaries])
    # boundaryPoints = list(set([tuple(x) for x in boundaryPoints]))
    # Extract no-duplicate set of cube-bounded points
    # boundaryPoints = list(set([tuple(Tau_flat[np.argmax(Tau_flat[:, 0])]),
    #                             tuple(Tau_flat[np.argmin(Tau_flat[:, 0])]),
    #                             tuple(Tau_flat[np.argmax(Tau_flat[:, 1])]),
    #                             tuple(Tau_flat[np.argmin(Tau_flat[:, 1])]),
    #                             tuple(Tau_flat[np.argmax(Tau_flat[:, 2])]),
    #                             tuple(Tau_flat[np.argmin(Tau_flat[:, 2])]),
    #                             tuple(predictedBoundaries[0])]))
    boundaryPoints = np.array([list(point) for point in boundaryPoints])

    pts_angles = list({tuple(p): a for p, a in zip(boundaryPoints, boundaryAngles)}.items())
    boundaryPoints = np.array([list(p) for p, _ in pts_angles])
    boundaryAngles = np.array([a for _, a in pts_angles])

    # Clean up results
    deletes = []
    minTorque = 1000
    for i in range(boundaryPoints.shape[0]):
        for j in range(boundaryPoints.shape[1]):
            # Remove small numbers and replace with 0
            if np.isclose(boundaryPoints[i,j],0):
                boundaryPoints[i,j]=0
            # track minimum torque to use as a z-axis boundary
            elif abs(boundaryPoints[i,j])<abs(minTorque):
                minTorque = -abs(boundaryPoints[i,j])
        # Get rid of stupid origin points that show up
        if np.array_equal(boundaryPoints[i,:],np.array([0,0,0])):
            deletes.append(i)
    # delete everything to delete
    boundaryPoints = np.delete(boundaryPoints, deletes, axis=0)
    boundaryAngles = np.delete(boundaryAngles, deletes, axis=0)
    # Add the z-axis boundary
    # boundaryPoints = np.vstack((boundaryPoints, np.array([0,0,minTorque])))
    # boundaryAngles = np.vstack((boundaryAngles, np.array([0, 0])))
    return boundaryPoints, boundaryAngles

def generateGrasps(L):
    taus, thetaPhis = generateNecessaryVertices(L)
    return np.array([[[theta[0], theta[0], theta[0]], list(tau)] for tau, theta in zip(taus, thetaPhis)])

def main():
    l = np.array([1.375,1.4375,1.23])
    verts, angles = generateNecessaryVertices(l)
    print(verts)
    jointResl=100
    forceResl=100

    jointAngles = np.linspace(0,np.pi/2,jointResl)
    forceAngles = np.linspace(0,np.pi/2,forceResl)

    F = F_global
    F_t = np.array([[0],[F]])
    Tau = np.zeros([len(jointAngles), len(forceAngles), 3])

    j=0
    for phi in forceAngles:
        i = 0
        for theta in jointAngles:
            # print(i,j)
            Q = np.array([theta, theta, theta])
            F = trans(Q, l)[:-1,:-1] @ (np.array([[np.cos(phi),-np.sin(phi)],
                                                [np.sin(phi), np.cos(phi)]]) @ F_t)
            F = np.vstack((F,0))
            Tau[i,j,:] = (jac(Q,l).T @ F).flatten()
            i+=1
        j+=1

    THETA, PHI = np.meshgrid(forceAngles, jointAngles)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [
        (1, 0, 0),  # red for τ1
        (0, 1, 0),  # green for τ2
        (0, 0, 1)   # blue for τ3
    ]

    for t in range(3):
        ax.plot_surface(
            THETA, PHI, Tau[:, :, t],
            color=colors[t],
            alpha=0.6,       # transparency
            edgecolor='none',
            shade=False
        )
    # ax.scatter(*verts.T)
    zmin, zmax = ax.get_zlim()  # or set manually if you want fixed range

    # Draw vertical lines
    print(angles)
    for phi, theta in angles:
        ax.plot(
            [theta, theta],   # X = theta constant
            [phi, phi],       # Y = phi constant
            [zmin, zmax],     # Z from min to max
            color='black', linewidth=3
        )

    ax.set_xlabel('Force Angle φ')
    ax.set_ylabel('Joint Angle θ')
    ax.set_zlabel('Torque')
    ax.set_title('Target Grasps in Force Space')
    ax.view_init(elev=30, azim=45)
    ax.grid(True)

    # Matplotlib doesn't auto-legend 3D surfaces; you have to fake it:
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=colors[0], label=r'$\tau_1$'),
        Patch(color=colors[1], label=r'$\tau_2$'),
        Patch(color=colors[2], label=r'$\tau_3$')
    ]
    ax.legend(handles=legend_patches)

    list_of_torque_vectors = np.array([Tau[i, j, :] for i in range(Tau.shape[0]) for j in range(Tau.shape[1])])

    R_vals = np.linspace(0,1,jointResl)
    G_vals = np.linspace(0,1,forceResl)
    # Broadcast R and G into matching shapes
    R_grid = np.repeat(R_vals[:, np.newaxis], forceResl, axis=1)  # shape (num_R, num_G)
    G_grid = np.repeat(G_vals[np.newaxis, :], jointResl, axis=0)  # shape (num_R, num_G)
    # Flatten everything to 1D arrays for scatter
    R_flat = R_grid.flatten()
    G_flat = G_grid.flatten()

    Tau_flat = Tau.reshape(-1, 3)
    B_vals = np.linalg.norm(Tau_flat, axis=1)
    # B_norm = (B_vals - B_vals.min()) / (B_vals.max() - B_vals.min())
    B_norm = np.array([0.5]*jointResl*forceResl)
    colors = np.stack([R_flat, G_flat, B_norm], axis=1)
    # print(R_flat.shape, R_flat.shape, B_norm.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Tau_flat[:, 0], Tau_flat[:, 1], Tau_flat[:, 2],c=colors, alpha=0.15)
    ax.scatter(*verts.T, color='black', alpha=1, s=100)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
    ax.set_xlabel('τ₁', fontsize=18)
    ax.set_ylabel('τ₂', fontsize=18)
    ax.set_zlabel('τ₃', fontsize=18)
    ax.set_title('Target Grasps in Torque Space')


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # torqueHull = ConvexHull(list_of_torque_vectors)
    # # print(torqueHull.vertices)
    # ax.scatter(*torqueHull.points[torqueHull.vertices].T, color=colors[torqueHull.vertices], alpha=1)
    # for simplex in torqueHull.simplices:
    #     triangle = torqueHull.points[simplex]
    #     ax.add_collection3d(Poly3DCollection([triangle], color='xkcd:blue', alpha=0.2))
    # plt.tight_layout()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # zlim = ax.get_zlim()
    # ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
    # ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
    # ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)

    # print(find_axis_extent_lp(torqueHull.points[torqueHull.vertices], [1,0,0]))
    # print(find_axis_extent_lp(torqueHull.points[torqueHull.vertices], [0,1,0]))
    # print(find_axis_extent_lp(torqueHull.points[torqueHull.vertices], [0,0,1]))

    # print(verts)
    # print(angles)

    # print(len(torqueHull.vertices))
    # print(len(Tau_flat))
    # jointAngleValue = jointAngles*

    # Define resolution of the color key

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # legend_ax = fig.add_axes([0.8, 0.6, 0.15, 0.15])
    # legend_ax = inset_axes(ax,
    #                        width="20%",  # size of legend relative to main axes
    #                        height="20%",
    #                        loc="lower right",  # position
    #                        borderpad=1.5)

    res = forceResl

    # Create normalized R and G grids
    R = np.linspace(0, 1, res)
    G = np.linspace(0, 1, res)
    R_grid, G_grid = np.meshgrid(R, G)

    # Fixed B value
    B_val = 0.5

    # Compose RGB image with shape (res, res, 3)
    color_key = np.zeros((res, res, 3))
    color_key[:, :, 0] = R_grid      # R channel
    color_key[:, :, 1] = G_grid      # G channel
    color_key[:, :, 2] = B_val       # B channel fixed


    legend_ax = fig.add_axes([0.85, 0.85, 0.15, 0.15])  # [left, bottom, width, height]
    legend_ax.imshow(color_key, origin='lower', extent=[0, 1, 0, 1])

    # Keep axis labels but remove ticks
    legend_ax.set_xlabel(r'$\theta$', fontsize=16, labelpad=2)
    legend_ax.set_ylabel(r'$\phi$', fontsize=16, labelpad=2)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])

    plt.show()

if __name__ == '__main__':
    # main()
    l = np.array([1.375,1.4375,1.23])
    print(generateGrasps(l))