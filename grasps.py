import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError, HalfspaceIntersection

from utils import find_axis_extent_lp, f_for_jac, rot, trans, jac

F_global = 6.12

def generateAllVertices(L):
    print("generating grasps")
    jointResl=50
    forceResl=50

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
    forceResl=2 # Critical grasps will only occur at full hook and full push

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
            if theta==0 or theta==np.pi/2:
                predictedBoundaries.append(Tau[i,j,:])
            i+=1
        j+=1
    # Flatten Torque Result
    Tau_flat = Tau.reshape(-1, 3)
    boundaryPoints = np.array([(Tau_flat[np.argmax(Tau_flat[:, 0])]),
                               (Tau_flat[np.argmin(Tau_flat[:, 0])]),
                               (Tau_flat[np.argmax(Tau_flat[:, 1])]),
                               (Tau_flat[np.argmin(Tau_flat[:, 1])]),
                               (Tau_flat[np.argmax(Tau_flat[:, 2])]),
                               (Tau_flat[np.argmin(Tau_flat[:, 2])])])
    boundaryPoints = np.vstack([boundaryPoints, predictedBoundaries])
    boundaryPoints = list(set([tuple(x) for x in boundaryPoints]))
    # Extract no-duplicate set of cube-bounded points
    # boundaryPoints = list(set([tuple(Tau_flat[np.argmax(Tau_flat[:, 0])]),
    #                             tuple(Tau_flat[np.argmin(Tau_flat[:, 0])]),
    #                             tuple(Tau_flat[np.argmax(Tau_flat[:, 1])]),
    #                             tuple(Tau_flat[np.argmin(Tau_flat[:, 1])]),
    #                             tuple(Tau_flat[np.argmax(Tau_flat[:, 2])]),
    #                             tuple(Tau_flat[np.argmin(Tau_flat[:, 2])]),
    #                             tuple(predictedBoundaries[0])]))
    boundaryPoints = np.array([list(point) for point in boundaryPoints])
    
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
    # Add the z-axis boundary
    boundaryPoints = np.vstack((boundaryPoints, np.array([0,0,minTorque])))
    return boundaryPoints

if __name__ == '__main__':
    l = np.array([1.375,1.4375,1.23])
    verts = generateNecessaryVertices(l)
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
            print(i,j)
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
            edgecolor='none'
        )

    ax.set_xlabel('Force Angle Phi')
    ax.set_ylabel('Joint Angle Theta')
    ax.set_zlabel('Torque')
    ax.set_title('Torque Components τ₁, τ₂, τ₃')
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
    ax.scatter(Tau_flat[:, 0], Tau_flat[:, 1], Tau_flat[:, 2],c=colors, alpha=1)
    ax.scatter(*verts.T, color=[0,0,1], s=100)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    torqueHull = ConvexHull(list_of_torque_vectors)
    # print(torqueHull.vertices)
    ax.scatter(*torqueHull.points[torqueHull.vertices].T, color=colors[torqueHull.vertices], alpha=1)
    for simplex in torqueHull.simplices:
        triangle = torqueHull.points[simplex]
        ax.add_collection3d(Poly3DCollection([triangle], color='xkcd:blue', alpha=0.2))
    plt.tight_layout()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.plot(xlim, [0, 0], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], ylim, [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [0, 0], zlim, color='black', linewidth=1)

    print(find_axis_extent_lp(torqueHull.points[torqueHull.vertices], [1,0,0]))
    print(find_axis_extent_lp(torqueHull.points[torqueHull.vertices], [0,1,0]))
    print(find_axis_extent_lp(torqueHull.points[torqueHull.vertices], [0,0,1]))

    # print(len(torqueHull.vertices))
    # print(len(Tau_flat))
    # jointAngleValue = jointAngles*

    # # Define resolution of the color key
    # res = resl

    # # Create normalized R and G grids
    # R = np.linspace(0, 1, res)
    # G = np.linspace(0, 1, res)
    # R_grid, G_grid = np.meshgrid(R, G)

    # # Fixed B value
    # B_val = 0.5

    # # Compose RGB image with shape (res, res, 3)
    # color_key = np.zeros((res, res, 3))
    # color_key[:, :, 0] = R_grid      # R channel
    # color_key[:, :, 1] = G_grid      # G channel
    # color_key[:, :, 2] = B_val       # B channel fixed

    # # Plot the color key
    # plt.figure(figsize=(6, 6))
    # plt.imshow(color_key, origin='lower', extent=[0, 1, 0, 1])
    # plt.xlabel('Normalized R value')
    # plt.ylabel('Normalized G value')
    # plt.title('Color key: R vs G (B fixed at 0.5)')
    # plt.colorbar(label='Intensity (Not used here, colors show RGB)')  # optional, can remove

    plt.show()