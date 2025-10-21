from strucMatrices import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

springStiffness = 40.13
tendonStiffness = 40.13

def plot_ellipsoid(K, color):

    U, s, rotation = np.linalg.svd(K_J)
    radii = np.sqrt(s)

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation)

    if plot_ellipsoid.has_been_called:
        plot_ellipsoid.ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)
    else:
        plot_ellipsoid.fig = plt.figure()
        plot_ellipsoid.ax = plot_ellipsoid.fig.add_subplot(111, projection='3d')
        plot_ellipsoid.ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)
        plot_ellipsoid.has_been_called = True

largeSpringStiffnesses = np.linspace(tendonStiffness, tendonStiffness*10, 10)
smallSpringStiffnesses = np.linspace(tendonStiffness, tendonStiffness/10, 10)
# print(springStiffnesses)

def majorRadius(ks):
    Ks = [testStructure() @ np.linalg.inv((np.diag([1/tendonStiffness]*4)+np.diag([1/k]*4))) @ testStructure().T for k in ks]
    return [np.sqrt(np.max(np.linalg.eigh(K).eigenvalues)) for K in Ks]
def volume(ks):
    Ks = [testStructure() @ np.linalg.inv((np.diag([1/tendonStiffness]*4)+np.diag([1/k]*4))) @ testStructure().T for k in ks]
    return [4.0/3.0*np.pi*np.linalg.det((K)) for K in Ks]
testStructure = individualType1
plt.figure()
plt.plot(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses]), volume(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses])))
testStructure = canonB
plt.plot(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses]), volume(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses])))

testStructure = individualType1
plt.figure()
plt.plot(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses]), majorRadius(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses])))
testStructure = canonB
plt.plot(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses]), majorRadius(np.concatenate([smallSpringStiffnesses[::-1], largeSpringStiffnesses])))

testStructure = canonB

plot_ellipsoid.has_been_called = False
for springStiffness in largeSpringStiffnesses:
    K_A = np.linalg.inv((np.diag([1/tendonStiffness]*4)+np.diag([1/springStiffness]*4)))
    K_J = testStructure() @ K_A @ testStructure().T
    print(K_J)
    # print(np.linalg.eigh(K_J))
    plot_ellipsoid(K_J, color='r')
for springStiffness in smallSpringStiffnesses:
    K_A = np.linalg.inv((np.diag([1/tendonStiffness]*4)+np.diag([1/springStiffness]*4)))
    K_J = testStructure() @ K_A @ testStructure().T
    print(K_J)
    # print(np.linalg.eigh(K_J))
    plot_ellipsoid(K_J, color='b')

plt.show()