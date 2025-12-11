import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,np.pi*2,10000)

d= 1.5
R = 8.09
n = 10
E = 0.625

r = d/2
n_p = n+1

X =  (R*np.cos(t))-(r*np.cos(t+np.arctan(np.sin(-n*t)/((R/(n_p*E))-np.cos(-n*t)))))-(E*np.cos(n_p*t))
Y = (-R*np.sin(t))+(r*np.sin(t+np.arctan(np.sin(-n*t)/((R/(n_p*E))-np.cos(-n*t)))))+(E*np.sin(n_p*t))

# print(np.stack((X,Y,np.zeros_like(X)),axis=1))
np.savetxt('path2.sldcrv', np.stack((X,Y,np.zeros_like(X)),axis=1))
plt.plot(X,Y)
plt.show()