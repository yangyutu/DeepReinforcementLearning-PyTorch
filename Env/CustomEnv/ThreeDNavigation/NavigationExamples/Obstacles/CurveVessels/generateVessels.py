import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')


fig = plt.figure(1)
ax = fig.gca(projection='3d')

z_c = np.linspace(0, 500, 100)

k1 = 0.05
k2 = 0.02
R0 = 10
R1 = 10
midRadius = 25


y_c = np.cos(k1 * z_c)
x_c = np.zeros_like(z_c)



R_c = R0 * np.cos(k2 * z_c) + midRadius


ax.plot(x_c, y_c, z_c)

fig = plt.figure(2)
ax = fig.gca()

ax.plot(z_c, R_c)

theta = np.linspace(0, 2*np.pi, 40)
Theta, Z = np.meshgrid(theta, z_c)

Theta = Theta.flatten()
Z = Z.flatten()
R = R0 * np.cos(k2 * Z) + midRadius

YC = np.cos(k1 * Z) * R1
XC = np.zeros_like(Z)


fig = plt.figure(3)
ax = fig.gca(projection='3d')

ax.scatter(XC, YC, Z)

X = XC + R * np.cos(Theta)
Y = YC + R * np.sin(Theta)


fig = plt.figure(4)
ax = fig.gca(projection='3d')

ax.scatter(X, Y, Z)