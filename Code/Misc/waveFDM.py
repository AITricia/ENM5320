# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# Parameters
L = 1.0  # Length of the domain
T = 30.0  # Total time
nx = 200  # Number of spatial points
nt = 6000  # Number of time steps
c = 1.0  # Wave speed

# Discretization
dx = L / (nx - 1)
dt = T / nt
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
# Initial conditions
t0 = 0.0

# Solution arrays
q = np.zeros((nt, nx))
p = np.zeros((nt, nx))

# Initial condition
q[0, :] = np.exp(-100 * (x - 0.5*L)**2)
p[0, :] = 0.0

# Time-stepping loop using Verlet scheme
for n in range(nt-1):
    print('Step:', n)
    # Recall H = \sum_i 0.5 * p_i**2 h**-1 + 0.5 (D-q)_i**2 h

    # Verlet step 1: pnplushalf = pn - k/2 partial_q V(qn)
    partialqV = np.zeros_like(q[n, :])
    for i in range(nx):
        partialqV[i] = dx* (-c**2 / dx**2) * (q[n, (i + 1)%(nx-1)] - 2 * q[n, i] + q[n, (i - 1)%(nx-1)])
    pnplushalf = p[n, :] - 0.5 * dt * partialqV 

    # Verlet step 2: qnplusone = qn + k partial_p T(pnplushalf)
    partialpT = pnplushalf
    q[n + 1, :] = q[n,:] + dt * partialpT

    # Verlet step 3: pnplusone = pnplushalf - k/2 partial_q V(qnplusone)
    partialqV = np.zeros_like(q[n, :])
    for i in range(nx):
        partialqV[i] = dx*(-c**2 / dx**2) * (q[n + 1, (i + 1)%(nx-1)] - 2 * q[n + 1, i] + q[n + 1, (i - 1)%(nx-1)])
    p[n + 1, :] = pnplushalf - 0.5 * dt * partialqV


# %%
# Visualize solution in spacetime 
fig, ax = plt.subplots()
ax.imshow(q, aspect='auto', extent=[0, L, 0, T], cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')

# %%
# %% Visualize solution in spacetime as a 3D plot
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, q, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('q')
fig.colorbar(surf)
plt.title('Solution to Wave Equation')
plt.show()
# %%
