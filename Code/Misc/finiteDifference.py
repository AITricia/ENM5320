import numpy as np

import matplotlib.pyplot as plt

# Parameters
L = 10.0  # Length of the domain
T = 2.0   # Total time
nx = 100  # Number of spatial points
nt = 500  # Number of time steps
alpha = 0.01  # Thermal diffusivity

# Discretization
dx = L / nx
dt = T / nt
x = np.linspace(0, L, nx)
u = np.sin(2 * np.pi * x / L)  # Initial condition

# Time-stepping loop
for n in range(nt):
    u_new = u.copy()
    for i in range(nx):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[(i+1) % nx] - 2*u[i] + u[(i-1) % nx])
    u = u_new

# Plot the final solution
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('1D Heat Equation with Periodic Boundary Conditions')
plt.show()