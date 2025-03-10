# %%

import numpy as np
import pandas as pd
import seaborn as sns
import torch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# %%
# Define points and h as needed
Csq = 1
Xdomain = 1
Tdomain = 2
meshsizex = 16
meshsizet = meshsizex*int(2.0*Tdomain/Xdomain)
h = Xdomain/float(meshsizex-1)
k = Tdomain/float(meshsizet-1)
meshsize = meshsizex*meshsizet

pointsx = torch.tensor(np.linspace(0,Xdomain,meshsizex), dtype=torch.float64)  # Replace [...] with actual values
pointst = torch.tensor(np.linspace(0,Tdomain,meshsizet), dtype=torch.float64)  # Replace [...] with actual values
points = torch.cartesian_prod(pointsx, pointst)

def evalPhi_i(x,pts,delta):
    x_expanded = torch.unsqueeze(x, 0)
    points_expanded = torch.unsqueeze(pts, 1)
    return torch.relu(1.0 - (torch.abs(x_expanded - points_expanded)) / delta)

def evalGradPhi_i(x,pts,delta):
    suppPhi = (evalPhi_i(x,pts,delta) > 0).double()
    signPlus = (torch.unsqueeze(pts, 1) > torch.unsqueeze(x, 0)).double()
    signNeg = (torch.unsqueeze(pts, 1) <= torch.unsqueeze(x, 0)).double()
    return suppPhi * (-signPlus + signNeg) / delta

# %% Get Quadrature points in space and time
xql = pointsx[:-1].numpy() + h * (0.5 + 1. / (2. * np.sqrt(3)))
xqr = pointsx[:-1].numpy() + h * (0.5 - 1. / (2. * np.sqrt(3)))
xq = np.sort(np.concatenate([xql, xqr]))
xq = torch.tensor(xq, dtype=torch.float64)
tql = pointst[:-1].numpy() + k * (0.5 + 1. / (2. * np.sqrt(3)))
tqr = pointst[:-1].numpy() + k * (0.5 - 1. / (2. * np.sqrt(3)))
tq = np.sort(np.concatenate([tql, tqr]))
tq = torch.tensor(tq, dtype=torch.float64)

# Evaluate basis functions on quadrature points
psi0_ij = torch.einsum('iq,jp->ijqp', evalPhi_i(xq,pointsx,h), evalPhi_i(tq,pointst,k))
gradpsix_ij = torch.einsum('iq,jp->ijqp', evalGradPhi_i(xq,pointsx,h), evalPhi_i(tq,pointst,k))
gradpsit_ij = torch.einsum('iq,jp->ijqp', evalPhi_i(xq,pointsx,h), evalGradPhi_i(tq,pointst,k))
gradpsi_ij = torch.cat([torch.unsqueeze(gradpsix_ij,0), torch.unsqueeze(gradpsit_ij,0)], dim=0)


# %% Construct matrices
# mass matrix of P1 basis functions
M0 = (h/2)*(k/2)*torch.einsum('ijqp,klqp->ijkl', psi0_ij, psi0_ij)
# time stiffness matrix
St = (h/2)*(k/2)*torch.einsum('ijqp,klqp->ijkl', gradpsit_ij, gradpsit_ij)
Sx = (h/2)*(k/2)*torch.einsum('ijqp,klqp->ijkl', gradpsix_ij, gradpsix_ij)

# Build space-time stiffness matrix
S = St - Csq*Sx 

# %%
# Build flag vector to identify Dirichlet boundary nodes
boundary = torch.zeros((meshsizex,meshsizet), dtype=torch.float64)


#Initial condition
boundary[:,0] = 2 # flag to set Dirichlet BC
#Final condition
boundary[:,-1] = 0

#Spatial BC
boundary[-1,:] = 1 # flag to set periodic BC
# boundary[-1,:] = 2 # flag to set Dirichlet BC
# boundary[ 0,:] = 2  # flag to set Dirichlet BC


boundary_flat = boundary.flatten()

# %%
# Set up forcing function evaluated on the nodes and specify dirichlet conditions
forcing = torch.zeros((meshsizex,meshsizet), dtype=torch.float64)
for i in range(meshsizex):
    for j in range(meshsizet):
        if boundary[i,j] == 2: 
            forcing[i,j] = torch.sin(2*np.pi*(pointsx[i]-pointst[j]))+torch.sin(2*np.pi*(pointsx[i]+pointst[j])) # Initial condition
            # forcing[i,j] = torch.cos(2*np.pi*(pointsx[i]-pointst[j]))+torch.cos(2*np.pi*(pointsx[i]+pointst[j])) # Initial condition

# Flatten
forcing_flat = forcing.flatten()
# %% Solve the linear system SADSDAF
#Build matrices
Amat = torch.zeros_like(S)
for i in range(meshsizex):
    for j in range(meshsizet):
        if boundary[i,j] == 0: # Solve galerkin eqn
            Amat[i,j,:,:] = S[i,j,:,:]
        if boundary[i,j] == 2: # Dirichlet BC on IC
            Amat[i,j,i,j] = 1.
        if boundary[i,j] == 1: # Periodic BC
            Amat[i,j,i,j] =  1.
            Amat[i,j,0,j] = -1.
            Amat[0,j,i,:] += S[0,j,i,:]
# Flatten into a matrix
Amat_flat = Amat.reshape(meshsize,meshsize)            
# Solve the linear system
u_sol = torch.linalg.solve(Amat_flat, forcing_flat)


# %% Visualize solution
u_sol_grid = u_sol.reshape(meshsizex,meshsizet)
plt.imshow(u_sol_grid.detach().numpy(), cmap='viridis')
plt.title('Solution to wave equation')
plt.colorbar()
plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D

# Visualize solution as a 3D surface plot
u_sol_grid = u_sol.reshape(meshsizex, meshsizet).detach().numpy()
X, T = np.meshgrid(pointsx.numpy(), pointst.numpy())

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, u_sol_grid.T, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('Solution to wave equation')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
# %%
