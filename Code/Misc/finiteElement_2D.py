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
Tdomain = 1
Xdomain = 1
meshsizex = 40
meshsizet = 40
h = Xdomain/float(meshsizex-1)
k = Tdomain/float(meshsizet-1)
meshsize = meshsizex*meshsizet

pointsx = torch.tensor(np.linspace(0,Xdomain,meshsizex), dtype=torch.float64)  # Replace [...] with actual values
pointst = torch.tensor(np.linspace(0,Tdomain,meshsizet), dtype=torch.float64)  # Replace [...] with actual values
points = torch.cartesian_prod(pointsx, pointst)
finepointsx = torch.tensor(np.linspace(0,Xdomain,20*meshsizex), dtype=torch.float64)  # Replace [...] with actual values
finepointst = torch.tensor(np.linspace(0,Tdomain,20*meshsizet), dtype=torch.float64)  # Replace [...] with actual values
finepoints = torch.cartesian_prod(finepointsx, finepointst)

def evalPhi_i(x,pts):
    x_expanded = torch.unsqueeze(x, 0)
    points_expanded = torch.unsqueeze(pts, 1)
    return torch.relu(1.0 - (torch.abs(x_expanded - points_expanded)) / h)

def evalGradPhi_i(x,pts):
    suppPhi = (evalPhi_i(x,pts) > 0).double()
    signPlus = (torch.unsqueeze(pts, 1) > torch.unsqueeze(x, 0)).double()
    signNeg = (torch.unsqueeze(pts, 1) <= torch.unsqueeze(x, 0)).double()
    return suppPhi * (-signPlus + signNeg) / h

# %% Visualize shape functions and their derivatives evaluated over a fine grid
X, T = torch.meshgrid(finepointsx, finepointst)
xflat = X.flatten()
tflat = T.flatten()
psi0_ij = torch.einsum('iq,jq->ijq', evalPhi_i(xflat,pointsx), evalPhi_i(tflat,pointst))

# Reshape back into meshgrid format
psi0_ij = psi0_ij.reshape(meshsizet,meshsizex, 20*meshsizet, 20*meshsizex)
# Grid of 3D plots
def plot_multiple_3d(n_rows=meshsizex, n_cols=meshsizet):
    fig = plt.figure(figsize=(15, 15))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i*n_cols + j + 1, projection='3d')
            Z = psi0_ij[i,j,:,:].numpy()
            surf = ax.plot_surface(X.numpy(), T.numpy(), Z,
                                 cmap='viridis',
                                 linewidth=0,
                                 antialiased=True)
            ax.set_title(f'ψ({i},{j})')
            # fig.colorbar(surf, ax=ax)
    plt.tight_layout()
    plt.show()
plot_multiple_3d()

# %% Get Quadrature points in space and time
xql = pointsx[:-1].numpy() + h * (0.5 + 1. / (2. * np.sqrt(3)))
xqr = pointsx[:-1].numpy() + h * (0.5 - 1. / (2. * np.sqrt(3)))
xq = np.sort(np.concatenate([xql, xqr]))
xq = torch.tensor(xq, dtype=torch.float64)
tql = pointst[:-1].numpy() + h * (0.5 + 1. / (2. * np.sqrt(3)))
tqr = pointst[:-1].numpy() + h * (0.5 - 1. / (2. * np.sqrt(3)))
tq = np.sort(np.concatenate([tql, tqr]))
tq = torch.tensor(tq, dtype=torch.float64)

# Evaluate basis functions on quadrature points
psi0_ij = torch.einsum('iq,jp->ijqp', evalPhi_i(xq,pointsx), evalPhi_i(tq,pointst))
gradpsix_ij = torch.einsum('iq,jp->ijqp', evalGradPhi_i(xq,pointsx), evalPhi_i(tq,pointst))
gradpsit_ij = torch.einsum('iq,jp->ijqp', evalPhi_i(xq,pointsx), evalGradPhi_i(tq,pointst))
gradpsi_ij = torch.cat([torch.unsqueeze(gradpsix_ij,0), torch.unsqueeze(gradpsit_ij,0)], dim=0)
psi1_ijkl = torch.einsum('ijqp,aklqp->aijklqp', psi0_ij, gradpsi_ij)-torch.einsum('aijqp,klqp->aijklqp', gradpsi_ij, psi0_ij)


# %% Construct matrices
# mass matrix of P1 basis functions
M0 = (h/2)**2*torch.einsum('ijqp,klqp->ijkl', psi0_ij, psi0_ij)
M1 = (h/2)**2*torch.einsum('aijklqp,auvwxqp->ijkluvwx', psi1_ijkl, psi1_ijkl)
# Construct adjacency matrix
D = torch.zeros((meshsizex,meshsizet,meshsizex,meshsizet), dtype=torch.float64)
for i in range(meshsizex):
    for j in range(meshsizet):
        for k in range(meshsizex):
            for l in range(meshsizet):
                D[i,j,k,l] += 1.
                D[i,j,i,j] -= 1.
# Construct stiffness matrix
S = (h/2)**2*torch.einsum('aijqp,aklqp->ijkl', gradpsi_ij, gradpsi_ij)
            
# Confirm the identity the D^T*M1*D = S
S_identity = torch.einsum('ijkl,ijkluvwx,uvwx->ijuv', D, M1, D)
print('Unit test: S - D^T*M1*D = ',np.abs((S-S_identity).detach().numpy()).sum())
# %% Construct discretization for Poisson with Dirichlet BCs on left and right

# Build flag vector to identify boundary nodes
boundary = torch.zeros((meshsizex,meshsizet), dtype=torch.float64)
boundary[0,:] = 1
boundary[-1,:] = 1
boundary[:,0] = 1
boundary[:,-1] = 1
boundary_flat = boundary.flatten()

# Set up forcing function evaluated on the nodes and specify dirichlet conditions
forcing = (1.-boundary_flat)*torch.ones(meshsize, dtype=torch.float64)
uLHS = 1.0
uRHS = 0.0

#Build matrices
Amat = torch.zeros_like(S)
for i in range(meshsizex):
    for j in range(meshsizet):
        if boundary[i,j] == 1:
            Amat[i,j,i,j] = 1.
        else:
            Amat[i,j,:,:] = S[i,j,:,:]
# Flatten into a matrix
Amat_flat = Amat.reshape(meshsize,meshsize)            
# Solve the linear system
u_sol = torch.linalg.solve(Amat_flat, forcing)


# %% Visualize solution
u_sol_grid = u_sol.reshape(meshsizet,meshsizex)
plt.imshow(u_sol_grid.detach().numpy(), cmap='viridis')
plt.colorbar()
plt.title('Solution to Poisson equation')
plt.show()

# %%
