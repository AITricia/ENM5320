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
meshsizex = 10
meshsizet = 10
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
plot_multiple_3d(3,3)

# %% Get Quadrature points on reference element
q1 = h * (0.5 + 1. / (2. * np.sqrt(3)))
q2 = h * (0.5 - 1. / (2. * np.sqrt(3)))
xq = np.array([[q1,q1], [q2,q1], [q1,q2],[q2,q2]]) 


# %% Construct mass and stiffness matrices in sparse matrix format

# Initialize sparse tensors with pre-allocated space
indices_list_M0 = []
values_list_M0 = []
indices_list_S = []
values_list_S = []
indices_list_M1 = []
values_list_M1 = []

# For every element
for i in range(meshsizex-1):
    for j in range(meshsizet-1):
        el_xq = torch.stack([pointsx[i] + q1, pointsx[i] + q2])
        el_tq = torch.stack([pointst[j] + q1, pointst[j] + q2])
        
        # Evaluate basis functions on quadrature points
        psi0_ij = torch.einsum('iq,jp->ijqp', evalPhi_i(el_xq,pointsx), evalPhi_i(el_tq,pointst))
        gradpsix_ij = torch.einsum('iq,jp->ijqp', evalGradPhi_i(el_xq,pointsx), evalPhi_i(el_tq,pointst))
        gradpsit_ij = torch.einsum('iq,jp->ijqp', evalPhi_i(el_xq,pointsx), evalGradPhi_i(el_tq,pointst))
        gradpsi_ij = torch.cat([torch.unsqueeze(gradpsix_ij,0), torch.unsqueeze(gradpsit_ij,0)], dim=0)
        psi1_ijkl = torch.einsum('ijqp,aklqp->aijklqp', psi0_ij, gradpsi_ij)-torch.einsum('aijqp,klqp->aijklqp', gradpsi_ij, psi0_ij)
        
        # Local to global mapping
        local_nodes = [(i,j), (i+1,j), (i,j+1), (i+1,j+1)]
        
        # M0 construction
        for idx1, (a,b) in enumerate(local_nodes):
            for idx2, (c,d) in enumerate(local_nodes):
                val = (h/2)**2 * (psi0_ij[a,b,:,:]*psi0_ij[c,d,:,:]).sum()
                if abs(val) > 1e-14:  # Only store non-zero values
                    indices_list_M0.append([a,b,c,d])
                    values_list_M0.append(val)
                val = (h/2)**2 * (gradpsi_ij[:,a,b,:,:]*gradpsi_ij[:,c,d,:,:]).sum()
                if abs(val) > 1e-14:  # Only store non-zero values
                    indices_list_S.append([a,b,c,d])
                    values_list_S.append(val)    
                # M1 construction
                for idx3, (e,f) in enumerate(local_nodes):
                    for idx4, (g,h) in enumerate(local_nodes):
                        val = (h/2)**2*(psi1_ijkl[:,a,b,c,d,:,:]*
                                      psi1_ijkl[:,e,f,g,h,:,:]).sum()
                        if abs(val) > 1e-14:  # Only store non-zero values
                            indices_list_M1.append([a,b,c,d,e,f,g,h])
                            values_list_M1.append(val)

# Create sparse tensors in one shot
indices = torch.tensor(indices_list_M0).t()
values = torch.tensor(values_list_M0)
M0 = torch.sparse_coo_tensor(indices, values, 
                            size=(meshsizex,meshsizet,meshsizex,meshsizet))

# Create sparse tensors in one shot
indices = torch.tensor(indices_list_S).t()
values = torch.tensor(values_list_S)
S = torch.sparse_coo_tensor(indices, values, 
                            size=(meshsizex,meshsizet,meshsizex,meshsizet))

indices_M1 = torch.tensor(indices_list_M1).t()
values_M1 = torch.tensor(values_list_M1)
M1 = torch.sparse_coo_tensor(indices_M1, values_M1, 
                            size=(meshsizex,meshsizet,meshsizex,meshsizet,
                                 meshsizex,meshsizet,meshsizex,meshsizet))

#Coalesce the sparse tensors
M0 = M0.coalesce()
S = S.coalesce()
M1 = M1.coalesce()
# %%
#Convert M0 to numpy matrix format and plot with spy
M0_dense = M0.to_dense()
plt.spy(torch.reshape(M0_dense,(100,100)).numpy())
# Also store matrices in 2d format for later use


# # %% Construct coboundary matrix
# # Construct adjacency matrix
D = torch.zeros((meshsizex,meshsizet,meshsizex,meshsizet), dtype=torch.float64)
for i in range(meshsizex):
    for j in range(meshsizet):
        for k in range(meshsizex):
            for l in range(meshsizet):
                D[i,j,k,l] += 1.
                D[i,j,i,j] -= 1.

# %% Construct discretization for Poisson with Dirichlet BCs on left and right

# Build flag vector to identify boundary nodes
boundary = torch.zeros((meshsizex,meshsizet), dtype=torch.float64)
boundary[0,:] = 1
boundary[-1,:] = 1
boundary[:,0] = 1
boundary[:,-1] = 1
boundary_flat = boundary.flatten()

# Set up forcing function evaluated on the nodes and specify dirichlet conditions
forcing = torch.einsum('ijkl,kl->ij', M0_dense, torch.ones_like(boundary))
rhs = (1.-boundary_flat)*torch.flatten(forcing)

#Build matrices
Amat = torch.zeros((meshsizex,meshsizet,meshsizex,meshsizet), dtype=torch.float64)
for i in range(meshsizex):
    for j in range(meshsizet):
        if boundary[i,j] == 1:
            Amat[i,j,i,j] = 1.
        else:
            # Very painful to do matrix slicing with sparse pytorch tensors
            # Get indices and values for the specific i,j slice
            mask = (S.indices()[0] == i) & (S.indices()[1] == j)
            slice_indices = S.indices()[:,mask][[2,3]]  # Keep only k,l indices
            slice_values = S.values()[mask]

            # Create new sparse tensor for this slice
            S_slice = torch.sparse_coo_tensor(
                slice_indices, 
                slice_values,
                size=(meshsizex, meshsizet)
            )
            Amat[i,j,:,:] = S_slice.to_dense()
# Flatten into a matrix
Amat_flat = Amat.reshape(meshsize,meshsize)            
# Solve the linear system
u_sol = torch.linalg.solve(Amat_flat, rhs)


# %% Visualize solution
u_sol_grid = u_sol.reshape(meshsizet,meshsizex)
plt.imshow(u_sol_grid.detach().numpy(), cmap='viridis')
plt.colorbar()
plt.title('Solution to Poisson equation')
plt.show()

# %% Visualize solution
u_sol_grid = u_sol.reshape(meshsizet,meshsizex)

# Create meshgrid for plotting
X, Y = np.meshgrid(pointsx.numpy(), pointst.numpy())

# Create 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u_sol_grid.detach().numpy(), 
                      cmap='viridis',
                      linewidth=0,
                      antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.colorbar(surf)
plt.title('Solution to Poisson equation')
plt.show()
# %%
