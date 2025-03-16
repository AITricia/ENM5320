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
meshsize= 4
h = 1./float(meshsize-1)
points = torch.tensor(np.linspace(0,1,meshsize), dtype=torch.float64) 
finepoints = torch.tensor(np.linspace(0,1,21*meshsize), dtype=torch.float64)  

def evalPhi_i(x):
    x_expanded = torch.unsqueeze(x, 0)
    points_expanded = torch.unsqueeze(points, 1)
    return torch.relu(1.0 - (torch.abs(x_expanded - points_expanded)) / h)

def evalGradPhi_i(x):
    suppPhi = (evalPhi_i(x) > 0).double()
    signPlus = (torch.unsqueeze(points, 1) > torch.unsqueeze(x, 0)).double()
    signNeg = (torch.unsqueeze(points, 1) <= torch.unsqueeze(x, 0)).double()
    return suppPhi * (-signPlus + signNeg) / h

# Visualize shape functions and their derivatives evaluated over a fine grid
phi_i = evalPhi_i(finepoints)
grad_phi_i = evalGradPhi_i(finepoints)

plt.plot(finepoints.numpy(), phi_i.numpy().T)
plt.title("Phi_i")
plt.show()
plt.figure()
plt.plot(finepoints.numpy(), grad_phi_i.numpy().T)
plt.title("Grad Phi_i")
plt.show()

# %% Get Quadrature points
xql = points[:-1].numpy() + h * (0.5 + 1. / (2. * np.sqrt(3)))
xqr = points[:-1].numpy() + h * (0.5 - 1. / (2. * np.sqrt(3)))
xq = np.sort(np.concatenate([xql, xqr]))
xq = torch.tensor(xq, dtype=torch.float64)

# %% Construct matrices
#mass matrix of P1 basis functions
nodal_basisEval = evalPhi_i(xq)
Mnodal = (h / 2.) * torch.sum(torch.unsqueeze(nodal_basisEval, 0) * torch.unsqueeze(nodal_basisEval, 1), dim=2)

#stiffness matrix of P1 basis functions
nodal_gradbasisEval = evalGradPhi_i(xq)
Snodal = (h / 2.) * torch.einsum('ijq->ij', torch.unsqueeze(nodal_gradbasisEval, 0) * torch.unsqueeze(nodal_gradbasisEval, 1))

# %% Construct discretization for Poisson with Dirichlet BCs on left and right

# Set up forcing function evaluated on the nodes and specify dirichlet conditions
forcingvec = torch.ones_like(xq)
forcing = (h / 2.) * torch.sum(forcingvec * nodal_basisEval, dim=1)
uLHS = 0.0

# %% Build matrices
solution_rhs = torch.cat([torch.tensor([uLHS], dtype=torch.float64), forcing[1:meshsize]], dim=0)
solution_mat = torch.cat([
    torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(0), meshsize).double(), 0),
    Snodal[1:meshsize, :]
], dim=0)


# %% Solve the linear system and plot solution
u_sol = torch.linalg.solve(solution_mat, solution_rhs)
uexact = finepoints.numpy()*(-0.5*finepoints.numpy()+1.0)
plt.plot(points.numpy(), u_sol.numpy(),'.--',label='Computed',markersize=20)
plt.plot(finepoints.numpy(), uexact,label='Exact')
plt.legend()
plt.title("Solution")
plt.show()


# %%
