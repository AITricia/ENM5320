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

# Visualize shape functions and their derivatives evaluated over a fine grid
# phi_i = evalPhi_i(finepointsx,pointsx)
psi0_ij = torch.einsum('iq,jq->ijq', evalPhi_i(pointsx,pointsx), evalPhi_i(pointst,pointst))
gradpsi_ij = torch.einsum('iq,jq->ijq', evalGradPhi_i(pointsx,pointsx), evalPhi_i(pointst,pointst))
psi1_ijkl = torch.einsum('ijq,ijq->ijq', psi0_ij, gradpsi_ij)-torch.einsum('ijq,ijq->ijq', gradpsi_ij, psi0_ij)

# %%
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

#to integrate \int f(x,y) dx dy
integrand = evalIntegrand(xq)
integral = (h / 2.) * torch.sum(integrand, dim=1)

#stiffness matrix of P1 basis functions
nodal_gradbasisEval = evalGradPhi_i(xq)
Snodal = (h / 2.) * torch.einsum('ijq->ij', torch.unsqueeze(nodal_gradbasisEval, 0) * torch.unsqueeze(nodal_gradbasisEval, 1))

# %% Construct discretization for Poisson with Dirichlet BCs on left and right

# Set up forcing function evaluated on the nodes and specify dirichlet conditions
forcing = torch.ones(meshsize, dtype=torch.float64)
uLHS = 1.0
uRHS = 0.0

#Build matrices
solution_rhs = torch.cat([torch.tensor([uLHS], dtype=torch.float64), forcing[1:meshsize-1], torch.tensor([uRHS], dtype=torch.float64)], dim=0)
solution_mat = torch.cat([
    torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(0), meshsize).double(), 0),
    Snodal[1:meshsize-1, :],
    torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(meshsize-1), meshsize).double(), 0)
], dim=0)


# %% Solve the linear system and plot solution
u_sol = torch.linalg.solve(solution_mat, solution_rhs)
plt.plot(points.numpy(), u_sol.numpy())
plt.title("Solution")
plt.show()

