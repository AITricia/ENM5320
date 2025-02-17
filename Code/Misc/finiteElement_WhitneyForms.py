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
meshsize= 10
h = 1./float(meshsize-1)
points = torch.tensor(np.linspace(0,1,meshsize), dtype=torch.float64)  # Replace [...] with actual values
finepoints = torch.tensor(np.linspace(0,1,20*meshsize), dtype=torch.float64)  # Replace [...] with actual values


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

# %% Build up Whitney forms
Nlevels = 3
Npous = 2**(Nlevels)

def binary_to_int(binary_str):
    return int(binary_str, 2)

def int_to_binary(integer, width):
    return format(integer, f"0{width}b")

print(int_to_binary(7, Nlevels))
# %%

class MLPWithSoftmax(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPWithSoftmax, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,-1)  # Apply softmax along the last dimension
        return x


# Example usage
input_dim = 1
hidden_dim = 10
model = MLPWithSoftmax(input_dim, hidden_dim)
input_data = torch.randn(4, input_dim)  # Batch size of 4
output_data = model(input_data)
output_data.shape

# Store indices across hierarchy levels
indexLevels = [[2**l+z-1 for z in range(2**(l))] for l in range(Nlevels)]
POUnets = [[MLPWithSoftmax(input_dim, hidden_dim)] for l in range(Nlevels)]


# %%
# Generate convex comb networks of same size as POUnets
whitney_0s = [[0 for _ in sublist] for sublist in POUnets]
whitney_0s[0][0] = POUnets[0][0]

#W_i1,...,iN = \prod_{l=1}^{N} W_{i_l|i_l-1,...,i_1}
for l in range(1,Nlevels):
    for i in range(2**l):
        print(f"Level {l}, index {i}")
        parents = int_to_binary(i, l)[::-1]
        print(parents)
        W = POUnets[0][0]
        for p in range(l):
            print(p,parents[p])
            W  = W*POUnets[p][parents[p]]
        whitney_0s[l][i] = W
# %%
