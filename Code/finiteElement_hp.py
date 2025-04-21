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
meshsize= 32
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
    return suppPhi * (signPlus - signNeg) / h

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

# %% Build POU coarsening
Npou_int = 2 #INTERNAL POUS
Wijlogit = torch.nn.Parameter(torch.randn(Npou_int, meshsize-2, dtype=torch.float64))
Wijlogit.requires_grad = True
# Wij = torch.nn.functional.softmax(Wijlogit+1e5*torch.eye(Npou), dim=0)
Wij = torch.nn.functional.softmax(Wijlogit, dim=0)
optimizer = optim.Adam([Wijlogit], lr=0.0001)


# %% Calculate Gauss-legendre weights and points of a given order
Nquad = 10
# polyOrder = 2
qpts, qws = np.polynomial.legendre.leggauss(Nquad)  # Use NumPy to compute
qpts = torch.tensor(qpts, dtype=torch.float64)
qws = torch.tensor(qws, dtype=torch.float64)
xq = (points[:-1].unsqueeze(-1) + 0.5*h + 0.5*h*qpts.unsqueeze(0)).flatten()
qws = (qws.unsqueeze(0).repeat(meshsize, 1)).flatten()  # Repeat weights for each element
indexsort = torch.argsort(xq)
xq = xq[indexsort]
qws = qws[indexsort]


# %% Train
Npou = Npou_int + 2


# %%
for i in range(100000):
    optimizer.zero_grad()
    for polyOrder in range(1,2):
        W_softmax = torch.softmax(Wijlogit, dim=0)
        W_toprow = torch.zeros(1, meshsize)
        W_toprow[0, 0] = 1.0
        W_bottomrow = torch.zeros(1, meshsize)
        W_bottomrow[0, -1] = 1.0

        W_middleblock = torch.cat(
            [
                torch.zeros(Npou_int, 1),
                W_softmax,
                torch.zeros(Npou_int, 1),
            ],
            dim=1,
        )
        Wij = torch.cat([W_toprow, W_middleblock, W_bottomrow], dim=0)

        W_middleblock = torch.cat(
            [
                torch.zeros(Npou_int, 1),
                W_softmax,
                torch.zeros(Npou_int, 1),
            ],
            dim=1,
        )
        Wij = torch.cat([W_toprow, W_middleblock, W_bottomrow], dim=0)
        
        # build pou basis and derivatives
        nodal_basisEval = evalPhi_i(xq)
        nodal_gradbasisEval = evalGradPhi_i(xq)
        pou_i = Wij @ nodal_basisEval
        gradpou_i = Wij @ nodal_gradbasisEval

        # build polynomial basis and derivatives
        # polyOrder = torch.randint(0,3,(1,))[0].item()
        N0forms = (polyOrder+1)*Npou
        poly_i = torch.stack([xq**i for i in range(polyOrder+1)])
        gradpoly_i = torch.stack([i*xq**(i-1) for i in range(polyOrder+1)])

        # build 0-forms
        psi0_i = torch.einsum('iq,jq->ijq', poly_i, pou_i).flatten(0,1)
        gradpsi0_i = torch.einsum('iq,jq->ijq', gradpoly_i, pou_i).flatten(0,1)+torch.einsum('iq,jq->ijq', poly_i, gradpou_i).flatten(0,1)
        psi1_ij = torch.einsum('iq,jq->ijq',psi0_i,gradpsi0_i)-torch.einsum('iq,jq->jiq',psi0_i,gradpsi0_i)

        M0 =  (h/2.)*torch.einsum('iq,jq,q->ij',psi0_i,psi0_i,qws)
        # M1 =  (h/2.)*torch.einsum('ijq,klq,q->ijkl',psi1_ij,psi1_ij,qws)
        S0 = -(h/2.)*torch.einsum('iq,jq,q->ij',gradpsi0_i,gradpsi0_i,qws)

        # Construct discretization for Poisson with Dirichlet BCs on left and right

        # Set up forcing function evaluated on the nodes and specify dirichlet conditions
        #pick random k betweren 1 and 3
        # k = np.random.randint(1, 3)
        k = 3
        uexact = torch.cos(2.0*np.pi*k*xq)
        dudxexact = -2.0*np.pi*k*torch.sin(2.0*np.pi*k*xq)
        forcing = -(4.0*np.pi**2*k**2)*torch.cos(2.0*np.pi*k*xq)
        
        # uexact = torch.ones_like(xq)+0.0*xq
        # dudxexact = 0.*torch.ones_like(xq)
        # forcing = -0.*torch.ones_like(xq)

        RHS = (h/2.)*torch.einsum('iq,q,q->i',psi0_i,forcing,qws)        
        uLHS = uexact[0]
        uLHS2 = uexact[-1]

        #  Build matrices
        solution_rhs = torch.cat([uLHS.unsqueeze(0), RHS[1:-1], uLHS2.unsqueeze(0)], dim=0)
        solution_mat = torch.cat([
            torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(0), N0forms).double(), 0),
            S0[1:-1, :],
            torch.unsqueeze(torch.nn.functional.one_hot(torch.tensor(N0forms-1), N0forms).double(), 0),
        ], dim=0)


        # Solve the linear system and plot solution
        u_sol = torch.linalg.lstsq(S0, RHS, rcond = 1e-9)[0]
        # u_sol = torch.linalg.solve(solution_mat, solution_rhs)
        
        # expand solution in basis on the quadrature points and shift to match constant offset
        u_sol_expanded = torch.einsum('iq,i->q',psi0_i,u_sol)
        gradu_sol_expanded = torch.einsum('iq,i->q',gradpsi0_i,u_sol)
        u_sol_expanded = u_sol_expanded + uexact[0] - u_sol_expanded[0]
        h1seminorm = torch.linalg.norm(gradu_sol_expanded - dudxexact, 2)
        # take gradient descent step to update wij
        error = u_sol_expanded - uexact
        loss = 1.0*torch.linalg.norm(error, 2) + 0.*h1seminorm
        print(f"Iteration {i+1}, Error: {loss.item()}, l2 norm: {torch.linalg.norm(error, 2).item()}, h1 seminorm: {h1seminorm.item()}")
        loss.backward(retain_graph=True)
    optimizer.step()

# %%
plt.plot(xq.numpy(), u_sol_expanded.detach().numpy(), label='Solution')
plt.plot(xq.numpy(), uexact.detach().numpy(), label='Solution')

# %%
plt.plot(xq.numpy(), gradu_sol_expanded.detach().numpy(), label='Solution')
plt.plot(xq.numpy(), dudxexact.detach().numpy(), label='Solution')

# %% visualize pous
plt.plot(points.numpy(), Wij.detach().numpy().T)
plt.title("POU basis functions")

# %%
