# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %% Construct model
class FiniteDifferenceOperator(nn.Module):
    def __init__(self,Nleft,Nright,dx):
        super(FiniteDifferenceOperator, self).__init__()
        # Number of total nodes in finite difference stencils
        self.Nstencil = Nleft + Nright + 1
        self.Nleft = Nleft
        self.Nright = Nright

        # Define a learnable finite difference stencil with zero sum        
        self.stencil1 = torch.nn.Parameter(torch.tensor([1,-1],dtype=torch.float64))
        self.stencil2 = -self.stencil1.sum()
        self.stencil = torch.cat((self.stencil1, self.stencil2.unsqueeze(0)))
        # Define a MLP to model the nonlinearity
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).to(dtype=torch.float64)

        self.h = dx

    def forward(self, xx):
        # Apply the finite difference stencil to the gridfunction x, assuming periodic BC
        N_nodes = xx.shape[0]
        xx[-1] = xx[0]

        # Recompute stencil from stencil1 and stencil2
        self.stencil2 = -self.stencil1.sum()
        self.stencil = torch.cat((self.stencil1, self.stencil2.unsqueeze(0)))

        
        # Goal - build up D^* grad(N[D x])

        # Step 1 - apply D stencil to x 
        Dh_x = torch.zeros_like(x)
        for i in range(N_nodes-1):
          # Wrap indices periodically using the modulo operator (%)
          indices = [(i + j - self.Nleft) % (N_nodes-1) for j in range(self.Nstencil)]
          # Grab solution at indices
          xstencil = xx[indices]
          # Apply learned stencil to xstencil
          Dh_x[i] = torch.sum(self.stencil * xstencil)/self.h
        Dh_x[-1] = Dh_x[0]
        
        # Step 2 - calculate gradient of mlp applied to Dh_x
        N_of_Dx = 1.0+self.mlp(Dh_x.unsqueeze(1))
        grad_N_of_Dx = torch.autograd.grad(N_of_Dx.sum(), Dh_x, create_graph=True)[0]
        nonlinearity = N_of_Dx.squeeze(-1)*grad_N_of_Dx

        # Step 3 - apply D* stencil to grad_N_of_Dx
        Dstar_grad_N_of_Dx = torch.zeros_like(x)
        for i in range(N_nodes-1):
          # Wrap indices periodically using the modulo operator (%)
          indices = [(i + j - self.Nleft) % (N_nodes-1) for j in range(self.Nstencil)]

          # Grab solution at indices
          grad_N_of_Dx_stencil = nonlinearity[indices]

          # Apply learned stencil to grad_N_of_Dx_stencil
          Dstar_grad_N_of_Dx[i] = torch.sum(torch.flip(self.stencil,[0]) * grad_N_of_Dx_stencil)/self.h


        Dstar_grad_N_of_Dx[-1] = Dstar_grad_N_of_Dx[0]
        
        # Return result, which is the gradient of the potential in the Lagrangian
        return Dstar_grad_N_of_Dx*self.h

# %% Confirm implementation by checking action against a single fourier mode
#        --> if the nonlinearity is turned off and the stencil is set to [1,-1],
#            the operator should be the same as the second derivative operator

# Parameters
L = 1.0  # Length of the domain
nx = 40  # Number of spatial points

# Discretization
x = torch.from_numpy(np.linspace(0, L, nx,dtype=np.float64))
dx = x[1]-x[0]
dt = 0.25*dx

Dx = FiniteDifferenceOperator(1,1,dx)  # Finite difference operator w a neighbor on either side
def getData(x,k):
  return torch.sin(2.*np.pi*k*x/L)
u = getData(x,1.0)
dxxu = Dx(u).detach().numpy()
h=dx.numpy()
plt.plot(x,u.detach().numpy())

plt.plot(x,-dxxu/(4.0*np.pi**2)/h)
plt.show()

# %% Generate training data

# Parameters
c = 1.0  # Wave speed
L = 1.0  # Length of the domain
nx = 40  # Number of spatial points

# Discretization
x = torch.from_numpy(np.linspace(0, L, nx,dtype=np.float64))
dx = x[1]-x[0]
dt = 0.25*dx                # Set compatible w CFL condition
T = 3.0                     # Total time for training data
nt = int(T/dt)              # Number of time steps
t = torch.from_numpy(np.linspace(0, T, nt))

def f(x):
    #Initial condition
    return np.sin(2.*np.pi*x/L)
def fprime(x):
    #Time derivative of initial condition
    return (2.*np.pi/L)*np.cos(2.*np.pi*x/L)

q = np.zeros((nt,nx))
p = np.zeros((nt,nx))
for i in range(nt):
    fl = f(x-c*t[i])
    fr = f(x+c*t[i])
    q[i,:] = (fl+fr)
    fpl = -c*fprime(x-c*t[i])
    fpr =  c*fprime(x+c*t[i])
    p[i,:] = (fpl+fpr)*dx

q = torch.from_numpy(q)
p = torch.from_numpy(p)

# %% Visualize solution to q in spacetime as a 3d plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
XX, TT = np.meshgrid(x.numpy(), t.numpy())
ax.plot_surface(XX, TT, q.numpy(), cmap='viridis')
plt.show()


# %% Initialize model
Dx = FiniteDifferenceOperator(1,1,dx)  # Finite difference operator w a neighbor on either side
optimizer = optim.Adam(Dx.parameters(), lr=1e-3)
# Print parameters before optimization
print("Parameters before optimization:")
for name, param in Dx.named_parameters():
    print(f"{name}: {param.data}")

# %% Train model

losslog = []
batchsize = 5

# Generate loss function comparison predicted timestep to actual timestep
for epoch in range(10000):


    # choose a random subtrajectory of batchsize length
    start = np.random.randint(0, len(t)-batchsize)
    end = start + batchsize
    tbatch = t[start:end]
    qbatch = q[start:end]
    pbatch = p[start:end]

    # Evaluate current solution using the data-driven model
    solq = torch.zeros_like(qbatch)
    solp = torch.zeros_like(pbatch)
    solq[0, :] = qbatch[0, :]
    solp[0, :] = pbatch[0, :]
    
    #Integrate model over minibatch using Verlet scheme
    for step in range(1, len(tbatch)):
        qold = solq[step-1,:]
        pold = solp[step-1,:]
        
        # Verlet step 1: pnplushalf = pn - k/2 partial_q V(qn)
        partialqV = Dx(qold)
        pnplushalf = pold - 0.5 * dt *  partialqV 

        # Verlet step 2: qnplusone = qn + k partial_p T(pnplushalf)
        partialpT = pnplushalf
        qnplusone = qold + dt * partialpT / dx

        # Verlet step 3: pnplusone = pnplushalf - k/2 partial_q V(qnplusone)
        partialqV = Dx(qnplusone)
        pnplusone = pnplushalf - 0.5 * dt *  partialqV

        # Save solution
        solq[step, :] = qnplusone
        solp[step, :] = pnplusone


    # Compute loss
    lossq = (solq - qbatch).pow(2).sum()
    lossp = (solp - pbatch).pow(2).sum()
    loss = lossq + lossp
    lossrelq = lossq/(solq.pow(2).sum())
    losslog.append(lossq.item())
    # Clear the gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        # Print parameters after optimization
        print('Epoch:', epoch, 'Loss:', loss.item(), 'Relative Loss:', lossrelq.item(), Dx.stencil1)

        


# %%
# Roll out learned model
nrollout = 2
tro = np.linspace(0, nrollout*T, nrollout*nt+1)

qdata = q
pdata = p
solq = torch.zeros(nrollout*nt+1,nx)
solp = torch.zeros(nrollout*nt+1,nx)
solq[0, :] = qdata[0,:]
solp[0, :] = pdata[0,:]
for step in range(1, len(tro)):
    if step%50==0:
        print('Step:', step)
    qold = solq[step-1,:]
    pold = solp[step-1,:]
    
    # Verlet step 1: pnplushalf = pn - k/2 partial_q V(qn)
    partialqV = Dx(qold)
    pnplushalf = pold - 0.5 * dt * partialqV 

    # Verlet step 2: qnplusone = qn + k partial_p T(pnplushalf)
    partialpT = pnplushalf
    qnplusone = qold + dt * partialpT / dx

    # Verlet step 3: pnplusone = pnplushalf - k/2 partial_q V(qnplusone)
    partialqV = Dx(qnplusone)
    pnplusone = pnplushalf - 0.5 * dt * partialqV

    # Save solution
    solq[step, :] = qnplusone
    solp[step, :] = pnplusone

# %%
# Plot the results
plt.figure()
tcomp = q.shape[0]-1
tcomp=50
# plt.plot(x, q[0, :], label='Initial condition')
plt.plot(x, q[tcomp, :], label='True solution')
plt.plot(x, solq[tcomp, :].detach().numpy(), label='Learned solution')
plt.legend()
plt.show()
plt.figure()

# %% Visualize solution in spacetime as a 3D plot
XX, TT = np.meshgrid(x.numpy(), tro)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XX, TT, solq.detach().numpy(), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
plt.title('Solution to Wave Equation')
plt.show()

# %% Visualize as a contour plot
plt.figure()
plt.contourf(XX,TT,solq.detach().numpy())
plt.xlabel('x')
plt.ylabel('t')
plt.title('Solution to Wave Equation')
plt.colorbar()
plt.show()

# %%
