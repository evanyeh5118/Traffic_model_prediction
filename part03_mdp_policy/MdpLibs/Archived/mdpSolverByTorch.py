import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .mdpKernel import MdpKernel


###############################################################################
# The solver class that performs the DP or other MDP logic
###############################################################################
class MdpSolverStochasticByTorch:
    def __init__(self, params):
        self.mdpKernel = MdpKernel(params)
        self.N_markov = self.mdpKernel.N_markov
        self.N_markov_grouped = self.mdpKernel.N_markov_grouped
        self.N_Ax = self.mdpKernel.N_Ax
        self.N_Ay = self.mdpKernel.N_Ay
        self.max_cost_x = params['max_cost_x']
        self.max_cost_y = params['max_cost_y']
        self.LEN_horizon = params['LEN_horizon']
        
        # We'll store policies only as arguments in computeJ (not in the solver).
        self.lagrange_lambda = torch.tensor(1.0, dtype=torch.float32)
        
        # Pre-build cost schedules as constant Tensors (no gradients needed).
        self.c_schdule_x = self.max_cost_x * torch.ones((self.LEN_horizon,), dtype=torch.float32)
        self.c_schdule_y = self.max_cost_y * torch.ones((self.LEN_horizon,), dtype=torch.float32)

    def costFunction(self, t, x, y):
        """
        Convert x,y to Tensor if needed (commonly x,y are discrete actions 
        or indices; might not truly require grad).
        """
        return (self.c_schdule_x[t] * torch.tensor(x, dtype=torch.float32)
              + self.c_schdule_y[t] * torch.tensor(y, dtype=torch.float32))

    def immediateLossFunction(self, t, s, x, y):
        """
        Convert the external loss function to a Torch Tensor.
        If mdpKernel.lossFunction(...) returns float, we wrap it in torch.tensor.
        """
        external_loss = torch.tensor(
            self.mdpKernel.lossFunction(s, x, y),
            dtype=torch.float32
        )
        return external_loss + self.lagrange_lambda * self.costFunction(t, x, y)

    def computeJ(self, policy_x, policy_y, pi_0=None):
        """
        Create fresh DP Tensors (V, L, Phi) for each pass.
        Then do your DP or whatever logic. Finally return the objective J.
        """
        # Default initial distribution as one-hot on state 0
        if pi_0 is None:
            pi_0 = torch.zeros((self.N_markov,), dtype=torch.float32)
            pi_0[0] = 1.0
        else:
            pi_0 = torch.tensor(pi_0, dtype=torch.float32)

        # Allocate fresh DP arrays for this forward pass
        V = torch.zeros((self.LEN_horizon+1, self.N_markov), dtype=torch.float32)
        L = torch.zeros((self.LEN_horizon, self.N_markov, self.N_Ax, self.N_Ay), 
                        dtype=torch.float32)
        Phi = torch.zeros((self.LEN_horizon, self.N_markov, self.N_Ax, self.N_Ay), 
                          dtype=torch.float32)

        # Initialize the terminal value function at time T
        # just as in your old code
        for s in range(self.N_markov):
            V[self.LEN_horizon, s] = torch.tensor(
                self.mdpKernel.lossFunction(s, 0, 0), dtype=torch.float32
            )

        # Compute immediate losses L[t, s, x, y]
        for t in range(self.LEN_horizon):
            for s in range(self.N_markov):
                for x in range(self.N_Ax):
                    for y in range(self.N_Ay):
                        L[t, s, x, y] = self.immediateLossFunction(t, s, x, y)

        # Compute Phi[t, s, x, y] = sum_{s'} P(s'|s,x,y) * V[t, s']
        # (assuming your transitions are from the same time t to next time t,
        #  though your original code doesn't do time-lag, so we replicate that logic.)
        for t in range(self.LEN_horizon):
            for s in range(self.N_markov):
                for x in range(self.N_Ax):
                    for y in range(self.N_Ay):
                        # getMdpTrans[s, :, x, y] is a np array => convert to torch
                        trans_row = torch.tensor(self.mdpKernel.getMdpTrans[s, :, x, y],
                                                 dtype=torch.float32)
                        Phi[t, s, x, y] = trans_row @ V[t, :]

        # Now fill V[t, s] via your policy and those tables
        # The code in your snippet does a forward iteration:
        for t in range(self.LEN_horizon):
            for s in range(self.N_markov):
                # (L[t, s] + Phi[t, s]) has shape [N_Ax, N_Ay].
                # policy_x[t, :] shape = [N_Ax]
                # policy_y[t, :, :] shape = [N_Ay, N_markov_grouped].
                # getMdpObv[s, :] shape = [N_markov_grouped].
                # The multiplication order in your snippet was:
                #
                #   V[t,s] = policy_x[t,:]
                #             @ (L[t,s] + Phi[t,s])
                #             @ policy_y[t,:,:]
                #             @ getMdpObv[s,:]
                #
                # We'll replicate that exactly:
                val = policy_x[t, :] @ (L[t, s] + Phi[t, s])  # => shape [N_Ay]
                val = val @ policy_y[t, :, :]                 # => shape [N_markov_grouped]
                val = val @ torch.tensor(self.mdpKernel.getMdpObv[s, :], dtype=torch.float32)
                V[t, s] = val

        # The final objective: do NOT detach V[0], so gradient can flow
        J = pi_0 @ V[0]  # shape: scalar
        return J


###############################################################################
# Main nn.Module wrapper
###############################################################################
class MdpTorchKernel(nn.Module):
    def __init__(self, params):
        super(MdpTorchKernel, self).__init__()
        self.mdpSolver = MdpSolverStochasticByTorch(params)
        self.N_markov = self.mdpSolver.mdpKernel.N_markov
        self.N_markov_grouped = self.mdpSolver.mdpKernel.N_markov_grouped
        self.N_Ax = self.mdpSolver.mdpKernel.N_Ax
        self.N_Ay = self.mdpSolver.mdpKernel.N_Ay
        self.LEN_horizon = params['LEN_horizon']

        self.initialPolicyVariable()

    def initialPolicyVariable(self):
        """
        Create a single trainable parameter that holds the flattened contents 
        of what were previously policy_x_varaible and policy_y_varaible.
        """
        # Suppose your initial (policy_x, policy_y) are zeros (or random).
        # We'll create random trainable parameters for the entire flatten.
        policy_x_numel = self.LEN_horizon * self.N_Ax               # shape [LEN_horizon, N_Ax]
        policy_y_numel = self.LEN_horizon * self.N_Ay * self.N_markov_grouped
        total_numel = policy_x_numel + policy_y_numel

        # A single parameter for everything
        self.policy_var = nn.Parameter(
            torch.randn(total_numel, dtype=torch.float32) * 0.01,
            requires_grad=True
        )

    def fromVariableToPolicy(self):
        """
        Given the single flattened self.policy_var, separate it out into policy_x
        and policy_y with their original shapes. Then apply softmax appropriately.
        """
        # We'll apply softmax along dim=1 for each time step
        softmax = nn.Softmax(dim=1)

        # Identify shapes
        policy_x_shape = (self.LEN_horizon, self.N_Ax)
        policy_y_shape = (self.LEN_horizon, self.N_Ay, self.N_markov_grouped)

        policy_x_numel = policy_x_shape[0] * policy_x_shape[1]
        # Alternatively, np.prod(policy_x_shape)

        # Slice out the piece for policy_x
        policy_x_flat = self.policy_var[:policy_x_numel]
        policy_x_raw = policy_x_flat.view(*policy_x_shape)

        # Slice out the piece for policy_y
        policy_y_flat = self.policy_var[policy_x_numel:]
        policy_y_raw = policy_y_flat.view(*policy_y_shape)

        # Apply softmax on the action dimension for each time step
        # For policy_x: shape [LEN_horizon, N_Ax], softmax along dim=1
        policy_x = softmax(policy_x_raw)

        # For policy_y: shape [LEN_horizon, N_Ay, N_markov_grouped]
        #  We'll do a softmax along dim=1 (the N_Ay dimension)
        #  i.e. for each t, s-group, we sum to 1 over the y dimension
        policy_y = softmax(policy_y_raw)

        return policy_x, policy_y

    def forward(self, pi_0=None):
        """
        Standard forward pass: 
          1) Convert self.policy_var -> (policy_x, policy_y)
          2) Compute J via the MdpSolver
        """
        policy_x, policy_y = self.fromVariableToPolicy()
        J = self.mdpSolver.computeJ(policy_x, policy_y, pi_0)
        return J

