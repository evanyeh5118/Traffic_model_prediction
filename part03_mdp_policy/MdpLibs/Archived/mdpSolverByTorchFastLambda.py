import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .mdpKernel import MdpKernel


###############################################################################
# The solver class that performs the DP or other MDP logic
###############################################################################
class MdpSolverStochasticByTorch:
    def __init__(self, params, device):
        self.mdpKernel = MdpKernel(params)
        self.N_markov = self.mdpKernel.N_markov
        self.N_markov_grouped = self.mdpKernel.N_markov_grouped
        self.N_Ax = self.mdpKernel.N_Ax
        self.N_Ay = self.mdpKernel.N_Ay
        self.max_cost_x = params['max_cost_x']
        self.max_cost_y = params['max_cost_y']
        self.LEN_horizon = params['LEN_horizon']
        #lagrange_lambda = params['lagrange_lambda']
        self.device = device
        
        # Pre-build cost schedules as constant Tensors (no gradients needed).
        c_schdule_x = self.max_cost_x * torch.ones((self.LEN_horizon,), dtype=torch.float32, device=self.device)
        c_schdule_y = self.max_cost_y * torch.ones((self.LEN_horizon,), dtype=torch.float32, device=self.device)
        self.updateCost(c_schdule_x, c_schdule_y)
        self.initialize()
    
    def initialize(self):
        # If not already built, precompute external loss E[s, x, y]:
        self.E_torch = torch.empty(self.N_markov, self.N_Ax, self.N_Ay, dtype=torch.float32, device=self.device)
        for s in range(self.N_markov):
            for x in range(self.N_Ax):
                for y in range(self.N_Ay):
                    self.E_torch[s, x, y] = self.mdpKernel.lossFunction(s, x, y)

        # Convert transitions to a PyTorch Tensor
        self.T_torch = torch.tensor(self.mdpKernel.getMdpTrans, dtype=torch.float32, device=self.device)
        self.obv_torch = torch.tensor(self.mdpKernel.getMdpObv, dtype=torch.float32, device=self.device)

    def updateCost(self, c_schdule_x, c_schdule_y):
            (self.c_schdule_x, self.c_schdule_y) = (c_schdule_x, c_schdule_y)

    def getRewardsAndCosts(self, t, s, x, y):
        traffic, AoS = self.mdpKernel.fromMarkovVariableToStates(s)
        u = self.mdpKernel.utilityTable[traffic, x, y]
        (cx, cy) = (self.c_schdule_x[t]*x, self.c_schdule_y[t]*y)
        return (u, AoS, cx.cpu().detach().numpy(), cy.cpu().detach().numpy())

    def computeJ(self, policy_x, policy_y, pi_0=None):
        # 1) Build pi_0 if needed
        if pi_0 is None:
            pi_0 = torch.zeros(self.N_markov, dtype=torch.float32, device=self.device)
            pi_0[0] = 1.0
        else:
            pi_0 = torch.tensor(pi_0, dtype=torch.float32, device=self.device)

        V_r = self.UpdateRewardValueTable(policy_x, policy_y)
        V_c = self.UpdateCostValueTable(policy_x, policy_y)
        J_r = pi_0 @ V_r[0]
        J_c = pi_0 @ V_c[0]

        return J_r, J_c

    def UpdateRewardValueTable(self, policy_x, policy_y):
        V =  torch.zeros(self.LEN_horizon+1, self.N_markov, dtype=torch.float32, device=self.device)
        V[self.LEN_horizon] = torch.tensor(
            [self.mdpKernel.lossFunction(s, 0, 0) for s in range(self.N_markov)],
            dtype=torch.float32, device=self.device
        )
        Phi = torch.zeros(self.LEN_horizon, self.N_markov, self.N_Ax, self.N_Ay, dtype=torch.float32, device=self.device)
        T_4d = self.T_torch.permute(0, 2, 3, 1)  # shape: (N_markov, N_Ax, N_Ay, N_markov)
        for t in range(self.LEN_horizon-1, -1, -1):
            # -- Compute Phi[t] = sum_{s'} T_4d[s, x, y, s'] * V[t, s']
            Phi[t] = (T_4d * V[t+1].view(1, 1, 1, -1).to(self.device)).sum(dim=-1)
            A_t = self.E_torch + Phi[t]  # shape [N_markov, N_Ax, N_Ay]
            B_t = (A_t * policy_x[t].view(1, -1, 1)).sum(dim=1)  # B_t => shape [N_markov, N_Ay]
            C_t = B_t.matmul(policy_y[t])  # => shape [N_markov, N_markov_grouped]
            # Finally multiply elementwise by obv_torch[s], then sum => shape [N_markov]
            V[t] = (C_t * self.obv_torch).sum(dim=1)
        return V

    def UpdateCostValueTable(self, policy_x, policy_y):
        V =  torch.zeros(self.LEN_horizon+1, self.N_markov, dtype=torch.float32, device=self.device)
        Phi = torch.zeros(self.LEN_horizon, self.N_markov, self.N_Ax, self.N_Ay, dtype=torch.float32, device=self.device)
        T_4d = self.T_torch.permute(0, 2, 3, 1)  # shape: (N_markov, N_Ax, N_Ay, N_markov)
        x_range = torch.arange(self.N_Ax, dtype=torch.float32, device=self.device)
        y_range = torch.arange(self.N_Ay, dtype=torch.float32, device=self.device)
        for t in range(self.LEN_horizon-1, -1, -1):
            cost_xy = (self.c_schdule_x[t] * x_range.unsqueeze(1)
                    + self.c_schdule_y[t] * y_range.unsqueeze(0))
            # -- Compute Phi[t] = sum_{s'} T_4d[s, x, y, s'] * V[t, s']
            Phi[t] = (T_4d * V[t+1].view(1, 1, 1, -1)).sum(dim=-1)
            A_t = cost_xy + Phi[t]  # shape [N_markov, N_Ax, N_Ay]
            B_t = (A_t * policy_x[t].view(1, -1, 1)).sum(dim=1)  # B_t => shape [N_markov, N_Ay]
            C_t = B_t.matmul(policy_y[t])  # => shape [N_markov, N_markov_grouped]
            # Finally multiply elementwise by obv_torch[s], then sum => shape [N_markov]
            V[t] = (C_t * self.obv_torch).sum(dim=1)
        return V


###############################################################################
# Main nn.Module wrapper
###############################################################################
class MdpTorchKernel(nn.Module):
    def __init__(self, params, device):
        super(MdpTorchKernel, self).__init__()
        self.mdpSolver = MdpSolverStochasticByTorch(params, device)
        self.N_markov = self.mdpSolver.mdpKernel.N_markov
        self.N_markov_grouped = self.mdpSolver.mdpKernel.N_markov_grouped
        self.N_Ax = self.mdpSolver.mdpKernel.N_Ax
        self.N_Ay = self.mdpSolver.mdpKernel.N_Ay
        self.LEN_horizon = params['LEN_horizon']
        self.device = device

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

    def getRewardsAndCosts(self, t, s, x, y):
        return self.mdpSolver.getRewardsAndCosts(t, s, x, y)

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

    def forward(self, lagrange_lambda, pi_0=None):
        """
        Standard forward pass: 
          1) Convert self.policy_var -> (policy_x, policy_y)
          2) Compute J via the MdpSolver
        """
        policy_x, policy_y = self.fromVariableToPolicy()
        J_r, J_c = self.mdpSolver.computeJ(policy_x, policy_y, pi_0)
        J_total = J_r + lagrange_lambda*J_c
        return J_total, J_r, J_c

