import numpy as np
from .mdpKernel import MdpKernel

class MdpSolverDP:
    def __init__(self, params):
        self.mdpKernel = MdpKernel(params)  # Not shown here
        self.N_markov = self.mdpKernel.N_markov
        self.N_markov_grouped = self.mdpKernel.N_markov_grouped
        self.N_Ax = self.mdpKernel.N_Ax
        self.N_Ay = self.mdpKernel.N_Ay
        self.max_cost_x = params['max_cost_x']
        self.max_cost_y = params['max_cost_y']
        self.LEN_horizon = params['LEN_horizon']

        self.V = []
        self.c_schdule_x = []
        self.c_schdule_y = []
        self.policy_x = []
        self.policy_y = []

        self.loss_table = []
        self.L_table = []
        self.M_table = []
        
        self.initialize()

    def initialize(self):
        self.V_r = np.zeros((self.LEN_horizon+1, self.N_markov))
        self.V_c = np.zeros((self.LEN_horizon+1, self.N_markov))
        for s in range(self.N_markov):
            self.V_r[s] = self.mdpKernel.lossFunction(s, 0, 0)

        c_schdule_x = self.max_cost_x * np.ones((self.LEN_horizon,))
        c_schdule_y = self.max_cost_y * np.ones((self.LEN_horizon,))
        self.updateCost(c_schdule_x, c_schdule_y)

        self.policy_x = np.zeros((self.LEN_horizon, ))
        self.policy_y = np.zeros((self.LEN_horizon, self.N_markov_grouped))

        for s in range(self.N_markov):
            for x in range(self.N_Ax):
                for y in range(self.N_Ay):
                    self.loss_table[s, x, y] = self.mdpKernel.lossFunction(s, x, y)

    def updateCost(self, c_schdule_x, c_schdule_y):
        self.c_schdule_x = c_schdule_x
        self.c_schdule_y = c_schdule_y

    def greedyXAtT(self, t):
        L = np.zeros((self.N_markov, ))
        M = np.zeros((self.N_markov, self.N_markov))
        for x in range(self.N_Ax):
            for s in range(self.N_markov):
                L[s] = self.mdpKernel.mdp_obv[s, :] @ self.loss_table[s, x, self.policy_y[t, :]]
                for s_next in range(self.N_markov):
                    M[s, s_next] = self.mdpKernel.mdp_obv[s, :] @ self.mdpKernel.getMdpTrans[s, s_next, x, self.policy_y[t, :]]

                       

    def updateX(self):
        for t in range(self.LEN_horizon-1,-1,-1):
            pass