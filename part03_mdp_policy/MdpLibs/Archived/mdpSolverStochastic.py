from ..mdpKernel import MdpKernel
import numpy as np

class MdpSolverStochastic:
    def __init__(self, params):
        self.mdpKernel = MdpKernel(params)
        self.N_markov = self.mdpKernel.N_markov
        self.N_markov_grouped = self.mdpKernel.N_markov_grouped
        self.N_Ax = self.mdpKernel.N_Ax
        self.N_Ay = self.mdpKernel.N_Ay 
        self.max_cost_x = params['max_cost_x']
        self.max_cost_y = params['max_cost_y']
        self.LEN_horizon = params['LEN_horizon']
        
        # ======================================
        # ========= Policy =========
        # ======================================
        self.policy_x = []
        self.policy_y = []
        self.lagrange_lambda = []
        # ======================================
        # ==========Kernel Dependency===========
        # ======================================
        self.phi_vector = []
        self.Phi = []
        self.V = []

        self.initialize()

    def initialize(self):
        self.policy_x = np.zeros((self.LEN_horizon, self.N_Ax))
        self.policy_y = np.zeros((self.LEN_horizon, self.N_Ay, self.N_markov_grouped))
        self.lagrange_lambda  = 0.0
        #self.phi_vector = np.zeros(())
        self.Phi = np.zeros((self.LEN_horizon, self.N_markov, self.N_Ax, self.N_Ay))
        self.L = np.zeros((self.LEN_horizon, self.N_markov, self.N_Ax, self.N_Ay))
        self.V = np.zeros((self.LEN_horizon+1, self.N_markov)) #Only Value function consider T, e.g V(T)
        self.initialV_T()
        self.updateCostTable()
        self.computePhiMatrix()
        self.computeLMatrix()

    def immediateLossFunction(self, t, s, x, y):
        return self.mdpKernel.lossFunction(s, x, y) + self.lagrange_lambda*self.costFunction(t, x, y)
    
    def initialV_T(self):
        for s in range(self.N_markov):
            self.V[self.LEN_horizon, s] = self.mdpKernel.lossFunction(s, 0, 0)
        
    def updateCostTable(self):
        self.c_schdule_x = self.max_cost_x*np.ones((self.LEN_horizon, ))
        self.c_schdule_y = self.max_cost_y*np.ones((self.LEN_horizon, ))

    def costFunction(self, t, x, y):
        return self.c_schdule_x[t]*x + self.c_schdule_y[t]*y

    def computeJ(self, pi_0):
        if len(pi_0) != self.N_markov:
            raise ValueError("length of initial distribution is not matched the Markov model")
        for t in range(self.LEN_horizon):
            for s in range(self.N_markov):
                self.V[t, s] = (
                    self.policy_x[t,:] @ (self.L[t,s] + self.Phi[t,s]) @ 
                    self.policy_y[t, :, :] @ self.mdpKernel.getMdpObv[s, :]
                )
        J = np.array(pi_0).T @ self.V[0]
        return J

    def computePhiMatrix(self):
        for t in range(self.LEN_horizon):
            for s in range(self.N_markov):
                for x in range(self.N_Ax):
                    for y in range(self.N_Ay):
                        self.Phi[t, s, x, y] =  (
                            self.mdpKernel.getMdpTrans[s, :, x, y] @ self.V[t,:]
                        )
        
    def computeLMatrix(self):
        for t in range(self.LEN_horizon):
            for s in range(self.N_markov):
                for x in range(self.N_Ax):
                    for y in range(self.N_Ay):
                        self.L[t, s, x, y] =  (
                            self.immediateLossFunction(t, s, x, y)
                        )
    '''
    def initialPolicy(self, type):
        if type == "zeros":
            self.policy_x = np.zeros((self.LEN_horizon, self.N_Ax))
            self.policy_y = np.zeros((self.LEN_horizon, self.N_Ay, self.N_markov_grouped))
        else:
            self.policy_x = np.random.uniform(low=0.0, high=1.0, size=(self.LEN_horizon, self.N_Ax))
            self.policy_y = np.random.uniform(low=0.0, high=1.0, size=(self.LEN_horizon, self.N_Ay))
        return self.policy_x, self.policy_y
    '''