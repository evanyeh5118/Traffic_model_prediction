
#from .utility_builder import satisfaction
import numpy as np
from .UtilityBuilder import createUtilityTable

class MdpKernel:
    def __init__(self, params):
        self.N_traffic = params['N_traffic'] #[0, MAX]
        self.N_traffic_grouped = params['N_traffic_grouped']
        self.MAX_arrival_packet = self.N_traffic-1
        self.AoS_th = params['AoS_th'] 
        self.N_aos = self.AoS_th+1
        self.N_markov = self.N_traffic*self.N_aos
        self.N_markov_grouped = self.N_traffic_grouped*self.N_aos
        self.N_Ax = params['N_Ax']
        self.N_Ay = params['N_Ay']
        self.max_Ax = params['max_Ax']
        self.max_Ay = params['max_Ay']
        self.max_cost_x = params['max_cost_x']
        self.max_cost_y = params['max_cost_y']
        self.utility_ref = params['utility_ref']
        self.T_traffic = np.array(params['T_traffic'])
        self.P_vEst_given_v = params['P_vEst_given_v']
        self.aggregate_thresholds = params['aggregate_thresholds'] 
        self.gamma = params['gamma'] 
        self.utility_weight = params['utility_weight']

        # Assigned after initializing
        self.r_list_x = []
        self.r_list_y = []
        self.c_schdule_x = []
        self.c_schdule_y = []
        self.utilityTable = []
        self.costTable = []
        self.mdp_trans = [] #(self.N_markov, self.N_markov, self.N_Ax, self.N_Ay)
        self.mdp_obv = []   #(self.N_markov, self.N_markov_grouped)
        self.initialize()

        # x: [0~N_Ax-1] 
        # y: [0~N_Ay-1]

    def initialize(self):
        self.buildUtilityTable()
        self.setMdpTrans()
        self.setMdpObv()
       
    def buildUtilityTable(self):
        r_params = (self.max_Ax, self.N_Ax, self.max_Ay, self.N_Ay)
        #Note that createUtilityTable only cosider the pacekt in [1, MAX_arrival_packet]
        #We add the case pacekt=0 manually, ane define the utility is always 1 no matter how much resource is allocated 
        self._utilityTable, self.r_list_x, self.r_list_y = createUtilityTable(
            self.MAX_arrival_packet, r_list_mixed=r_params)
        self.utilityTable = np.ones((1, self.N_Ax, self.N_Ay))
        self.utilityTable = np.concatenate((self.utilityTable, self._utilityTable), axis=0)

    def lossFunction(self, s, x, y):
        traffic, aos, = self.fromMarkovVariableToStates(s)
        #The AoS is in [0, self.AoS_th] --> self.N_aos
        retval = self.utility_weight*(1-self.utilityTable[traffic, x, y]) + ((self.gamma**(aos) - 1)/(self.gamma - 1))
        return retval
    #self.utilityTable[s, x, y] + lagrange_lambda*self.costFunction(t, x, y)
    
    @property
    def getMdpTrans(self):
        return self.mdp_trans.copy()
    @property
    def getMdpObv(self):
        return self.mdp_obv.copy()
    
    def getMdpTransHelper(self, s, s_next, x, y):
        traffic, aos = self.fromMarkovVariableToStates(s)
        traffic_next, aos_next = self.fromMarkovVariableToStates(s_next)
        if aos_next - aos > 1:
            return 0
        u = self.utilityTable[traffic, x, y]
        #print(f"u_ref:{self.utility_ref}, u: {u}, aos:{aos}, aos_next:{aos_next}")
        if u >= self.utility_ref:
            if aos_next == 0:
                return self.T_traffic[traffic,traffic_next]
        else: 
            if (aos_next == (aos + 1) or
                aos_next == aos and aos == self.AoS_th):
                return self.T_traffic[traffic,traffic_next]
        return 0
            

    def setMdpTrans(self):
        self.mdp_trans = np.zeros((self.N_markov, self.N_markov, self.N_Ax, self.N_Ay))
        # s = (traffic, AoS)
        # To build MarkovTrans, we need to assign the prob. according to the update rule of AoS
        for s in range(self.mdp_trans.shape[0]):
            for s_next in range(self.mdp_trans.shape[1]):
                # i -> j
                for x in range(self.N_Ax):
                    for y in range(self.N_Ay):
                        self.mdp_trans[s, s_next, x, y] = self.getMdpTransHelper(s, s_next, x, y)

    def setMdpObv(self):
        def aggregateMapping(traffic, threshold):
            for i, t in enumerate(threshold):
                if traffic < t:
                    return i
            return len(threshold)

        self.mdp_obv = np.zeros((self.N_markov, self.N_markov_grouped)) # true -> f(.) -> grouped -> grouped_est
        for s in range(self.mdp_obv.shape[0]):
            for s_group_obv in range(self.mdp_obv.shape[1]):
                traffic, aos = self.fromMarkovVariableToStates(s)
                traffic_group = aggregateMapping(traffic, self.aggregate_thresholds)
                traffic_group_obv, aos_group_obv = self.fromMarkovVariableToStates(s_group_obv)
                if aos == aos_group_obv:
                    self.mdp_obv[s, s_group_obv] = self.P_vEst_given_v[traffic_group, traffic_group_obv] 

   
    def fromMarkovVariableToStates(self, s):
        idx_traffic = np.floor(s / (self.N_aos))
        idx_AoS = np.mod(s, (self.N_aos))
        return int(idx_traffic), int(idx_AoS)

    def fromStatesToMarkovVariable(self, idx_traffic, idx_AoS):
        if (idx_traffic >= self.N_traffic) or (idx_AoS >= self.N_aos):
            raise ValueError("States out of bound") 
        return idx_traffic*self.N_aos + idx_AoS