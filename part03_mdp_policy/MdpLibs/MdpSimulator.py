import numpy as np

class ActionGenerator:
    def __init__(self, mdpSolverInterface, mdpParams, aggregatedMdpModel):
        self.mdpSolverInterface = mdpSolverInterface
        self.mdpParams = mdpParams
        policy_x, policy_y = self.mdpSolverInterface.fromVariableToPolicy()
        self.policy_x = policy_x.cpu().detach().numpy()
        self.policy_y = policy_y.cpu().detach().numpy()
        self.trafficData = aggregatedMdpModel['trafficData']

    def getAction(self, t, traffic, trafficEst, AoS, mode="mdp"):
        # mode = ["mdp", "const", "np"]
        if mode == "mdp":
            (x, y) = self.getMixedTimeAction(t, trafficEst, AoS)
        elif mode == "const":
            #(x, y) = (1, 0)
            (x, y) = (int(self.mdpParams['N_Ax']*self.mdpParams['budget_discount']), 0)
        elif mode == "np":
            actionHigh = (int(self.mdpParams['N_Ax']*self.mdpParams['budget_discount']), 1)
            actionLow = (2, 0)
            if t == 0:
                (x, y) = actionHigh
            else:
                if traffic >= np.mean(self.trafficData)*0.7:
                    (x, y) = actionHigh
                else:
                    (x,y) = actionLow
        else:
            raise ValueError("Not matching policy mode")
        return (x, y)
    
    def getMixedTimeAction(self, t, traffic, AoS):
        def getActionFromDistribution(actionDistribution):
            if not np.isclose(np.sum(actionDistribution), 1.0):
                raise ValueError("The distribution should sum to 1.")
            
            return np.random.choice(len(actionDistribution), p=actionDistribution)
        sEst = self.mdpSolverInterface.mdpSolver.mdpKernel.fromStatesToMarkovVariable(traffic, AoS)[0]
        return (
            getActionFromDistribution(self.policy_x[t, :]),
            getActionFromDistribution(self.policy_y[t, :, sEst])
        )


class PolicySimulator:
    def __init__(self, mdpSolverInterface, mdpParams, aggregatedMdpModel):
        self.aggregatedMdpModel = aggregatedMdpModel
        self.mdpParams = mdpParams
        self.mdpSolverInterface = mdpSolverInterface
        self.actionGenerator = ActionGenerator(mdpSolverInterface, mdpParams, aggregatedMdpModel) 

    def simulation(self, policyType="mdp"):
        #threshold = self.aggregatedMdpModel['thereshold']
        trafficData = self.aggregatedMdpModel['trafficData']
        trafficAggregated = self.aggregatedMdpModel['trafficDataAggregated']
        trafficAggregatedEst = self.aggregatedMdpModel['trafficDataAggregatedEst']
        numAggregatedTraffic = self.aggregatedMdpModel['numAggregatedTraffic']
        LEN_traffic = len(trafficData)
        
        AoS = 0
        utility_ref = self.mdpParams['utility_ref']
        LEN_horizon = self.mdpParams['LEN_horizon']

        N_window = int(np.floor(LEN_traffic/LEN_horizon))-1
        simTrafficEst_history = []
        simTraffic_history = []
        simU_history = []
        simAoS_history = []
        simS_history = []
        simX_history = []
        simY_history = []
        simCx_history = []
        simCy_history = []
        simCostWindow_history = []
        for i in range(N_window):
            trafficWindow = trafficData[i*LEN_horizon:(i+1)*LEN_horizon]
            trafficAggregatedWindow = trafficAggregated[i*LEN_horizon:(i+1)*LEN_horizon]
            trafficAggregatedEstWindow = trafficAggregatedEst[i*LEN_horizon:(i+1)*LEN_horizon]
            costWindow = 0
            for t in range(LEN_horizon):
                AosForPolicy = np.clip(AoS, 0, self.mdpParams['AoS_th'])
                (x, y) = self.actionGenerator.getAction(t, trafficWindow[t], trafficAggregatedEstWindow[t], AosForPolicy, mode=policyType)
                s = self.mdpSolverInterface.mdpSolver.mdpKernel.fromStatesToMarkovVariable(trafficWindow[t], AosForPolicy)
                (u, _, cx, cy) = self.mdpSolverInterface.mdpSolver.getRewardsAndCosts(t, s, x, y)
                if u >= utility_ref:
                    AoS = 0
                else:
                    AoS = AoS + 1
                
                costWindow += (cx + cy)
                simU_history.append(u)
                simAoS_history.append(AoS)
                simTraffic_history.append(trafficWindow[t])
                simTrafficEst_history.append(trafficAggregatedEstWindow[t]) 
                simS_history.append(s)
                simX_history.append(x)
                simY_history.append(y)
                simCx_history.append(cx)
                simCy_history.append(cy)
            simCostWindow_history.append(costWindow)

        simResults = {}     
        
        simResults['simU_history'] = simU_history
        simResults['simAoS_history'] = simAoS_history
        simResults['simTraffic_history'] = simTraffic_history
        simResults['simTrafficEst_history'] = simTrafficEst_history
        simResults['simS_history'] = simS_history
        simResults['simX_history'] = simX_history
        simResults['simY_history'] = simY_history
        simResults['simCx_history'] = simCx_history
        simResults['simCy_history'] = simCy_history
        simResults['simCostWindow_history'] = simCostWindow_history
        simResults['estAccuracy'] = calCccuracy(trafficAggregated, trafficAggregatedEst, numAggregatedTraffic)[0]
        
        # Convert all lists in simResults to numpy arrays
        simResults['simU_history'] = np.array(simResults['simU_history'])
        simResults['simAoS_history'] = np.array(simResults['simAoS_history']) 
        simResults['simTraffic_history'] = np.array(simResults['simTraffic_history'])
        simResults['simTrafficEst_history'] = np.array(simResults['simTrafficEst_history'])
        simResults['simS_history'] = np.array(simResults['simS_history'])
        simResults['simX_history'] = np.array(simResults['simX_history'])
        simResults['simY_history'] = np.array(simResults['simY_history'])
        simResults['simCx_history'] = np.array(simResults['simCx_history'])
        simResults['simCy_history'] = np.array(simResults['simCy_history'])
        simResults['simCostWindow_history'] = np.array(simResults['simCostWindow_history'])
        return simResults
    

def calCccuracy(data, dataEst, numElement):
    freqCount = np.zeros((numElement, ))
    
    if len(data) != len(dataEst):
        raise ValueError("data and dataEst must have the same length")
    
    correct_count = 0
    total_count = len(data)
    
    for i in range(total_count):
        if data[i] == dataEst[i]:
            correct_count += 1
        freqCount[data[i]] += 1  # Counting occurrences of each element in data

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, freqCount