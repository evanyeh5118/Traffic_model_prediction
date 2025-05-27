import torch
import numpy as np
import matplotlib.pyplot as plt

from .MdpSolver import MdpSolverTorchInterface

'''
"lenWindow": self.lenWindow,
"trafficTrans":self.trafficTrans,
"P_vEst_given_v":self.P_vEst_given_v,
"thereshold":self.threshold,
"numAggregatedTraffic": self.numAggregatedTraffic,
"numTraffic":self.numTraffic,
"trafficData":self.trafficData,
"trafficDataEst":self.trafficDataEst,
"trafficDataAggregated":self.trafficDataAggregated,
"trafficDataAggregatedEst":self.trafficDataAggregatedEst
'''

def defaultMdpParams(aggregatedMdpModel):
    mdp_params = {}
    mdp_params['LEN_horizon'] = 50
    mdp_params['AoS_th'] = 5
    mdp_params['gamma'] = 1.2
    mdp_params['N_Ax'] = 10
    mdp_params['N_Ay'] = 15
    mdp_params['max_Ax'] = 500
    mdp_params['max_Ay'] = 300
    mdp_params['max_cost_x'] = 1
    mdp_params['max_cost_y'] = 1
    mdp_params['utility_ref'] = 0.97
    mdp_params['utility_weight'] = 0.99
    mdp_params['budget_discount'] = 0.2
    mdp_params['B'] = mdp_params['max_cost_x']*mdp_params['LEN_horizon']*mdp_params['N_Ax']*mdp_params['budget_discount']
    #Normalize cost by budget
    mdp_params['max_cost_x'] = mdp_params['max_cost_x']/mdp_params['B']
    mdp_params['max_cost_y'] = mdp_params['max_cost_y']/mdp_params['B']
    
    return mdp_params

def optimizeMixedTimePolicy(aggregatedMdpModel, trainPrams, mdpParams=None, verbose=False):
    trafficData = aggregatedMdpModel['trafficData']
    if mdpParams is None:
        mdpParams = defaultMdpParams(aggregatedMdpModel)
    
    mdpParams['aggregate_thresholds'] = aggregatedMdpModel['thereshold']
    mdpParams['N_traffic'] = aggregatedMdpModel['numTraffic']
    mdpParams['N_traffic_grouped'] = aggregatedMdpModel['numAggregatedTraffic']
    mdpParams['T_traffic'] = aggregatedMdpModel['trafficTrans']
    mdpParams['P_vEst_given_v'] =aggregatedMdpModel['P_vEst_given_v']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdpSolverTorch = MdpSolverTorchInterface(mdpParams, device).to(device)

    initialD = getInitialDistribution(trafficData, mdpSolverTorch.N_markov, mdpParams['N_traffic'])
    mdpSolverTorchOpt, J_history, J_r_history, J_c_history, lambda_history = optimizationPolicy(
        mdpSolverTorch, initialD, trainPrams, verbose=verbose)

    return mdpSolverTorchOpt, mdpParams, J_history, J_r_history, J_c_history, lambda_history

def optimizationPolicy(mdpSolverTorch, initialD, trainPrams, verbose=False):
    optimizer = torch.optim.Adam(mdpSolverTorch.parameters(), lr=0.01)

    accumulation_steps = trainPrams['accumulation_steps']
    lagrange_lambda = trainPrams['lagrange_lambda']
    lr_lambda = trainPrams['lr_lambda']
    N_iter = trainPrams['N_iter']
    lambda_reset_threshold =  trainPrams['lambda_reset_threshold']
    grad_lambda = 0
    B_real = 1.0
    J_r_best = 10000

    J_history =  []
    J_r_history =  []
    J_c_history =  []
    lambda_history =  []
    mdpSolverTorch_best = None
    for step in range(N_iter):     
        optimizer.zero_grad()
        J, J_r, J_c = mdpSolverTorch(lagrange_lambda, pi_0=initialD)  # forward
        J = J / accumulation_steps
        J.backward() # backprop  
        grad_lambda += (J_c.item() - B_real)/ accumulation_steps

        # ===== Update =====
        if (step + 1) % accumulation_steps == 0:  # Update every `accumulation_steps`
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Reset gradients
            if J_c.item() < lambda_reset_threshold :
                lagrange_lambda = 0
                #if verbose == True: print("Lambda reset")
            else:
                lagrange_lambda = np.clip(lagrange_lambda + lr_lambda*grad_lambda, 0, np.inf)

        grad_lambda = 0

        if  mdpSolverTorch_best is None:
            mdpSolverTorch_best = mdpSolverTorch
        # ====== update the best policy ======
        if  J_r.item() < J_r_best and J_c.item() < B_real:
            #if verbose == True: print("Update")
            J_r_best =  J_r.item()
            mdpSolverTorch_best = mdpSolverTorch

        J_history.append(J.item())
        J_r_history.append(J_r.item())
        J_c_history.append(J_c.item())
        lambda_history.append(lagrange_lambda)
        if verbose == True:
            if step%(N_iter/10) == 0: 
                print(f"Step {step}, J={J.item()}, J_r={J_r.item()}, J_c={J_c.item()}, lambda:{lagrange_lambda}")
                
    print(f"Step {step}, J={J.item()}, J_r={J_r.item()}, J_c={J_c.item()}, lambda:{lagrange_lambda}")
    return mdpSolverTorch_best, J_history, J_r_history, J_c_history, lambda_history

def getInitialDistribution(traffic_data, N_markov, N_traffic):
    freqs_traffic = np.ones((N_traffic, ))
    for t in traffic_data:
        freqs_traffic[int(t)] += 1
    freqs_traffic = freqs_traffic/np.sum(freqs_traffic)

    pi_0 = np.zeros((N_markov, ))
    N_AoS = int(N_markov/N_traffic)
    for i in range(N_traffic):
        pi_0[i*N_AoS:(i+1)*N_AoS] = freqs_traffic[i]/N_AoS

    return pi_0

def visualizeUtilityTable(mdpSolverTorch, dsiplay_step=5):
    utilityTable = mdpSolverTorch.mdpSolver.mdpKernel.utilityTable
    r_list_x = mdpSolverTorch.mdpSolver.mdpKernel.r_list_x
    r_list_y = mdpSolverTorch.mdpSolver.mdpKernel.r_list_y
    # Create index arrays
    i = np.arange(len(r_list_x))
    j = np.arange(len(r_list_y))
    I, J = np.meshgrid(i, j, indexing='ij')  # Meshgrid for indices

    # Plot the 3D mesh
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for n in range(1, utilityTable.shape[0], dsiplay_step):
        ax.plot_surface(I, J, utilityTable[n, :, :], cmap='viridis', edgecolor='k')  # 3D surface

    # Labels
    ax.set_xlabel('Index i')
    ax.set_ylabel('Index j')
    ax.set_zlabel('Value x[i,j]')
    ax.set_title('3D Mesh Plot')
