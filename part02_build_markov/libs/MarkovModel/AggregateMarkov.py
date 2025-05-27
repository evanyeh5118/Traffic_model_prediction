import numpy as np
import pickle
from .Modeling import compute_context_aware_transition_matrix

class AggregateMarkov:
    def __init__(self):
        pass
        
    def registerTrafficData(self, trafficSource, trafficTargetDistribution, trafficTarget_actual, trafficTarget_predicted, lenWindow):
        self.trafficSource = trafficSource.astype(int)
        self.trafficTargetDistribution = trafficTargetDistribution
        self.trafficTarget_actual = trafficTarget_actual.astype(int)
        self.trafficTarget_predicted = trafficTarget_predicted.astype(int)
        self.lenWindow = lenWindow
        self.numTraffic = lenWindow+1
        self.trafficTrans, _ = compute_context_aware_transition_matrix(self.trafficSource, self.trafficTargetDistribution, self.lenWindow)

    def updateAggregateModel(self, numAggregatedTraffic):
        self.numAggregatedTraffic = min(numAggregatedTraffic, self.numTraffic)
        self.threshold = getAggregateThereshold(self.trafficTarget_actual, numAggregatedTraffic)
        self.trafficDataAggregated = aggregateFunc(self.trafficTarget_actual, self.threshold )
        self.trafficDataAggregatedEst = aggregateFunc(self.trafficTarget_predicted, self.threshold )
        self.P_vEst_given_v = estimateMatrixTrafficState(self.trafficTarget_actual, self.trafficDataAggregatedEst, self.numTraffic, self.numAggregatedTraffic)

    def saveResult(self, fileName):
        results = {
            "lenWindow": self.lenWindow,
            "trafficTrans":self.trafficTrans,
            "P_vEst_given_v":self.P_vEst_given_v,
            "thereshold":self.threshold,
            "numAggregatedTraffic": self.numAggregatedTraffic,
            "numTraffic":self.numTraffic,
            "trafficData":self.trafficTarget_actual,
            "trafficDataEst":self.trafficTarget_predicted,
            "trafficDataAggregated":self.trafficDataAggregated,
            "trafficDataAggregatedEst":self.trafficDataAggregatedEst
        }
        with open(fileName, "wb") as file:
            pickle.dump(results, file)
        # Close the pickle file
        file.close()

    def validResults(self):
        for i in range(self.trafficTrans.shape[0]):
            if abs(np.sum(self.trafficTrans[i,:])) > 1e-6:
                return False
        for i in range(self.P_vEst_given_v.shape[0]):
            if abs(np.sum(self.P_vEst_given_v[i,:])) > 1e-6:
                return False
        return True

    def display(self):
        print(f"lenWindow:{self.lenWindow}\n")

def computeMarkovTransition(data, K):
    data = np.asarray(data, dtype=int)
    if np.any(data < 0) or np.any(data >= K):
        raise ValueError(f"All states must be in the range [0, {K-1}].")
    
    T_counts = np.zeros((K, K), dtype=float)
    for i in range(len(data) - 1):
        curr_state = data[i]
        next_state = data[i + 1]
        T_counts[curr_state, next_state] += 1
    
    row_sums = T_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = T_counts / row_sums
        T[np.isnan(T)] = 0.0
    
    return T

def aggregateFunc(s, thresholds):
    group_class = np.zeros_like(s, dtype=int)
    for i, threshold in enumerate(thresholds):
        group_class[s >= threshold] = i + 1

    return group_class

def getAggregateThereshold(arr, N_state):
    # Compute the frequency distribution
    unique_elements, counts = np.unique(arr, return_counts=True)
    cumulative_counts = np.cumsum(counts)
    total_count = cumulative_counts[-1]

    # Determine thresholds to divide the data into N_state groups with similar distribution
    thresholds = []
    target_count_per_group = total_count / N_state

    current_target = target_count_per_group
    for i in range(len(cumulative_counts)):
        if cumulative_counts[i] >= current_target:
            thresholds.append(unique_elements[i] + 1)
            current_target += target_count_per_group

            if len(thresholds) == N_state - 1:
                break

    return thresholds


def estimateMatrixTrafficState(v, vEst, N_v, N_vEst):
    """
    Computes the conditional probability matrix O(v | vEst).
    """
    # Initialize the joint counts array: rows index v, columns index vEst
    counts = np.zeros((N_v, N_vEst), dtype=np.float64)
    
    # Tally the occurrences of each pair (v[i], vEst[i])
    for vi, vgivei in zip(v, vEst):
        # Safety check: skip if out of expected range
        if 0 <= vi <= N_v and 0 <= vgivei <= N_vEst:
            counts[vi, vgivei] += 1
    
    # Sum each column separately
    counts = counts.T
    col_sums = counts.sum(axis=0, keepdims=True)
    
    # Conditional distribution: O[v, vEst] = counts[v, vEst] / sum over v of counts[v, vEst]
    # Avoid division by zero by setting those columns with 0 sum to 0. 
    O = np.divide(counts, col_sums, where=(col_sums!=0))
    O[:, col_sums[0]==0] = 1.0/N_v
    
    return O.T