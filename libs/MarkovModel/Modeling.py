import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)

def compute_context_aware_transition_matrix(resultsTrain, lenWindow, alpha=1e-20):
    classDistribu_predicted = resultsTrain["classDistribu_predicted"]
    trafficSource = resultsTrain["trafficSource_actual"]

    P = np.zeros((lenWindow + 1, lenWindow + 1))
    traffic_count = np.zeros((lenWindow + 1, 1))

    for i in range(trafficSource.shape[0] - 1):
        traffic = int(trafficSource[i, 0])
        P[traffic] += softmax(classDistribu_predicted[i, :])
        traffic_count[traffic] += 1

    for i in range(lenWindow + 1):
        P[i, :] /= (traffic_count[i] + 1)

    P = P + alpha
    P = P / P.sum(axis=1, keepdims=True)

    return P, traffic_count

def compute_transition_matrix(x_t_minus_1, x_t, L, alpha=1e-20):
    x_t_minus_1 = x_t_minus_1.astype(int)
    x_t = x_t.astype(int)
    P = np.zeros((L , L ))

    for i, j in zip(x_t_minus_1, x_t):
        P[i, j] += 1

    # Normalize rows to get probabilities
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(P, row_sums, where=row_sums != 0)  # avoid division by zero

    P = P + alpha
    P = P / P.sum(axis=1, keepdims=True)

    return P

def compute_log_likelihood_and_perplexity(M, x_test0, x_test1, N=10):
    x_test0 = x_test0.astype(int)
    x_test1 = x_test1.astype(int)

    #M_smooth = M + alpha
    #M_smooth = M_smooth / M_smooth.sum(axis=1, keepdims=True)

    transitions = list(zip(x_test0, x_test1))
    chunks = np.array_split(transitions, N)

    loglike_list = []
    perplexity_list = []

    for chunk in chunks:
        log_likelihood = 0.0
        for i, j in chunk:
            prob = max(M[i, j], 1e-12)
            log_likelihood += np.log(prob)
        perplexity = np.exp(-log_likelihood / len(chunk))

        loglike_list.append(log_likelihood)
        perplexity_list.append(perplexity)

    return loglike_list, perplexity_list

