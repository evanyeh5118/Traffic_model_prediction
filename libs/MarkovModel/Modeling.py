import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)


def compute_context_aware_transition_matrix(trafficSource, classDistribu_predicted, lenWindow, alpha=1e-20):
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

def compute_transition_matrix(x_t_minus_1, x_t, lenWindow, alpha=1e-20):
    L = lenWindow + 1
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

def generate_balanced_thresholds(arr, N):
    if N <= 1:
        raise ValueError("N must be greater than 1.")

    # Count the frequency of each unique value
    unique, counts = np.unique(arr, return_counts=True)
    freq_dict = dict(zip(unique, counts))

    # Sorting the unique values by their frequencies
    sorted_values = sorted(freq_dict.keys())
    total_samples = len(arr)

    # Initialize variables for threshold calculation
    thresholds = []
    cum_count = 0
    group_size = total_samples / N
    current_group_count = 0

    # Calculate N-1 thresholds
    for value in sorted_values:
        cum_count += freq_dict[value]
        current_group_count += freq_dict[value]

        # Check if the current group is full
        if current_group_count >= group_size:
            thresholds.append(value)
            current_group_count = 0

        # Adjust the group size dynamically to ensure N-1 thresholds
        remaining_groups = N - len(thresholds) - 1
        if remaining_groups > 0:
            group_size = (total_samples - cum_count) / remaining_groups

        # Stop if we reach exactly N-1 thresholds
        if len(thresholds) == N - 1:
            break

    # Ensure the final threshold list is exactly N-1
    while len(thresholds) < N - 1:
        thresholds.append(sorted_values[-1])

    return thresholds


def assign_groups(arr, thresholds):
    group_arr = np.zeros_like(arr)
    for i, thres in enumerate(thresholds):
        group_arr[arr > thres] = i + 1

    return group_arr



