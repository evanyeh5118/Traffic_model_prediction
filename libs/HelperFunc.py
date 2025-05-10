import re
import torch.nn as nn
import torch
import numpy as np

def encode_float_filename(number: float) -> str:
    """
    Encodes a float into a filename-safe string by combining zero-padding and scientific notation.
    - If number is between 0.01 and 9999, use fixed-point format with zero-padding.
    - Otherwise, use scientific notation with a safe separator.
    """
    if 1 <= abs(number) < 0.1:
        formatted = f"{number:08.2f}".replace('.', '_')  # Zero-padding, 2 decimal places
    else:
        formatted = f"{number:.2e}".replace('.', '_').replace('+', '')  # Scientific notation
    
    return f"{formatted}.txt"

def decode_float_filename(filename: str) -> float:
    """
    Decodes a filename back into a float by identifying the encoding method used.
    - Detects scientific notation and fixed-point encoding.
    """
    match = re.search(r"([0-9eE_\-]+)\.txt", filename)
    if not match:
        raise ValueError("Invalid filename format")

    encoded_number = match.group(1)
    
    # Check if it's scientific notation (contains 'e' or 'E')
    if 'e' in encoded_number or 'E' in encoded_number:
        decoded = float(encoded_number.replace('_', '.'))
    else:
        decoded = float(encoded_number.replace('_', '.'))  # Convert back to float
    
    return decoded

def compute_f1_scores(predictions: np.ndarray, ground_truth: np.ndarray, num_classes: int):
    """
    Computes the F1-Score for each class in a multi-class classification problem.

    Parameters:
    - predictions (np.ndarray): Array of predicted class labels.
    - ground_truth (np.ndarray): Array of actual class labels.
    - num_classes (int): Total number of classes.

    Returns:
    - f1_scores (np.ndarray): Array of F1-Score values for each class.
    """
    # Initialize arrays for True Positives, False Positives, False Negatives
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    # Calculate TP, FP, FN for each class
    for i in range(num_classes):
        TP[i] = np.sum((predictions == i) & (ground_truth == i))
        FP[i] = np.sum((predictions == i) & (ground_truth != i))
        FN[i] = np.sum((predictions != i) & (ground_truth == i))

    # Calculate Precision, Recall, F1-Score for each class
    precision = TP / (TP + FP + 1e-10)  # Adding epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    return f1_scores

def compute_weighted_f1_score(predictions: np.ndarray, ground_truth: np.ndarray, num_classes: int):
    """
    Computes the Weighted-Averaged F1-Score for a multi-class classification problem.

    Parameters:
    - predictions (np.ndarray): Array of predicted class labels.
    - ground_truth (np.ndarray): Array of actual class labels.
    - num_classes (int): Total number of classes.

    Returns:
    - weighted_f1 (float): The Weighted-Averaged F1-Score.
    """
    # Initialize arrays for True Positives, False Positives, False Negatives
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    # Calculate TP, FP, FN for each class
    for i in range(num_classes):
        TP[i] = np.sum((predictions == i) & (ground_truth == i))
        FP[i] = np.sum((predictions == i) & (ground_truth != i))
        FN[i] = np.sum((predictions != i) & (ground_truth == i))

    # Calculate Precision, Recall, F1-Score for each class
    precision = TP / (TP + FP + 1e-10)  # Adding epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Calculate weighted average
    class_counts = np.array([np.sum(ground_truth == i) for i in range(num_classes)])
    weighted_f1 = np.sum(f1_scores * class_counts) / np.sum(class_counts)

    return weighted_f1

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
    group_index = 0

    # Calculate N-1 thresholds
    for value in sorted_values:
        cum_count += freq_dict[value]
        current_group_count += freq_dict[value]

        # Check if the current group is full
        if current_group_count >= group_size:
            thresholds.append(value)
            group_index += 1
            current_group_count = 0

            # If we have enough thresholds, stop
            if group_index == N - 1:
                break

    return thresholds


def assign_groups(arr, thresholds):
    group_arr = np.zeros_like(arr)
    for i, thres in enumerate(thresholds):
        group_arr[arr > thres] = i + 1

    return group_arr



