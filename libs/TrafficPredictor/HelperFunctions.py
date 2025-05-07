from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def SmoothFilter(df, fc, fs, order):
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    # Butterworth filter design
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = fc / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    # Apply the filter to each column
    filtered_data = df.apply(lambda col: filtfilt(b, a, col), axis=0)
    # Return the filtered data as a DataFrame with the same index and columns
    return pd.DataFrame(filtered_data, index=df.index, columns=df.columns)

def DiscretizedTraffic(data):    
    outputs = []
    for d in data:
        outputs.append(round(d))
    return np.array(outputs)

def FindLastTransmissionIdx(transmission, current_idx):
    while current_idx-1 >= 0:
        current_idx = current_idx-1
        if transmission[current_idx] == 1:
            return current_idx
    return 0

def createDataLoaders(batch_size, dataset, shuffle=True):
    # Convert all input data into tensors and stack them
    tensor_list = [torch.stack([torch.from_numpy(d).float() for d in data]) for data in dataset]
    
    num_samples = tensor_list[0].shape[0]
    assert all(t.shape[0] == num_samples for t in tensor_list), "All input tensors must have the same number of samples."
    
    # Create a dataset
    dataset = TensorDataset(*tensor_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return dataloader

# Calculate total parameters
def countModelParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)