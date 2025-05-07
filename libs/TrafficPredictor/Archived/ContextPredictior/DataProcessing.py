import numpy as np

def PreparingDataset(dataUnit, parameters, verbose=True):
    trainRatio = parameters['trainRatio']
    dataAugment = parameters['dataAugment']
    fc = parameters['fc']
    order = parameters['order']

    train_size = int(trainRatio*dataUnit.dataLength)
    contextData = min_max_normalize(dataUnit.getContextDataProcessedAndSmoothed(fc, order))
    contextDataTrain = contextData[:train_size].copy()
    contextDataTest = contextData[train_size:].copy()

    dataTrain = PreparingDatasetHelper(contextDataTrain, parameters, dataAugment)
    dataTest = PreparingDatasetHelper(contextDataTest, parameters, dataAugment)

    if verbose == True:
        print(f"Train size: {len(contextDataTrain)}->{dataTrain[0].shape[0]}," + 
                f"Test size: {len(contextDataTest)}->{dataTest[1].shape[0]}")
  
    return (dataTrain, dataTest)


def PreparingDatasetHelper(contextData, parameters, dataAugment):
    lenSource = parameters['lenSource'] 
    lenTarget = parameters['lenTarget'] 

    lenDataset = len(contextData)
    dataSource = []
    dataTarget = []

    if dataAugment == True:
        idxs = [i for i in range(lenSource, lenDataset - lenTarget)]
    else:
        idxs = [i * lenTarget for i in range(int(lenSource/lenTarget)+1, int(np.floor(lenDataset / lenTarget)))]
    
    for i in idxs:
        row_source = contextData[i-lenSource:i]
        row_target = contextData[i:i+lenTarget]
        dataSource.append(row_source)
        dataTarget.append(row_target)
    #return np.expand_dims(np.array(dataSource), -1), np.expand_dims(np.array(dataTarget),-1)
    return np.array(dataSource), np.array(dataTarget)

def min_max_normalize(x):
    """
    Apply min-max normalization to each dimension (column) of the input array.

    Parameters:
        x (np.ndarray): A 2D NumPy array of shape (len, dim)

    Returns:
        np.ndarray: Normalized array with values in [0, 1] per dimension
    """
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    # Avoid division by zero for constant columns
    denominator = x_max - x_min
    denominator[denominator == 0] = 1.0
    return (x - x_min) / denominator