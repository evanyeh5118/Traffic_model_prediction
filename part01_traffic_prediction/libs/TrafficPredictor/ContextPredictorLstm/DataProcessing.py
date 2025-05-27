import numpy as np

def PreparingDatasetHelper(dataUnit, params):
    #numWindow = params['numWindow']
    lenSource = params['lenSource']
    lenTarget = params['lenTarget']
    dataAugment = params['dataAugment'] # True;False
    smoothFc = params['smoothFc']
    smoothOrder = params['smoothOrder']

    lenDataset = dataUnit.dataLength
    contextData = dataUnit.getContextDataProcessedAndSmoothed(smoothFc, smoothOrder)
    
    sources, targets = [], []
    if dataAugment == True:
        idxs = [i for i in range(lenSource, lenDataset - lenTarget)]
    else:
        idxs = [i*lenTarget for i in range(int(lenSource/lenTarget)+1, int(np.floor(lenDataset / lenTarget)))]
        
    for i in idxs:
        sources.append(contextData[i-lenSource:i])
        targets.append(contextData[i:i+lenTarget])
    #sources = np.concatenate(sources, axis=0)
    #targets = np.concatenate(targets, axis=0)
    return (
        np.array(sources), 
        np.array(targets)
    )

def PreparingDataset(dataUnit, parameters, verbose=True):
    trainRatio = parameters['trainRatio']

    train_size = int(trainRatio*dataUnit.dataLength)
    dataUnitTrain = dataUnit[:train_size]
    dataUnitTest = dataUnit[train_size:]

    if verbose == True:
        print(f"Train size: {dataUnitTrain.dataLength}, Test size: {dataUnitTest.dataLength}")

    return (
        PreparingDatasetHelper(dataUnitTrain, parameters),
        PreparingDatasetHelper(dataUnitTest, parameters)
    )


