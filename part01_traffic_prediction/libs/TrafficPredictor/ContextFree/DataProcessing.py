import numpy as np

def PreparingDataset(dataUnit, parameters, verbose=True):
    trainRatio = parameters['trainRatio']

    train_size = int(trainRatio*dataUnit.dataLength)
    dataUnitTrain = dataUnit[:train_size]
    dataUnitTest = dataUnit[train_size:]

    if verbose == True:
        print(f"Train size: {dataUnitTrain.dataLength}, Test size: {dataUnitTest.dataLength}")

    dataTrain = PreparingDatasetHelper(dataUnitTrain, parameters)
    dataTest = PreparingDatasetHelper(dataUnitTest, parameters)
    if parameters['dataAugment'] == True:
        dataTrainAugmented = dataTrain
        dataTestAugmented = dataTest
        for i in range(parameters['lenWindow']-1):
            dataUnitTrain = dataUnitTrain[1:]
            dataUnitTest = dataUnitTest[1:]
            dataTrainNew = PreparingDatasetHelper(dataUnitTrain, parameters)
            dataTestNew = PreparingDatasetHelper(dataUnitTest, parameters)
            dataTrainAugmented = concateDataset(dataTrainAugmented, dataTrainNew)
            dataTestAugmented = concateDataset(dataTestAugmented, dataTestNew)
        dataTrain = dataTrainAugmented
        dataTest = dataTestAugmented

    return (dataTrain, dataTest)

def concateDataset(datasetA, datasetB):
    if len(datasetA) != len(datasetB):
        raise ValueError("Datasets must have the same length")
    
    datasetC = tuple(np.concatenate((a, b), axis=0) for a, b in zip(datasetA, datasetB))
    return datasetC


def ConvertToTrafficState(transmitionFlags, parameters):
    lenWindow = parameters['lenWindow']

    N_slot = int(np.floor((transmitionFlags.shape[0])/lenWindow))
    windowTraffic = []
    for i in range(N_slot-1):
        traffic_state = np.sum(transmitionFlags[i*lenWindow:(i+1)*lenWindow])
        windowTraffic.append(traffic_state)
    return np.array(windowTraffic)


def PreparingDatasetHelper(dataUnit, parameters):
    lenSource = parameters['lenSource'] 
    lenTarget = parameters['lenTarget'] 
    transmitionFlags = dataUnit.getTransmissionFlags()
    windowTraffic = ConvertToTrafficState(transmitionFlags, parameters)
    dataSource = []
    dataTarget = []
    
    for i in range(lenSource, windowTraffic.shape[0]-lenTarget):
        row_source = windowTraffic[i-lenSource:i]
        row_target = windowTraffic[i:i+lenTarget]
        dataSource.append(row_source)
        dataTarget.append(row_target)
    return np.expand_dims(np.array(dataSource), -1), np.expand_dims(np.array(dataTarget),-1)
