import numpy as np
import pandas as pd
import math
import random

#import torch

from ..HelperFunctions import SmoothFilter, FindLastTransmissionIdx, DiscretizedTraffic


def PreparingDatasetHelper(dataUnit, params):
    #numWindow = params['numWindow']
    lenSource = params['lenSource']
    lenTarget = params['lenTarget']
    dataAugment = params['dataAugment'] # True;False
    smoothFc = params['smoothFc']
    smoothOrder = params['smoothOrder']

    lenDataset = dataUnit.dataLength
    #contextData = dataUnit.getContextDataProcessed()
    contextData = dataUnit.getContextDataProcessedAndSmoothed(smoothFc, smoothOrder)
    transmissionFlags = dataUnit.getTransmissionFlags()

    sources, targets, lastTranmittedContext, transmissionsVector, trafficStatesSource, trafficStatesTarget = [], [], [], [], [], []
    if dataAugment == True:
        idxs = [(i, FindLastTransmissionIdx(transmissionFlags, i)) 
                for i in range(lenSource, lenDataset - lenTarget)]
    else:
        idxs = [(i * lenTarget, FindLastTransmissionIdx(transmissionFlags, i * lenTarget)) 
                for i in range(int(lenSource/lenTarget)+1, int(np.floor(lenDataset / lenTarget)))]
        
    for i, last_transmission_idx in idxs:
        #sources.append(contextData[i-lenSource:i])
        sources.append(contextData[i:i+lenTarget])
        targets.append(contextData[i:i+lenTarget])
        transmissionsVector.append(transmissionFlags[i:i+lenTarget])
        trafficStatesSource.append(np.sum(transmissionFlags[i-lenSource:i]))
        trafficStatesTarget.append(np.sum(transmissionFlags[i:i+lenTarget]))
        lastTranmittedContext.append(contextData[last_transmission_idx:last_transmission_idx+1])
        
    trafficClassesTarget = DiscretizedTraffic(trafficStatesTarget) #[0 ~ L]
    return (
        np.array(sources), 
        np.array(targets),
        np.array(lastTranmittedContext),
        np.array(trafficStatesSource).reshape(-1,1),
        np.array(trafficStatesTarget).reshape(-1,1),
        np.array(trafficClassesTarget).reshape(-1,1),
        np.array(transmissionsVector)
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





