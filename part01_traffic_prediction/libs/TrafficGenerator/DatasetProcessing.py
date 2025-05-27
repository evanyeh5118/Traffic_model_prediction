import pandas as pd
import numpy as np

from .DatasetReader import DatasetReader
from .DeadbandReduction import DataReductionForDataUnit
from .Dataunit import DataUnit

class DatasetConvertor:
    def __init__(self, rawDatasetFolder):
        self.rawDatasetFolder = rawDatasetFolder
        self.dfRaw = []
        self.fingerDataUnits = []
        self.idxsContextForward = []
        self.idxsContextBackward = []
        self.datasetReader = []
        
        self.initialize()

    def initialize(self):
        self.configuration()
        self.updateRawDataset(self.rawDatasetFolder)
        self.seperateFingersDataByDirections()
        
    def configuration(self, idxsContextForward=None, idxsContextBackward=None):
        if idxsContextForward is None:
            self.idxsContextForward = {
                "thumb": [1, 2, 3],
                "index": [5, 6, 7],
                "middle": [9, 10, 11],
                #"palm": [13, 14, 15],
            }
        if idxsContextBackward is None:
            self.idxsContextBackward = {
                "thumb": [4],
                "index": [8],
                "middle": [12],
            }

    def getDataUnit(self, unitName):
        return self.fingerDataUnits[unitName]

    def processDataset(self, dbParameter=0.01, alpha=0.01, mode="fixed", direction="forward"):
        for fingerName in ['thumb', 'index', 'middle']:
            print(f"========== {fingerName} ============")
            if direction == "forward":
                self.fingerDataUnits[f"{fingerName}_fr"].resampleContextData()
                self.fingerDataUnits[f"{fingerName}_fr"].applyDpDr(dbParameter=dbParameter, alpha=alpha, mode=mode)
                self.fingerDataUnits[f"{fingerName}_fr"].interpolateCotextAfterDpDr()
                compressRate = self.fingerDataUnits[f"{fingerName}_fr"].compressionRate
                print(f"Forward: Compression rate:{compressRate}")
            else:
                self.fingerDataUnits[f"{fingerName}_bk"].resampleContextData()
                self.fingerDataUnits[f"{fingerName}_bk"].applyDpDr(dbParameter=dbParameter, alpha=alpha, mode=mode)
                self.fingerDataUnits[f"{fingerName}_bk"].interpolateCotextAfterDpDr()
                compressRate = self.fingerDataUnits[f"{fingerName}_bk"].compressionRate
                print(f"Backward: Compression rate:{compressRate}")
            
    def seperateFingersDataByDirections(self):
        self.fingerDataUnits = {}
        for fingerName, idxsContext in self.idxsContextForward.items():
            dataUnit = DataUnit()
            dataUnit.name = fingerName
            dataUnit.setContextData(self.dfRaw.iloc[:, idxsContext].to_numpy())
            dataUnit.timestamps = self.dfRaw.iloc[:, 0].to_numpy()
            self.fingerDataUnits[f"{fingerName}_fr"] = dataUnit

        for fingerName, idxsContext in self.idxsContextBackward.items():
            dataUnit = DataUnit()
            dataUnit.name = fingerName
            dataUnit.setContextData(self.dfRaw.iloc[:, idxsContext].to_numpy())
            dataUnit.timestamps = self.dfRaw.iloc[:, 0].to_numpy()
            self.fingerDataUnits[f"{fingerName}_bk"] = dataUnit

    def updateRawDataset(self, rawDatasetFolder):
        self.datasetReader = DatasetReader()
        self.rawDatasetFolder = rawDatasetFolder
        self.datasetReader.readRawDataset(self.rawDatasetFolder)
        #self.dfRaw = self.datasetReader.dfRawNormal
        self.dfRaw = self.datasetReader.dfRaw
