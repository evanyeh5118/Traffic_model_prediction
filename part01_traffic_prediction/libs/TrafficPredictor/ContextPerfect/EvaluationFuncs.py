import torch
import torch.optim as optim
import numpy as np

from .TrafficPredictor import TrafficPredictorContextAssisted, CustomLossFunction
from ..HelperFunctions import createDataLoaders, countModelParameters

def evaluateModel(traffic_predictor, test_data, batch_size=4096, verbose=True):
    # Create data loader
    validation_loader = createDataLoaders(batch_size=batch_size, dataset=test_data, shuffle=False)
    
    # Determine computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Storage for results
    transmissions_actual, transmissions_predicted = [], []
    classDistribu_actual, classDistribu_predicted = [], []
    trafficSource_actual = []
    trafficTarget_actual, trafficTarget_predicted = [], []
    #actual_context, predicted_context = [], []

    # Iterate over validation data
    for batch in validation_loader:
        sources, targets, last_trans_sources, trafficsSource, trafficsTarget, classesDistribu, transmissions = (data.to(device) for data in batch)

        # Permute tensor dimensions for model input compatibility
        sources, targets, last_trans_sources = map(lambda x: x.permute(1, 0, 2), (sources, targets, last_trans_sources))

        # Get model predictions
        pred_trafficTarget, pred_classDistribu, pred_transmissions, _ = traffic_predictor(sources, last_trans_sources)

        # Store results
        transmissions_actual.append(transmissions.cpu().detach().numpy())
        transmissions_predicted.append(pred_transmissions.cpu().detach().numpy())

        classDistribu_actual.append(classesDistribu.cpu().detach().numpy())
        classDistribu_predicted.append(pred_classDistribu.cpu().detach().numpy())

        trafficSource_actual.append(trafficsSource.cpu().detach().numpy())
        trafficTarget_actual.append(trafficsTarget.cpu().detach().numpy())
        trafficTarget_predicted.append(pred_trafficTarget.cpu().detach().numpy())

    # Concatenate results from all batches
    transmissions_actual = np.concatenate(transmissions_actual)
    transmissions_predicted = np.concatenate(transmissions_predicted)
    classDistribu_actual = np.concatenate(classDistribu_actual)
    classDistribu_predicted = np.concatenate(classDistribu_predicted)
    trafficSource_actual = np.concatenate(trafficSource_actual)
    trafficTarget_actual = np.concatenate(trafficTarget_actual)
    trafficTarget_predicted = np.concatenate(trafficTarget_predicted)

    results = {
        'transmissions_actual': transmissions_actual,
        'transmissions_predicted': transmissions_predicted,
        'classDistribu_actual': classDistribu_actual,
        'classDistribu_predicted': classDistribu_predicted,
        'trafficSource_actual': trafficSource_actual,
        'trafficTarget_actual': trafficTarget_actual,
        'trafficTarget_predicted': trafficTarget_predicted,
    }

    return results
