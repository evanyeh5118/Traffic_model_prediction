import torch
import torch.optim as optim
import numpy as np

from ..HelperFunctions import createDataLoaders

def evaluateModel(traffic_predictor, test_data, batch_size=4096, verbose=True):
    # Create data loader
    validation_loader = createDataLoaders(batch_size=batch_size, dataset=test_data, shuffle=False)
    
    # Determine computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Storage for results
    context_actual, context_predicted = [], []

    # Iterate over validation data
    for batch in validation_loader:
        sources, targets = (data.to(device) for data in batch)

        # Permute tensor dimensions for model input compatibility
        sources, targets = map(lambda x: x.permute(1, 0, 2), (sources, targets))

        # Get model predictions
        pred_context = traffic_predictor(sources, targets, teacher_forcing_ratio=0.0)

        # Store results
        context_actual.append(targets.permute(1, 0, 2).cpu().detach().numpy())
        context_predicted.append(pred_context.permute(1, 0, 2).cpu().detach().numpy())

    # Concatenate results from all batches
    context_actual = np.concatenate(context_actual)
    context_predicted = np.concatenate(context_predicted)

    results = {
        'context_actual': context_actual,
        'context_predicted': context_predicted 
    }

    return results
