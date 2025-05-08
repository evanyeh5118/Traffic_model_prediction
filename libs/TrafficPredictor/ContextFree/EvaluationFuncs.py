import torch
import numpy as np
from ..HelperFunctions import createDataLoaders

def evaluateModel(model, validData, batch_size=4096):
    source_data, target_data = validData
    validation_loader = createDataLoaders(batch_size, (source_data, target_data), shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    for i, (batch_sources, batch_target) in enumerate(validation_loader): 
        sources = batch_sources.permute(1, 0, 2).to(device)
        targets = batch_target.permute(1, 0, 2).to(device)
        outputs = model(src=sources, trg=targets, teacher_forcing_ratio=0.0)

        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        if i == 0:
            actual_traffic = targets
            predicted_traffic = outputs
        else:
            actual_traffic = np.append(actual_traffic, targets, axis=1)
            predicted_traffic = np.append(predicted_traffic, outputs, axis=1)

    actual_traffic = np.concatenate(actual_traffic)
    predicted_traffic = np.concatenate(predicted_traffic)
    return actual_traffic, predicted_traffic
