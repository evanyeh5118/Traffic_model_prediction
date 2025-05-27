import torch
import numpy as np
from ..HelperFunctions import createDataLoaders

def evaluateModel(model, validData, batch_size=4096):
    source_data, target_data = validData
    validation_loader = createDataLoaders(batch_size, (source_data, target_data), shuffle=False)

    actual_traffic = []
    predicted_traffic = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    for i, (batch_sources, batch_target) in enumerate(validation_loader): 
        sources = batch_sources.permute(1, 0, 2).to(device)
        targets = batch_target.permute(1, 0, 2).to(device)
        outputs = model(src=sources, trg=targets, teacher_forcing_ratio=0.0)

        targets = targets.permute(1, 0, 2).cpu().detach().numpy()
        outputs = outputs.permute(1, 0, 2).cpu().detach().numpy()

        actual_traffic.append(targets)
        predicted_traffic.append(outputs)

    #actual_traffic = np.concatenate(actual_traffic)
    #predicted_traffic = np.concatenate(predicted_traffic)
    return actual_traffic, predicted_traffic
