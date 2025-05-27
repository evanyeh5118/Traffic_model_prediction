import torch
import numpy as np
from ..HelperFunctions import createDataLoaders

def evaluateModel(model, validData, batch_size=4096):
    validation_loader = createDataLoaders(batch_size, validData, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    
    actual_traffic, predicted_traffic = [], []
    for batch in validation_loader: 
        sources, targets = (data.permute(1, 0, 2).to(device) for data in batch)
        outputs = model(src=sources, trg=targets, teacher_forcing_ratio=0.0)

        actual_traffic.append(targets.permute(1, 0, 2).cpu().detach().numpy())
        predicted_traffic.append(outputs.permute(1, 0, 2).cpu().detach().numpy())

    actual_traffic = np.concatenate(actual_traffic, axis=0)
    predicted_traffic = np.concatenate(predicted_traffic, axis=0)
    return actual_traffic.reshape(-1,), predicted_traffic.reshape(-1,)


def evaluateModelTest(validData, batch_size=4096):
    validation_loader = createDataLoaders(batch_size, validData, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actual_traffic = []
    for batch in validation_loader: 
        _, targets = (data.permute(1, 0, 2).to(device) for data in batch)
        actual_traffic.append(targets.permute(1, 0, 2).cpu().detach().numpy())
   
    actual_traffic = np.concatenate(actual_traffic, axis=0)
    return actual_traffic.reshape(-1,)