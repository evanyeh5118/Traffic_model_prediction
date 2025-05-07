import torch
import numpy as np
from ..HelperFunctions import createDataLoaders

def evaluateModel(model, validData, batch_size=4096):
    source_data, target_data = validData
    validation_loader = createDataLoaders(batch_size, (source_data, target_data), shuffle=False)

    actualContext = []
    predictedContext = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    for i, (batch_sources, batch_target) in enumerate(validation_loader): 
        sources = batch_sources.permute(1, 0, 2).to(device)
        targets = batch_target.permute(1, 0, 2).to(device)
        outputs = model(src=sources, trg=targets, teacher_forcing_ratio=0.0)

        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()
        actualContext.append(targets)
        predictedContext.append(outputs)

    return actualContext, predictedContext
