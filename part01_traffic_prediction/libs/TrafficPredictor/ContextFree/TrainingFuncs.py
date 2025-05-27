import torch
import torch.nn as nn
import torch.optim as optim

from .Seq2seqModel import Seq2Seq, CustomLossFunction
from ..HelperFunctions import createDataLoaders, countModelParameters


def trainModelByDefaultSetting(lenSource, lenTarget, trainData, testData, verbose=False):
    (sources, targets) = trainData
    parameters = {
        "teacher_forcing_ratio" : 0.1,
        "batch_size": 4096,
        "hidden_size": 12,
        "num_layers": 6,
        "dropout_rate": 0.0,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "lenSource": lenSource,
        "lenTarget": lenTarget,
    }
    parameters['input_size'] = sources.shape[2]
    parameters['output_size'] = targets.shape[2]
    best_model, avg_train_loss_history, avg_test_loss_history = trainModel(parameters, trainData, testData, verbose=verbose)
    return best_model, avg_train_loss_history, avg_test_loss_history, parameters

def trainModel(parameters, trainData, testData, verbose=False):
    model, criterion, optimizer, train_loader, test_loader, device = prepareTraining(
        parameters, trainData, testData, verbose=verbose)

    best_model, avg_train_loss_history, avg_test_loss_history = trainModelHelper(
        parameters, model, criterion, optimizer, device, train_loader, test_loader, verbose=verbose)
    return best_model, avg_train_loss_history, avg_test_loss_history

def trainModelHelper(parameters, model, criterion, optimizer, device, train_loader, test_loader, verbose=False):
    num_epochs = parameters['num_epochs']
    teacher_forcing_ratio = parameters['teacher_forcing_ratio']
    #==============================================
    #============== Training ======================
    #==============================================

    bestWeight = None
    best_metric = float('inf')  # Set to a large value
    avg_train_loss_history = []
    avg_test_loss_history = []
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for source_train, target_train in train_loader:
            sources = source_train.permute(1, 0, 2).to(device) # Shape: (sequence_len, batch_size, features)
            targets = target_train.permute(1, 0, 2).to(device) # Shape: (sequence_len, batch_size, features)
            
            optimizer.zero_grad()
            outputs = model(
                src=sources, 
                trg=targets, 
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        #======================Test Loss=====================
        model.eval()  # Set the model to evaluation mode
        total_test_loss = 0
        with torch.no_grad():
            for source_test, targets_test in test_loader:
                sources = source_test.permute(1, 0, 2).to(device)
                targets = targets_test.permute(1, 0, 2).to(device)
                
                outputs = model(
                    src=sources, 
                    trg=targets, 
                    teacher_forcing_ratio=0.0
                )
                
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
            if verbose == True:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Validation Loss: {avg_test_loss:.6f}")
        
        if avg_test_loss < best_metric:
            bestWights = model.state_dict()  # Save model state
            best_metric = avg_test_loss
            
        avg_train_loss_history.append(avg_train_loss)
        avg_test_loss_history.append(avg_test_loss)

    return bestWights, avg_train_loss_history, avg_test_loss_history

def prepareTraining(parameters, trainData, testData, verbose=False):
    #==============================================
    #=============== Hyperparameters ==============
    #==============================================
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']

    #==============================================
    #============== Create Dataloader =============
    #==============================================
    train_loader = createDataLoaders(
        batch_size=batch_size, dataset=trainData, shuffle=True
    )
    test_loader = createDataLoaders(
        batch_size=batch_size, dataset=testData, shuffle=False
    )
        
    #==============================================
    #============== Model Setup ===================
    #==============================================
    model, device = createModel(parameters)
    size_model = countModelParameters(model)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #==============================================
    #============== Verbose ===================
    #==============================================
    if verbose:
        print(f"Size of train loader: {len(train_loader)}, Size of test loader: {len(test_loader)}")
        print(f"Used device: {device}")
        print(f"Size of model: {size_model}")
        print(model)

    return model, criterion, optimizer, train_loader, test_loader, device

def createModel(parameters): 
    inputFeatureSize = parameters['input_size']
    outputFeatureSize = parameters['output_size']
    hidden_size = parameters['hidden_size']
    num_layers = parameters['num_layers']
    dropout_rate = parameters['dropout_rate']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(
        inputFeatureSize, outputFeatureSize, hidden_size, num_layers, dropout_rate
    ).to(device)
    return model, device