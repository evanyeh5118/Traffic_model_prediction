import torch
import torch.optim as optim
import copy

from .TrafficPredictor import TrafficPredictorContextAssisted, CustomLossFunction
from ..HelperFunctions import createDataLoaders, countModelParameters

def getDefaultModelParams(len_source, len_target, dataset):
    (source_train, _, _, _, _, _, transmission_train) = dataset
    input_size = source_train.shape[2]
    output_size = transmission_train.shape[1]
    parameters = {
        "input_size":input_size,
        "output_size":output_size,
        "batch_size": 4096,
        "hidden_size": 128,
        "num_layers": 5,
        "dropout_rate": 0.9,
        "num_epochs": 50,
        "learning_rate": 0.005,
        "dt": 0.01,
        "degree" : 3,
        "len_source": len_source,
        "len_target": len_target,
        "num_classes": len_target+1,
        "train_ratio": 0.6,
        "lambda_traffic_class": 200, 
        "lambda_transmission": 100,
    }
    return parameters

def trainModelByDefaultSetting(len_source, len_target, trainData, testData, verbose=False):
    parameters = getDefaultModelParams(len_source, len_target, trainData)
  
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
    
    #==============================================
    #============== Training ======================
    #==============================================
    best_metric = float('inf')  # Set to a large value
    avg_train_loss_history = []
    avg_test_loss_history = []
    best_model = None

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            sources, _, last_trans_sources, _, traffics, traffics_class, transmissions = (
                data.to(device) for data in batch
            )
            sources = sources.permute(1, 0, 2)
            traffics_class = traffics_class.view(-1).to(torch.long)
            last_trans_sources = last_trans_sources.permute(1, 0, 2)
            
            optimizer.zero_grad()
            out_traffic, out_traffic_class, out_trans, _ = model(sources, last_trans_sources)
            loss, _ = criterion(
                out_traffic, traffics,
                out_traffic_class, traffics_class,
                out_trans, transmissions,
            )
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        #======================Test Loss=====================
        model.eval()
        total_test_loss = 0
        total_test_loss_traffic = 0
        with torch.no_grad():
            for batch in test_loader:
                sources, _, last_trans_sources, _, traffics, traffics_class, transmissions = (
                    data.to(device) for data in batch
                )
                sources = sources.permute(1, 0, 2)
                traffics_class = traffics_class.view(-1).to(torch.long)
                last_trans_sources = last_trans_sources.permute(1, 0, 2)
                
                out_traffic, out_traffic_class, out_trans, _ = model(sources, last_trans_sources)
                loss, loss_traffic = criterion(
                    out_traffic, traffics,
                    out_traffic_class, traffics_class,
                    out_trans, transmissions,
                )
                total_test_loss += loss.item()
                total_test_loss_traffic += loss_traffic.item()

            avg_test_loss = total_test_loss / len(test_loader)
            avg_test_loss_traffic = total_test_loss_traffic / len(test_loader)

            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Validation Loss: {avg_test_loss:.4f}, "
                      f"Validation Loss (Traffic): {avg_test_loss_traffic:.4f}")
                
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

    lambda_transmission = parameters['lambda_transmission']
    lambda_traffic_class = parameters['lambda_traffic_class']
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
    criterion = CustomLossFunction(lambda_trans=lambda_transmission, lambda_class=lambda_traffic_class)
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
    len_source = parameters['len_source']
    len_target = parameters['len_target']
    num_classes = parameters['num_classes']    
    input_size = parameters['input_size']
    output_size = parameters['output_size']
    hidden_size = parameters['hidden_size']
    num_layers = parameters['num_layers']
    dropout_rate = parameters['dropout_rate']
    dt = parameters['dt']
    degree = parameters['degree']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficPredictorContextAssisted(
        input_size, hidden_size, output_size, num_classes, len_source, len_target, 
        dt, degree, device, num_layers=num_layers, dropout_rate=dropout_rate
    )

    return model, device