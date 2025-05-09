import torch
import torch.optim as optim
import copy

#from .TrafficPredictor import TrafficPredictorContextAssisted, CustomLossFunction
from .TrafficPredictorEnhanced import TrafficPredictorContextAssisted, CustomLossFunction
from ..HelperFunctions import createDataLoaders, countModelParameters

def getDefaultModelParams(len_source, len_target, dataset):
    (source_train, _, _, _, _, _, transmission_train) = dataset
    input_size = source_train.shape[2]
    output_size = transmission_train.shape[1]
    parameters = {
        "input_size":input_size,
        "output_size":output_size,
        "batch_size": 4096*2,
        "hidden_size": 64,
        "num_layers": 5,
        "dropout_rate": 0.8,
        "num_epochs": 50,
        "learning_rate": 0.01,
        "dt": 0.01,
        "degree" : 3,
        "len_source": len_source,
        "len_target": len_target,
        "num_classes": len_target+1,
        "train_ratio": 0.6,
        "lambda_traffic_class": 100, 
        "lambda_transmission": 500,
        "lambda_context":100.0
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
            sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions = (
                data.to(device) for data in batch
            )
            sources = sources.permute(1, 0, 2)
            targets = targets.permute(1, 0, 2)

            traffics_class = traffics_class.view(-1).to(torch.long)
            last_trans_sources = last_trans_sources.permute(1, 0, 2)
            
            optimizer.zero_grad()
            out_traffic, out_traffic_class, out_trans, out_target = model(sources, last_trans_sources)
            loss, _ = criterion(
                out_traffic, traffics,
                out_traffic_class, traffics_class,
                out_trans, transmissions,
                out_target, targets
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
                sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions = (
                    data.to(device) for data in batch
                )
                sources = sources.permute(1, 0, 2)
                targets = targets.permute(1, 0, 2)
                traffics_class = traffics_class.view(-1).to(torch.long)
                last_trans_sources = last_trans_sources.permute(1, 0, 2)
                
                out_traffic, out_traffic_class, out_trans, out_target = model(sources, last_trans_sources)
                loss, loss_traffic = criterion(
                    out_traffic, traffics,
                    out_traffic_class, traffics_class,
                    out_trans, transmissions,
                    out_target, targets
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

'''
def trainModelHelper(parameters, model, criterion, optimizer, device, train_loader, test_loader, verbose=False):
    import math, random, os, torch
    from tqdm import tqdm

    torch.cuda.set_per_process_memory_fraction(1.0)  # 100% of GPU memory
    torch.cuda.empty_cache()  # Clear unused memory
    # ------------------------------------------------------------------
    # 1.  Reproducibility: fix every RNG we can reach
    # ------------------------------------------------------------------
    def set_seed(seed: int = 42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    set_seed()

    # ------------------------------------------------------------------
    # 2.  Optimizer, weight-decay & LR scheduler
    #     – AdamW + OneCycleLR gives a very smooth loss curve
    # ------------------------------------------------------------------
    optimizer  = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr        = parameters['learning_rate'],
        epochs        = parameters['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start     = 0.1,            # 10 % warm-up
        anneal_strategy='cos',          # cosine cool-down
        div_factor     = 25.0,          # initial LR = max_lr / 25
        final_div_factor=1e3            # final LR = max_lr / 1000
    )

    # ------------------------------------------------------------------
    # 3.  Mixed precision & gradient clipping
    # ------------------------------------------------------------------
    scaler            = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    grad_clip_val     = 10.0                # L2-norm threshold
    ema_decay         = 0.999              # for optional EMA weights
    ema_shadow        = [p.detach().clone() for p in model.parameters()]

    best_metric       = math.inf
    train_hist, val_hist = [], []

    for epoch in range(parameters['num_epochs']):
        # ---------------------- TRAIN -------------------------------
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{parameters['num_epochs']}", leave=False)
        
        for batch in pbar:
            sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions = (
                data.to(device, non_blocking=True) for data in batch
            )
            # (B, T, C)  -> (T, B, C) expected by Transformer-style nets
            sources, targets, last_trans_sources = (
                x.permute(1,0,2) for x in (sources, targets, last_trans_sources)
            )
            traffics_class = traffics_class.view(-1).long()

            optimizer.zero_grad(set_to_none=True)
            out_traffic, out_traffic_class, out_trans, out_target = model(sources, last_trans_sources)
            loss, _ = criterion(
                out_traffic,  traffics,
                out_traffic_class, traffics_class,
                out_trans,   transmissions,
                out_target,  targets
            )

            scaler.scale(loss).backward()
            # gradient-norm clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Optional exponential moving-average weights (improves val-curve wobble)
            with torch.no_grad():
                for p, p_ema in zip(model.parameters(), ema_shadow):
                    p_ema.lerp_(p.data, 1.0 - ema_decay)

            epoch_loss += loss.item()
            pbar.set_postfix({"train_loss": epoch_loss / (pbar.n or 1)})
        
        train_hist.append(epoch_loss / len(train_loader))

        # ---------------------- VALIDATION --------------------------
        model.eval()
        val_loss = 0.0; val_loss_traffic = 0.0
        with torch.no_grad():
            for batch in test_loader:
                sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions = (
                    data.to(device, non_blocking=True) for data in batch
                )
                sources, targets, last_trans_sources = (
                    x.permute(1,0,2) for x in (sources, targets, last_trans_sources)
                )
                traffics_class = traffics_class.view(-1).long()

                out_t, out_tc, out_trans, out_target = model(sources, last_trans_sources)
                loss, loss_traffic = criterion(
                    out_t, traffics,
                    out_tc, traffics_class,
                    out_trans, transmissions,
                    out_target, targets
                )
                val_loss          += loss.item()
                val_loss_traffic  += loss_traffic.item()

        avg_val_loss         = val_loss / len(test_loader)
        avg_val_loss_traffic = val_loss_traffic / len(test_loader)
        val_hist.append(avg_val_loss)

        # ---------------------- LOGGING -----------------------------
        print(f"Epoch {epoch+1:2d}/{parameters['num_epochs']}  "
            f"Train: {train_hist[-1]:.4f}  "
            f"Val: {avg_val_loss:.4f}  "
            f"ValTraffic: {avg_val_loss_traffic:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.3e}")

        # ---------------------- CHECKPOINT --------------------------
        if avg_val_loss < best_metric:
            best_metric = avg_val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    return best_weights, train_hist, val_hist
'''
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
    criterion = CustomLossFunction(
        lambda_trans=parameters['lambda_transmission'], 
        lambda_class=parameters['lambda_traffic_class'],
        lambda_context=parameters['lambda_context'])
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