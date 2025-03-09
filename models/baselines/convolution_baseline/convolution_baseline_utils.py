import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc,roc_curve
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import Metric
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.sampler import DualSampler


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode(sequence):
    """One-hot encode a sequence of amino acids."""
    encoding = np.zeros((len(sequence), len(AMINO_ACIDS)))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1
    return encoding

def pad_sequence(encoded_sequence, max_length=50):
    """Pad the one-hot encoded sequence with zeros to a fixed length."""
    if len(encoded_sequence) >= max_length:
        return encoded_sequence[:max_length]
    else:
        padding = np.zeros((max_length - len(encoded_sequence), len(AMINO_ACIDS)))
        return np.vstack((encoded_sequence, padding))
    
def process_sequences(sequences):
    """Process a list of amino acid sequences by one-hot encoding and padding."""
    processed_sequences = []
    for seq in sequences:
        encoded = one_hot_encode(seq)
        padded = pad_sequence(encoded)
        processed_sequences.append(padded)
    return np.array(processed_sequences)

def check_if_parameters_can_create_valid_model(kernel_size, num_layers, dilation):
    """Check if the parameters can create a valid model."""
    length = 50
    stride = 1
    for _ in range(num_layers):
        length = ((length - dilation * (kernel_size - 1) - 1) // stride) + 1
        length = length // 2
        if length <= 0:
            return False
    return True

class CNNModel(nn.Module):
    def __init__(self, filters=32, kernel_size=3, num_layers=1,padding = 'valid',dilation = 1):
        super(CNNModel, self).__init__()
        layers = []
        length = 50
        in_channels = 20
        stride = 1
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,padding = padding,dilation = dilation))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
            if  padding == 'valid':
                 length = ((length - dilation * (kernel_size - 1) - 1) // stride) + 1  # (length−dilation×(kernel_size−1)−1)//stride+1
                # length = (length - kernel_size + 1)\
            length = length // 2
        self.conv_layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(filters*length, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def calculate_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def calculate_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def pu_metric_calculation(y_true, y_pred,modified):
    """
    Computes the custom metric: r^2 / Pr(ŷ = 1) for the optimal threshold.
    
    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).
    
    Returns:
        tuple: (max_metric, best_threshold)
            - max_metric: Maximum metric value.
            - best_threshold: Threshold corresponding to the maximum metric.
    """
    thresholds=np.linspace(0, 1, 101)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    max_metric = 0.0
    best_threshold = 0.0
    
    for threshold in thresholds:
        # Binarize predictions based on the threshold
        binary_pred = (y_pred >= threshold).astype(int)
        
        # True positives, false negatives, and false positives
        tp = np.sum((y_true == 1) & (binary_pred == 1))
        fn = np.sum((y_true == 1) & (binary_pred == 0))
        fp = np.sum((y_true == 0) & (binary_pred == 1))
        
        # Recall (r)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Pr(ŷ = 1)
        predicted_positive_rate = (tp + fp) / len(y_pred)
        
        # Metric computation
        metric = (recall ** 2) / predicted_positive_rate if predicted_positive_rate > 0 else 0.0
        if modified:
            metric = (recall ** 2) / fp if fp > 0 else 0.0
        
        # Update maximum metric and best threshold
        if metric > max_metric:
            max_metric = metric
            best_threshold = threshold
    
    return max_metric, best_threshold

def score_function_pu(engine,modified):
    y_true, y_pred = engine.state.output
    val_pu_metric, _ = pu_metric_calculation(y_true.cpu().numpy(), y_pred.cpu().numpy(),modified)
    print(f"Epoch {engine.state.epoch} - val-PU: {val_pu_metric}")

    return val_pu_metric


def score_function(engine):
    y_true, y_pred = engine.state.output
    val_pr_auc = calculate_pr_auc(y_true.cpu().numpy(), y_pred.cpu().numpy())
    engine.state.score = val_pr_auc
    return val_pr_auc

def roc_score_function(engine):
    print('roc_score_function')
    y_true, y_pred = engine.state.output
    val_roc_auc = calculate_roc_auc(y_true.cpu().numpy(), y_pred.cpu().numpy())
    engine.state.score = val_roc_auc
    return val_roc_auc

def modify_final_layer(model,depth):
    """
    Replace the final non-CNN layers (classifier head) with new, randomly initialized layers.
    In our CNNModel, these are the fully-connected layers fc1 and fc2.
    """
    # Get the number of input features for fc1 (kept from the convolutional layers)
    in_features = model.fc1.in_features

    # Replace fc1 and fc2 with new layers (here we use 128 hidden units and 1 output)
    model.fc1 = nn.Linear(in_features, depth)
    model.fc2 = nn.Linear(depth, 1)

    # Initialize the new layers
    nn.init.xavier_uniform_(model.fc1.weight)
    nn.init.zeros_(model.fc1.bias)
    nn.init.xavier_uniform_(model.fc2.weight)
    nn.init.zeros_(model.fc2.bias)

    print("Replaced final classifier layers with new initialized layers.")
    return model

def train_architecture_fold(filter,kernel_size,num_layers,batch_size,padding,dialation,epochs,fold_train_dict,X_train,y_train,ROC_metric,pesg_dict,pretrained_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if X_train is None:
        X_train = torch.tensor(process_sequences(fold_train_dict['sequences_train']), dtype=torch.float32).transpose(1, 2).to(device)
        y_train = torch.tensor(fold_train_dict['labels_train'], dtype=torch.float32).to(device)
    else:
        X_train = torch.tensor(process_sequences(X_train), dtype=torch.float32).transpose(1, 2).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_val = torch.tensor(process_sequences(fold_train_dict['sequences_validation']), dtype=torch.float32).transpose(1, 2).to(device)
    y_val = torch.tensor(fold_train_dict['labels_validation'], dtype=torch.float32).to(device)
    

    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle = True)
    train_dataset = TensorDataset(X_train, y_train)
    if pesg_dict is not None:
        train_sampler = DualSampler(train_dataset,batch_size=batch_size ,sampling_rate=0.5,labels=y_train.cpu().numpy())
        train_loader = DataLoader(train_dataset,batch_size=batch_size, sampler=train_sampler)
    else: 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
        class_weights = torch.FloatTensor(class_weights).to(device)

    print(f'Filter: {filter}, Kernel size: {kernel_size}, Num layers: {num_layers}, Batch size: {batch_size},Padding:{padding},Dialation:{dialation},ROC_metric:{ROC_metric},pesg_dict:{pesg_dict}')
    model = CNNModel(filters=filter, kernel_size=kernel_size, num_layers=num_layers,padding=padding,dilation=dialation).to(device)
    summary(model, (20, 50))                       
    if pretrained_model is not None:
        checkpoint = torch.load(pretrained_model, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {pretrained_model}")
        model = modify_final_layer(model,128)
        model.to(device)
    
    if pesg_dict is not None:
        y_training = torch.tensor(fold_train_dict['labels_train'], dtype=torch.float32).to(device)
        positive_ratio = float(torch.sum(y_training) / len(y_training))
        lr = pesg_dict['lr']
        weight_decay = pesg_dict['weight_decay']
        margin = pesg_dict['margin']
        loss_fn = AUCMLoss(margin=margin,imratio = positive_ratio)
        optimizer = PESG(model.parameters(), loss_fn=loss_fn, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        X_batch, y_batch = batch
        outputs = model(X_batch).squeeze(dim=1)
        if pesg_dict is not None:
            loss = loss_fn(outputs, y_batch)
        else:
            batch_weights = class_weights[y_batch.long()]
            criterion = nn.BCELoss(weight=batch_weights)
            loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            X_batch, y_batch = batch
            y_pred = model(X_batch).squeeze(dim=1)
            return y_batch, y_pred

    def accumulate_outputs(engine, batch):
        y_true, y_pred = batch
        if not hasattr(engine.state, 'all_y_true'):
            engine.state.all_y_true = []
            engine.state.all_y_pred = []
        engine.state.all_y_true.append(y_true)
        engine.state.all_y_pred.append(y_pred)

    def reset_accumulated_outputs(engine):
        print('reset_accumulated_outputs')
        engine.state.all_y_true = []
        engine.state.all_y_pred = []
        engine.state.score = 0
        

    def get_accumulated_outputs(engine):
        y_true = torch.cat(engine.state.all_y_true, dim=0)
        y_pred = torch.cat(engine.state.all_y_pred, dim=0)
        return y_true, y_pred

    trainer = Engine(train_step)
    evaluator = Engine(validation_step)
    evaluator.add_event_handler(Events.STARTED, reset_accumulated_outputs)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: accumulate_outputs(engine, engine.state.output))
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: setattr(engine.state, 'output', get_accumulated_outputs(engine)))

    @trainer.on(Events.EPOCH_COMPLETED)
    def calculate_pr_auc_engine(engine):
        model.eval()
        print(f' loss: {engine.state.output:.3f}')
        evaluator.run(val_loader)
        score = evaluator.state.score
        print(f"Epoch {engine.state.epoch} - metric_score: {score:.3f}")

    if ROC_metric:
        handler = EarlyStopping(patience=5,score_function = roc_score_function , trainer = trainer)
    else:
        handler = EarlyStopping(patience=5, score_function=score_function, trainer = trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)
                        
    # Calculate metric
    model.eval()
    evaluator.run([(X_val, y_val)])
    y_true, y_pred = evaluator.state.output
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if ROC_metric:
        val_metric = calculate_roc_auc(y_true, y_pred)
    else:
        val_metric = calculate_pr_auc(y_true, y_pred)
    return val_metric,model
