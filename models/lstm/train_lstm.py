import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, auc
from models.models_utils import prepare_training_data
import path

# Parameters
batch_size = 32
num_workers = 4
model_name = "facebook/esm2_t6_8M_UR50D"
pretrained_model_path = os.path.join(path.models_path, 'pretrained_lstm_model.pth')

# Prepare training data
dataloader, input_dim = prepare_training_data(batch_size, num_workers, model_name)

# Define the LSTM Model
class SimpleLanguageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleLanguageModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Define model parameters
hidden_dim = 128
output_dim = input_dim  # Assuming we want to reconstruct the input

# Initialize the model
model = SimpleLanguageModel(input_dim, hidden_dim, output_dim)

# Load pre-trained model weights if available
if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path))

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Calculate PR AUC
        precision, recall, _ = precision_recall_curve(batch.flatten().cpu().numpy(), outputs.flatten().detach().cpu().numpy())
        pr_auc = auc(recall, precision)

        # Backward pass and optimization
        loss = -pr_auc  # We minimize the negative PR AUC
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], PR AUC: {pr_auc:.4f}')

# Save the fine-tuned model
torch.save(model.state_dict(), pretrained_model_path)