
import torch
import torch.nn as nn
import torch.optim as optim
from AdvancedTimeSeriesModel.model import TimeSeriesModel
import AdvancedTimeSeriesModel.utils as utils

# Hyperparameters
input_size = 1
lstm_hidden_size = 64
transformer_hidden_size = 64
output_size = 1
num_layers = 1
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Generate synthetic data
data X, Y = utils.generate_synthetic_data()

# Create data loaders
train_loader, val_loader = utils.create_dataloaders(X, Y)

# Initialize model, loss, and optimizer
model = TimeSeriesModel(input_size, lstm_hidden_size, transformer_hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        X_batch = torch.tensor(X_batch, dtype=torch.float32)
        Y_batch = torch.tensor(Y_batch, dtype=torch.float32)
        
        # Forward pass
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


# Training the model
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

