import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from AdvancedTimeSeriesModel.model import TimeSeriesModel
from AdvancedTimeSeriesModel.utils import preprocess_data

# Hyperparameters
input_dim = 1  # Number of features
hidden_dim = 64  # Hidden dimension for LSTM
lstm_layers = 2  # Number of LSTM layers
transformer_layers = 2
learning_rate = 0.001
n_steps = 5  # Number of time steps for sequences
batch_size = 16
num_epochs = 50

# Sample synthetic data (replace with your data)
data = np.array([[i] for i in range(100)])  # Example sequential data

# Preprocess data
X, y, scaler = preprocess_data(data, n_steps)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = TimeSeriesModel(
    input_dim, hidden_dim, lstm_layers, transformer_layers
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
