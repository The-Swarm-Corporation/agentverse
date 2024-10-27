import torch
import torch.nn as nn


class TimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        lstm_layers,
        transformer_layers,
        output_dim,
    ):
        super(TimeSeriesModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, lstm_layers, batch_first=True
        )

        # Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=transformer_layers
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Pass through transformer encoder
        transformer_out = self.transformer(lstm_out)

        # Pass through fully connected layer
        out = self.fc(
            transformer_out[:, -1, :]
        )  # take the last element for sequence prediction
        return out


# Test the model with random input
if __name__ == "__main__":
    model = TimeSeriesModel(
        input_dim=10,
        hidden_dim=128,
        lstm_layers=2,
        transformer_layers=2,
        output_dim=1,
    )
    test_input = torch.randn(
        32, 5, 10
    )  # batch_size=32, sequence_length=5, input_dim=10
    test_output = model(test_input)
    print("Output shape:", test_output.shape)
