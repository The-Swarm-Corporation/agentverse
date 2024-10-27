import torch.nn as nn


class TimeSeriesHybridModel(nn.Module):
    def __init__(
        self,
        input_size,
        lstm_hidden_size,
        lstm_layers,
        transformer_layers,
        transformer_heads,
        transformer_dim_feedforward,
        output_size,
    ):
        super(TimeSeriesHybridModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            lstm_layers,
            batch_first=True,
        )

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim_feedforward,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # Pass data through LSTM
        lstm_out, _ = self.lstm(x)

        # Pass data through the Transformer
        transformer_out = self.transformer(lstm_out)

        # Flatten the output for the linear layer
        out = self.fc(transformer_out[:, -1, :])

        return out
