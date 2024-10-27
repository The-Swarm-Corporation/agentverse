import torch.nn as nn


class TimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        lstm_layers,
        transformer_layers,
        dropout=0.1,
    ):
        super(TimeSeriesModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # Final linear layer for output
        self.linear = nn.Linear(
            hidden_dim, 1
        )  # Adjust the output dimension as necessary

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Transformer
        transformer_out = self.transformer(
            lstm_out.permute(1, 0, 2)
        )  # Transformer expects input of shape (seq_len, batch, dim)

        # Final linear layer
        output = self.linear(
            transformer_out[-1, :, :]
        )  # Use the last time step for prediction

        return output
