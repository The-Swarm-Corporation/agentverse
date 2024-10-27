import torch.nn as nn


class TransformerLSTMTimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_size,
        lstm_hidden_size,
        lstm_num_layers,
        transformer_heads,
        transformer_layers,
    ):
        super(TransformerLSTMTimeSeriesModel, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
        )

        # Transformer Layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size, nhead=transformer_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=transformer_layers
        )

        # Output Layer
        self.output_layer = nn.Linear(
            lstm_hidden_size, 1
        )  # Assuming single regression output

    def forward(self, x):
        # LSTM Forward Pass
        lstm_out, _ = self.lstm(x)

        # Transformer Forward Pass
        transformer_out = self.transformer(lstm_out)

        # Output Layer
        out = self.output_layer(
            transformer_out[:, -1, :]
        )  # Use the last output for prediction

        return out
