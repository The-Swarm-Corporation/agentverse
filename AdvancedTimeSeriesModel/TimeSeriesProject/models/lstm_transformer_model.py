# Import necessary libraries
import torch.nn as nn


class LSTMTransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        lstm_hidden_size,
        transformer_hidden_size,
        nhead,
        num_layers,
        output_size,
    ):
        super(LSTMTransformerModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size, lstm_hidden_size, batch_first=True
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size, nhead=nhead
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        transformer_out = self.transformer(lstm_out)

        output = self.fc(transformer_out.mean(dim=1))

        return output
