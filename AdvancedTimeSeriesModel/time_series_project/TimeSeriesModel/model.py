import torch
import torch.nn as nn


class TimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_size,
        lstm_hidden_size,
        transformer_hidden_size,
        output_size,
        num_layers=1,
    ):
        super(TimeSeriesModel, self).__init__()

        # LSTM part
        self.lstm = nn.LSTM(
            input_size, lstm_hidden_size, num_layers, batch_first=True
        )

        # Transformer part
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_size,
            nhead=4,
            dim_feedforward=512,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=4
        )

        # Linear to output
        self.fc = nn.Linear(
            lstm_hidden_size + transformer_hidden_size, output_size
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the last output of LSTM

        # Transformer
        transformer_input = x.permute(
            1, 0, 2
        )  # Change to (seq_len, batch, input_size)
        transformer_out = self.transformer(transformer_input)
        transformer_out = transformer_out[
            -1, :, :
        ]  # Get the last output of Transformer

        # Concatenate LSTM and Transformer outputs
        combined = torch.cat((lstm_out, transformer_out), dim=1)

        # Fully connected layer
        output = self.fc(combined)

        return output
