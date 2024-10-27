import torch.nn as nn


class LSTMTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        lstm_hidden_dim,
        lstm_layers,
        transformer_hidden_dim,
        nhead,
        num_transformer_layers,
        output_dim,
    ):
        super(LSTMTransformer, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden_dim, lstm_layers, batch_first=True
        )

        # Transformer Encoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_dim,
            nhead=nhead,
            dim_feedforward=transformer_hidden_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_transformer_layers
        )

        # Output layer
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        # Pass data through LSTM
        lstm_out, _ = self.lstm(x)

        # Pass the LSTM output through the Transformer encoder
        transformer_out = self.transformer_encoder(lstm_out)

        # Feed the output of the transformer encoder to a fully connected layer
        output = self.fc(transformer_out)

        return output
