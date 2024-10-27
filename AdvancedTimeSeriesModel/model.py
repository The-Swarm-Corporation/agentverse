import torch.nn as nn
from transformers import BertModel


class TimeSeriesTransformerLSTM(nn.Module):
    def __init__(
        self,
        transformer_model_name="bert-base-uncased",
        lstm_hidden_size=128,
        num_lstm_layers=2,
        output_size=1,
    ):
        super(TimeSeriesTransformerLSTM, self).__init__()

        # Load a pre-trained transformer model
        self.transformer = BertModel.from_pretrained(
            transformer_model_name
        )

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Define a fully connected layer for the output
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # Pass the input data through the transformer
        transformer_outputs = self.transformer(
            input_ids=x, return_dict=True
        )

        # Get the last hidden state
        hidden_state = transformer_outputs.last_hidden_state

        # Pass the hidden state through the LSTM
        lstm_out, _ = self.lstm(hidden_state)

        # Get the output from the last time step
        out = self.fc(lstm_out[:, -1, :])

        return out
