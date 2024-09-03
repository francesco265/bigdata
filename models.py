import torch.nn as nn

class PollutionModel(nn.Module):
    def __init__(self, input_size, output_size, rnn_type = 'lstm', **kwargs):
        super(PollutionModel, self).__init__()

        assert rnn_type in ['lstm', 'gru'], 'rnn_type must be either "lstm" or "gru"'
        layer_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU

        self.input_size = input_size
        self.hidden_size = (input_size + output_size) // 2
        self.rnn_layer = layer_type(self.hidden_size, self.hidden_size, batch_first=True, **kwargs)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        if 'bidirectional' in kwargs and kwargs['bidirectional']:
            self.fc2 = nn.Linear(self.hidden_size * 2, output_size)
        else:
            self.fc2 = nn.Linear(self.hidden_size, output_size)

    def change_input_size(self, input_size):
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, self.hidden_size)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out, _ = self.rnn_layer(out)
        out = self.fc2(self.relu(out[:, -1, :]))
        return out
