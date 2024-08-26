import torch.nn as nn

class ModelLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, bidirectional=False):
        super(ModelLSTM, self).__init__()
        self.hidden_size = (input_size + output_size) // 2
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out, _ = self.lstm(out)
        out = self.fc2(self.relu(out[:, -1, :]))
        return out

class ModelGRU(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1):
        super(ModelGRU, self).__init__()
        self.hidden_size = (input_size + output_size) // 2
        self.num_layers = num_layers
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out, _ = self.gru(out)
        out = self.fc2(self.relu(out[:, -1, :]))
        return out
