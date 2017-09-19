import torch
import torch.nn as nn


class RNN(nn.Module):
    SEQ_LENGTH = 1000

    NUM_FEATURES = 5
    HIDDEN_UNITS = 128

    def __init__(self):
        super(RNN, self).__init__()

        self.model = nn.Sequential()

        self.rnn = nn.RNN(
            input_size=self.NUM_FEATURES,
            hidden_size=self.HIDDEN_UNITS,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.2)
        self.out = nn.Linear(self.HIDDEN_UNITS, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)

        delta = torch.zeros(x.size(0), self.SEQ_LENGTH - x.size(1), x.size(2))
        x_padded = torch.cat([x, delta], dim=1)

        r_out, h_state = self.rnn(x_padded, h_state)
        output = self.out(r_out)

        return torch.sigmoid(output.narrow(1, 0, x.size(1))), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state
