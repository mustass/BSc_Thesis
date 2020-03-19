import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, input_dim, batch_size,
                 hidden_dim, dropout, num_layers,  # First LSTM
                 output_dim=1):
        super(Model, self).__init__()
        # General:
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.output_dim = output_dim

        # First LSTM params:

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # First LSTM layer:
        self.LSTM = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout)

        # Output layer:

        self.linear = nn.Linear(hidden_dim, self.output_dim, bias=False)


    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as

        hidden = Variable(next(self.parameters()).data.new_zeros(self.num_layers, batch_size ,self.hidden_dim))
        cell = Variable(next(self.parameters()).data.new_zeros(self.num_layers, batch_size,  self.hidden_dim))
        return (hidden, cell)



    def forward(self, input):
        lstm_out_1, self.hidden = self.LSTM(input.view(len(input), self.batch_size, -1))
        preds = self.linear(lstm_out_1[-1])
        #preds = lstm_out_1[-1]
        return preds
