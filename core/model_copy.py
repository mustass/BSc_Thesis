import torch
import torch.nn as nn
from torch.backends import cudnn


class Model(nn.Module):

    def __init__(self, input_dim, batch_size,
                 hidden_dim, dropout, num_layers,  # First LSTM
                 hidden_dim_2, dropout_2, num_layers_2,  # Second LSTM
                 btwn_lyr_dropout,  # Designated dropout layer
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
        self.hidden_1 = None

        # Second LSTM params:
        self.hidden_dim_2 = hidden_dim_2
        self.dropout_2 = dropout_2
        self.num_layers_2 = num_layers_2
        self.hidden_2 = None

        # Designated dropout params:
        self.btwn_lyr_dropout = btwn_lyr_dropout

        # First LSTM layer:
        self.LSTM1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout)

        # Dropout Layer:

        self.dropout = nn.Dropout(self.btwn_lyr_dropout)

        # Second LSTM layer:
        self.LSTM2 = nn.LSTM(self.hidden_dim, self.hidden_dim_2, self.num_layers_2, dropout=self.dropout_2)

        # Output layer:

        self.linear = nn.Linear(hidden_dim_2, self.output_dim)

        # Hidden states:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True
        #self.hidden_1 = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device = device),
        #                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim,device = device))
        #self.hidden_2 = (torch.zeros(self.num_layers_2, self.batch_size, self.hidden_dim_2, device = device),
        #                 torch.zeros(self.num_layers_2, self.batch_size, self.hidden_dim_2, device = device))

    def init_hidden_1(self):
        # This is what we'll initialise our hidden state as
        # First LSTM:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True

        h_1 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device = device)
        c_1 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device = device)

        hidden_1 = (h_1, c_1)

        return hidden_1

    def init_hidden_2(self):
        # This is what we'll initialise our hidden state as
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True

        h_2 = torch.zeros(self.num_layers_2, self.batch_size, self.hidden_dim_2, device = device)
        c_2 = torch.zeros(self.num_layers_2, self.batch_size, self.hidden_dim_2, device = device)

        hidden_2 = (h_2, c_2)
        return hidden_2


    def forward(self, input):
        lstm_out_1, self.hidden_1 = self.LSTM1(input.view(len(input), self.batch_size, -1))

        lstm_out_1 = self.dropout(lstm_out_1)

        lstm_out_2, self.hidden_2 = self.LSTM2(lstm_out_1.view(len(lstm_out_1), self.batch_size, -1))

        preds = self.linear(lstm_out_2[-1])

        return preds
