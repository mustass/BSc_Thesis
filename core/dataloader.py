import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils import data


class DataLoader():
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, path, split, cols, label_col, normalise, start_from=None):
        filename = path
        dataframe = pd.read_csv(filename)

        if start_from is not None:
            dataframe.Date = pd.to_datetime(dataframe.Date)
            start = pd.to_datetime(start_from)
            dataframe = dataframe.loc[dataframe.Date > start]

        i_split = int(len(dataframe) * split)
        dataframe = dataframe.get(cols)
        self.data_train = dataframe.values[:i_split]
        self.data_test = dataframe.values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.label_col_indx = (dataframe.columns.get_loc(label_col))  # Get index of label column
        if normalise:
            self.scaler = MinMaxScaler()
            self.data_train = self.scaler.fit_transform(self.data_train)
            self.data_test = self.scaler.transform(self.data_test)


    def get_train_data(self, seq_len):
        '''
        Seq_len: total length, ie. the last gets to be the label
        '''
        seq_len += 1
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, 'train')
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_test_data(self, seq_len):
        '''
        Seq_len: total length, ie. the last gets to be the label
        '''
        seq_len += 1
        data_x = []
        data_y = []
        for i in range(self.len_test - seq_len):
            x, y = self._next_window(i, seq_len, 'test')
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def _next_window(self, i, seq_len, split):
        """Generates the next data window from the given index location i"""
        ''
        if split == 'train':
            window = self.data_train[i:i + seq_len]
            x = window[:-1]
            y = window[-1, [self.label_col_indx]]

        if split == 'test':
            window = self.data_test[i:i + seq_len]
            x = window[:-1]
            y = window[-1, [self.label_col_indx]]

        return x, y



class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dataset):
        'Initialization'
        self.data, self.labels = dataset

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        dt = torch.from_numpy(self.data).type(torch.Tensor)
        x = dt[index]
        y = self.labels[index]
        return x, y
