from torch.backends import cudnn
import torch
from core.dataloader import *
from numpy import newaxis
import numpy as np

"""
These functions will do a naive sequence prediction
Inputs:
- model 
- data
- length of sequence
Output: 
- A vector of sequences 

First function will treat the 'Volume' column by 
averaging of the factual window.   

Second function will treat the 'Volume' column by 
repeating the last value for the whole sequence.  
"""


def predict_seq_avg(model, dataset, timesteps, seq_len):
    print('[Model] Predicting Sequences Multiple...')
    ######## CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    model = model
    if torch.cuda.is_available():
        model.cuda()
    #######

    test_data = dataset
    prediction_seqs = []
    for i in range(int(len(test_data) / seq_len)):
        curr_frame = test_data[i]
        volume_avg = np.mean(curr_frame[:, 1])
        curr_frame = torch.from_numpy(curr_frame).type(torch.Tensor).detach().view(timesteps, 1, -1).to(device)
        predicted = []

        for j in range(seq_len):
            prediction = model(curr_frame)[0, 0].detach()
            curr_frame = curr_frame[1:]
            new_row = torch.Tensor([prediction, volume_avg]).view(1, 1, 2).to(device)
            curr_frame = torch.cat((curr_frame, new_row), dim=0)
            predicted.append(prediction.cpu().numpy())

        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_seq_last(model, dataset, timesteps, seq_len):
    print('[Model] Predicting Sequences Multiple...')
    ######## CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    model = model
    if torch.cuda.is_available():
        model.cuda()
    #######

    test_data = dataset
    prediction_seqs = []
    for i in range(int(len(test_data) / seq_len)):
        curr_frame = test_data[i]
        volume = curr_frame[-1, 1]
        curr_frame = torch.from_numpy(curr_frame).type(torch.Tensor).detach().view(timesteps, 1, -1).to(device)
        predicted = []

        for j in range(seq_len):
            prediction = model(curr_frame)[0, 0].detach()
            curr_frame = curr_frame[1:]
            new_row = torch.Tensor([prediction, volume]).view(1, 1, 2).to(device)
            curr_frame = torch.cat((curr_frame, new_row), dim=0)
            predicted.append(prediction.cpu().numpy())

        prediction_seqs.append(predicted)
    return prediction_seqs