from torch.backends import cudnn
import torch

'''
The function take a trained model and predict a sequence instead of 1 ahead. 
Args:
- model
- dataset
- sequence length

Returns: 
-Padded vector of predictions. 
'''

