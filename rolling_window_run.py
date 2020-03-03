import torch
import numpy as np
import pickle
from torch.backends import cudnn
import time
from ray import tune
from core.model import *
from ray.tune import track
from core.dataloader import *
from core.create_folder import *
from utils.get_hyperparams_df import configs_df
from core.rolling_window_train import rolling_window

for dt in ["DJI","GSPC", "IXIC", "N225"]:
    config = configs_df.filter(like=str(dt), axis=0)
    timesteps = int(config['Timesteps'])
    num_layers = int(config['Num. layers'])
    hidden_dim = int(config['Hidden dim'])
    lr = float(config['LR'])
    dropout = float(config['Dropout'])
    model_config = {'hidden_dim': hidden_dim,
                   'batch_size': 1,
                   'num_layers': num_layers,
                   'lr':lr,
                   'dropout': dropout}
    save_folder = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/returns/rolling_1/'
    data_folder = '/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'
    output = rolling_window(model_config =model_config, datasetName=dt, start_from = "2012-01-01",
                          restart=False,restart_window= 0, window_length=1,timesteps= timesteps,
                          max_epochs=50, data_folder= data_folder, save_folder=save_folder
                          )
    file_output = open(
        save_folder + dt + '/output_file.pickle', 'wb')
    pickle.dump(output, file_output)