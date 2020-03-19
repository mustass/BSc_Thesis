"""
File for gathering the results for hyperopt search.
"""
from ray.tune import Analysis
import torch
import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.pred_sequence import *

folders = []
folders.append("Run_w_11_timesteps_3_hiddenDim_1_layers_0.00021723989839730966_LR")
configs = []
analysis = Analysis("~/ray_results/Jan21")
configs.append(analysis.get_best_config(metric="error", mode="max"))
for i in range(2,17):
    analysis = Analysis("~/ray_results/"+str(i)+"forward")
    config = analysis.get_best_config(metric="error", mode="max")
    configs.append(config)
    print(config)
    folder_name = 'Predicting' + str(config["num_forward"]) + '_w_' + str(config["timesteps"]) + '_timesteps_' + str(
        config["hidden_dim"]) + '_hiddenDim_' + str(
        config["num_layers"]) + '_layers_' + str(config["lr"]) + "_LR"
    folders.append(folder_name)
    #print(folder_name)
print(len(folders))
print(len(configs))
model_keys =[]
config_multiple_models = {}
for i in range(0,16):
    key = str(i+1)+'forward'
    model_keys.append(key)
    path_to_checkpoint = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/'+folders[i]+'/checkpoint.pth.tar'
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(path_to_checkpoint)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(path_to_checkpoint,
                                map_location=lambda storage,
                                                    loc: storage)
    #print(i)
    config = {'hidden_dim': configs[i]['hidden_dim'], 'num_layers': configs[i]['num_layers'], 'timesteps': configs[i]['timesteps'],'state_dict': checkpoint['state_dict'], 'num_forward': i+1}
    config_multiple_models[key] = config

print(config_multiple_models['1forward'])

