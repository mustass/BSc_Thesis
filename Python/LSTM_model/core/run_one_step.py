from utils.get_hyperparams_df import configs_df
import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
import pickle
from core.create_folder import *


def run_one_step(dt, data_type):
    folder = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/' + data_type + '/one_step_epochs/' + dt
    config = configs_df.filter(like=str(dt), axis=0)

    if data_type == 'returns' and dt == 'DJI':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/' + dt + '.csv',
                             start_from= "1985-01-01",
                             end="2015-01-01",
                             split=0.8, cols=['log_ret'], label_col='log_ret', MinMax=False)
    if data_type == 'returns' and dt == 'GSPC':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/' + dt + '.csv',
                             start_from= "1985-01-01",
                             end="2015-01-01",
                             split=0.8, cols=['log_ret'], label_col='log_ret', MinMax=False)
    if data_type == 'returns' and dt == 'N225':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/' + dt + '.csv',
                             start_from= "1985-01-01",
                             end="2015-01-01",
                             split=0.8, cols=['log_ret'], label_col='log_ret', MinMax=False)
    if data_type == 'returns' and dt == 'IXIC':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/' + dt + '.csv',
                             start_from= "1985-01-01",
                             end="2015-01-01",
                             split=0.8, cols=['log_ret'], label_col='log_ret', MinMax=False)

    if data_type == 'volatility' and dt == "DJI":
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Volatility/' + dt + '.csv',
                             end="2018-01-01",
                             split=0.8, cols=['rv5'], label_col='rv5', returns=False, MinMax=False)
        config = {'Timesteps': 31, 'Num. layers': 1, 'Hidden dim': 9, 'LR': 0.000428908,
                  'Dropout': 0}
    if data_type == 'volatility' and dt == "GSPC":
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Volatility/' + dt + '.csv',
                             end="2018-01-01",
                             split=0.8, cols=['rv5'], label_col='rv5', returns=False, MinMax=False)
        config = {'Timesteps': 26, 'Num. layers': 1, 'Hidden dim': 14, 'LR': 0.00012277033326270226,
                  'Dropout': 0}

    timesteps = int(config['Timesteps'])
    num_layers = int(config['Num. layers'])
    hidden_dim = int(config['Hidden dim'])
    lr = float(config['LR'])
    dropout = float(config['Dropout'])

    train_dt = dataset.get_train_data(timesteps, False, 1)
    test_dt = dataset.get_test_data(timesteps, False, 1)

    dataloader_params_train = {'batch_size': 1,
                               'shuffle': True,
                               'drop_last': True,
                               'num_workers': 0}
    # Parameters
    dataloader_params_test = {'batch_size': 1,
                              'shuffle': False,
                              'drop_last': True,
                              'num_workers': 0}

    # Generators
    training_set = Dataset(train_dt)
    training_generator = data.DataLoader(training_set, **dataloader_params_train)
    test_set = Dataset(test_dt)
    test_generator = data.DataLoader(test_set, **dataloader_params_test)

    network_params = {'input_dim': 1,  # As many as there are of columns in data
                      'hidden_dim': hidden_dim,
                      'batch_size': 1,  # From dataloader_parameters
                      'output_dim': 1,
                      'dropout': dropout,
                      'num_layers': num_layers
                      }

    epochs = 200

    model = Model(**network_params)
    loss = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    model, loss_vals_train, loss_vals_test, error = train_model(model, loss, optimiser, None,
                epochs, training_generator,
                test_generator, timesteps,
                1, folder, False)

    path_to_checkpoint = folder + '/checkpoint.pth.tar'
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(path_to_checkpoint)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(path_to_checkpoint,
                                map_location=lambda storage,
                                                    loc: storage)
    epoch_of_best_accuracy = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']

    return loss_vals_train, loss_vals_test, epoch_of_best_accuracy, best_accuracy
#run_one_step("DJI", "returns")
#run_one_step("GSPC", "returns")
#run_one_step("IXIC", "returns")
#run_one_step("N225", "returns")

#run_one_step("DJI", "volatility")
#run_one_step("GSPC", "volatility")