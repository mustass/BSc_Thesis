import pickle
from utils.get_hyperparams_df import configs_df
from core.rolling_window_train import rolling_window


def run_window_training(dt, restart, restart_window, window_length, col,returns = True, datatype='returns', start_from="2012-01-01"):
    if dt == "GSPC" and datatype =='volatility':
        config = {'Timesteps': 26, 'Num. layers': 1, 'Hidden dim': 14, 'LR': 0.00012277033326270226,
                  'Dropout': 0}
    if dt == "DJI" and datatype =='volatility':
        config = {'Timesteps': 31, 'Num. layers': 1, 'Hidden dim': 9, 'LR': 0.000428908,
                  'Dropout': 0}
    if datatype == 'returns':
        config = configs_df.filter(like=str(dt), axis=0)


    timesteps = int(config['Timesteps'])
    num_layers = int(config['Num. layers'])
    hidden_dim = int(config['Hidden dim'])
    lr = float(config['LR'])
    dropout = float(config['Dropout'])
    model_config = {'hidden_dim': hidden_dim,
                    'batch_size': 1,
                    'num_layers': num_layers,
                    'lr': lr,
                    'dropout': dropout}
    save_folder = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/' + datatype + '/rolling_' + str(
        window_length) + '/'
    data_folder = '/home/s/Dropbox/KU/BSc Stas/Python/Data/Volatility/'
    output = rolling_window(model_config=model_config, datasetName=dt, start_from=start_from,
                            restart=restart, restart_window=restart_window, window_length=window_length,
                            timesteps=timesteps,
                            max_epochs=50, data_folder=data_folder, save_folder=save_folder, col=col,returns=returns
                            )
    file_output = open(
        save_folder + dt + '/output_file.pickle', 'wb')
    pickle.dump(output, file_output)
    return output


# DJI_1 = run_window_training("DJI", True, 1652, 1)
# GSPC_1 = run_window_training("GSPC", True, 1654, 1)
# IXIC_1 = run_window_training("IXIC", True, 1651, 1)
# N225_1 = run_window_training("N225", True, 1656, 1)
# DJI_3 = run_window_training("DJI", True, 1146, 3)
# GSPC_3 = run_window_training("GSPC", True, 1148, 3)
# IXIC_3 = run_window_training("IXIC", True, 1145, 3)
# N225_3 = run_window_training("N225", True, 1150, 3)

DJI_1 = run_window_training(dt = "DJI", restart = True, restart_window = 1004, window_length= 1, datatype = "volatility", col='rv5', returns = False,start_from="2015-01-01")
DJI_3 = run_window_training(dt = "DJI", restart = True, restart_window = 498, window_length= 3, datatype = "volatility", col='rv5', returns = False,start_from="2015-01-01")

GSPC_1 = run_window_training(dt = "GSPC", restart = True, restart_window = 1011, window_length= 1, datatype = "volatility", col='rv5', returns = False,start_from="2015-01-01")
GSPC_3 = run_window_training(dt = "GSPC", restart = True, restart_window = 505, window_length= 3, datatype = "volatility", col='rv5', returns = False,start_from="2015-01-01")

