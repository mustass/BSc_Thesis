from utils.get_hyperparams_df import configs_df
from core.training import *
from core.new_eval import *

def results(dt,data_type = 'returns'):
    folder = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/'+data_type+'/one_step/'+ dt
    config = configs_df.filter(like=str(dt), axis=0)



    if data_type == 'returns' and dt == 'DJI':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'+dt+'.csv', start_from="2015-01-01",
                         split=None, cols=['log_ret'], label_col='log_ret', MinMax=False)
    if data_type == 'returns' and dt == 'GSPC':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'+dt+'.csv', start_from="2015-01-01",
                         split=None, cols=['log_ret'], label_col='log_ret', MinMax=False)
    if data_type == 'returns' and dt == 'N225':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'+dt+'.csv', start_from="2015-01-01",
                         split=None, cols=['log_ret'], label_col='log_ret', MinMax=False)
    if data_type == 'returns' and dt == 'IXIC':
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'+dt+'.csv', start_from="2015-01-01",
                         split=None, cols=['log_ret'], label_col='log_ret', MinMax=False)

    if data_type == 'volatility' and dt =="DJI":
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Volatility/' + dt + '.csv',
                             start_from="2018-01-01",
                             split=None, cols=['rv5'], label_col='rv5', returns=False, MinMax=False)
        config = {'Timesteps': 31, 'Num. layers': 1, 'Hidden dim': 9, 'LR': 0.000428908,
                  'Dropout': 0}
    if data_type == 'volatility' and dt =="GSPC":
        dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Volatility/' + dt + '.csv',
                             start_from="2018-01-01",
                             split=None, cols=['rv5'], label_col='rv5', returns=False, MinMax=False)
        config = {'Timesteps': 26, 'Num. layers': 1, 'Hidden dim': 14, 'LR': 0.00012277033326270226,
                  'Dropout': 0}

    timesteps = int(config['Timesteps'])
    num_layers = int(config['Num. layers'])
    hidden_dim = int(config['Hidden dim'])
    lr = float(config['LR'])
    dropout = float(config['Dropout'])

    time_series = dataset.get_train_data(timesteps, False, 1)

    dataloader_params_test = {'batch_size': 1,
                               'shuffle': False,
                               'drop_last': True,
                               'num_workers': 0}


    # Generators
    test_time_series = Dataset(time_series)
    test_set_generator = data.DataLoader(test_time_series, **dataloader_params_test)

    network_params = {'input_dim': 1,  # As many as there are of columns in data
                      'hidden_dim': hidden_dim,
                      'batch_size': 1,  # From dataloader_parameters
                      'output_dim': 1,
                      'dropout': dropout,
                      'num_layers': num_layers
                      }


    model = Model(**network_params)
    loss = torch.nn.MSELoss()


    path_to_checkpoint = folder + '/checkpoint.pth.tar'
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(path_to_checkpoint)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(path_to_checkpoint,
                                map_location=lambda storage,
                                                    loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    predicted_labels, loss_vals_test, RMSE = eval_model(model, loss, test_set_generator, timesteps)
    if data_type == "returns":
        dates = dataset.dates[timesteps+2:].values
    if data_type == "volatility":
        dates = dataset.dates[timesteps+1:].values
    true_labels = time_series[1]
    plot_df = pd.DataFrame({'preds':predicted_labels.reshape((-1)), 'true_vals':true_labels.reshape((-1)), 'dates':dates})
    print(RMSE)
    return plot_df, RMSE
