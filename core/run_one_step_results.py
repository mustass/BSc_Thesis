from utils.get_hyperparams_df import configs_df
import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.new_eval import *
from plots.new_plots import *
import pickle
from core.create_folder import *

for dt in ["N225"]:
    folder = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/returns/one_step/'+ dt
    dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'+dt+'.csv', start_from="2009-04-29",
                         split=None, cols=['log_ret'], label_col='log_ret', MinMax=False)
    config = configs_df.filter(like= str(dt), axis= 0)
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

    dates = dataset.dates[timesteps+2:].values
    true_labels = time_series[1]
    plot_and_save(predicted_labels,true_labels, dates,
                  title = "Nikkei 225",
                  subtitle="One-step without re-estimation", folder = folder)
    with open(folder + '/rmse_test_set.txt', 'w+') as f:
        f.write(str(RMSE))  # Python 3.x
    print("RMSE saved")
