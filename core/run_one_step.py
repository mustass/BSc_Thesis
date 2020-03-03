from utils.get_hyperparams_df import configs_df
import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
import pickle
from core.create_folder import *

for dt in ["DJI"]:
    folder = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/returns/one_step/'+ dt
    dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/'+dt+'.csv', end="2009-04-29",
                         split=0.8, cols=['log_ret'], label_col='log_ret', MinMax=False)

    config = configs_df.filter(like= str(dt), axis= 0)
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

    last_model, loss_train_mtrx, loss_test_mtrx, error = train_model(model,loss, optimiser, None,
                                                                     epochs,training_generator,
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
    model.load_state_dict(checkpoint['state_dict'])
    ys, ys_testing, ys__denormalised, ys_testing_denormalised, loss_vals_test, loss_vals_train = eval_model(model,
                                                                                                            loss,
                                                                                                            dataset,
                                                                                                            timesteps,
                                                                                                            1)
    y_training = train_dt[1]
    y_testing = test_dt[1]
    plot_and_save(ys, ys_testing, y_training, y_testing, loss_train_mtrx, loss_test_mtrx,
                  loss_vals_train,
                  loss_vals_test, False, model,
                  folder)
    output = [loss_train_mtrx, loss_test_mtrx,ys, ys_testing, ys__denormalised, ys_testing_denormalised, loss_vals_test, loss_vals_train]
    file_output = open('/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/returns/one_step/'+ dt+'/output_file.pickle', 'wb')
    pickle.dump(output, file_output)