import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *


hidden = [1, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 17, 20, 25, 30,32,35]

for layers in range(2, 5, 1):
    for h in hidden:
        dataset = DataLoader('DJI.csv', 0.80, ['Adj Close', 'Volume'], 'Adj Close', True)
        timesteps = 50
        train_dt = dataset.get_train_data(timesteps)
        test_dt = dataset.get_test_data(timesteps)
        # Parameters
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
        network_params = {'input_dim': 2,  # As many as there are of columns in data
                          'hidden_dim': h,
                          'batch_size': dataloader_params_train['batch_size'],  # From dataloader_parameters
                          'output_dim': 1,
                          'dropout': 0,
                          'num_layers': layers
                          }
        epochs = 150
        Nice_model = Model(**network_params)
        Nice_loss = torch.nn.MSELoss()
        Nice_optimiser = torch.optim.Adam(Nice_model.parameters(), lr=0.005)
        Nice_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Nice_optimiser, epochs)
        Nice_model_trained, loss_train_mtrx, loss_test_mtrx = train_model(Nice_model, Nice_loss, Nice_optimiser,
                                                                          Nice_scheduler, epochs,
                                                                          training_generator,
                                                                          test_generator, timesteps,
                                                                          dataloader_params_train['batch_size'])
        ys, ys_testing, loss_vals_test, loss_vals_train = eval_model(Nice_model_trained, Nice_loss, train_dt,
                                                                     test_dt, timesteps)
        y_training = train_dt[1]
        y_testing = test_dt[1]
        folder_name = 'Test_with_' + str(layers) + '_batch_size_' + str(1) + '_hidden_dim_' + str(h)
        plot_and_save(ys, ys_testing, y_training, y_testing, loss_train_mtrx, loss_test_mtrx, loss_vals_train,
                      loss_vals_test, False, Nice_model_trained,
                      folder_name)