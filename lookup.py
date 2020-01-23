"""
File for evaluating and working with a previously trained model.
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

analysis = Analysis("~/ray_results/Jan21")
print(analysis.get_best_config(metric = "error", mode = "max"))
path_to_checkpoint = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/Run_w_11_timesteps_3_hiddenDim_1_layers_0.00021723989839730966_LR' + '/' + 'checkpoint' + '.pth.tar'

cuda = torch.cuda.is_available()
if cuda:
    checkpoint = torch.load(path_to_checkpoint)
else:
    # Load GPU model on CPU
    checkpoint = torch.load(path_to_checkpoint,
                            map_location=lambda storage,
                                                loc: storage)

print("Best accuracy, aka. lowest error is: "+str(checkpoint['best_accuracy']))


# detect the current working directory and print it
path = os.path.dirname(os.path.abspath(__file__))
print("The current working directory is %s" % path)




dataset = DataLoader('/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/DJI.csv', 0.80, ['Adj Close', 'Volume'],
                     'Adj Close', False)
timesteps = 11
train_dt = dataset.get_train_data(timesteps, True)
test_dt = dataset.get_test_data(timesteps, True)
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
                  'hidden_dim': 3,
                  'batch_size': dataloader_params_train['batch_size'],  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0,
                  'num_layers': 1
                  }

Nice_model = Model(**network_params)
Nice_loss = torch.nn.MSELoss()
Nice_model.load_state_dict(checkpoint['state_dict'])
ys, ys_testing, ys__denormalised, ys_testing_denormalised, loss_vals_test, loss_vals_train = eval_model(Nice_model, Nice_loss,
                                                              dataset,timesteps)
train_dt = dataset.get_train_data(timesteps, False)
test_dt = dataset.get_test_data(timesteps, False)
y_training = train_dt[1]
y_testing = test_dt[1]
plot_and_save(ys__denormalised, ys_testing_denormalised, y_training, y_testing, None, None,
              loss_vals_train,
              loss_vals_test, False, "Checkpoint_model",
              '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/Run_w_11_timesteps_3_hiddenDim_1_layers_0.00021723989839730966_LR' + '/')
sequences,sequences_denormalized = predict_seq_avg(Nice_model, dataset, timesteps, 25)

test_dt = dataset.get_test_data(timesteps, False)
y_testing = test_dt[1]
plot_results_multiple(sequences_denormalized, y_testing, 25,
                      '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/results/Run_w_11_timesteps_3_hiddenDim_1_layers_0.00021723989839730966_LR' + '/')