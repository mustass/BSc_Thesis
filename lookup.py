import torch
import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.predict_sequence import *
import matplotlib.pyplot as plt

path_to_checkpoint = '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/HypOpt results 2/Run_w_15timesteps_4hiddenDim1layers' + '/' + 'checkpoint' + '.pth.tar'

cuda = torch.cuda.is_available()
if cuda:
    checkpoint = torch.load(path_to_checkpoint)
else:
    # Load GPU model on CPU
    checkpoint = torch.load(path_to_checkpoint,
                            map_location=lambda storage,
                                                loc: storage)

print(checkpoint['best_accuracy'])





def plot_results_multiple(predicted_data, true_data, prediction_len, folder):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.savefig(fname=folder + '/Seq_prediction_' + '.png')
    plt.show()


# detect the current working directory and print it
path = os.path.dirname(os.path.abspath(__file__))
print("The current working directory is %s" % path)




dataset = DataLoader('/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/DJI.csv', 0.80, ['Adj Close', 'Volume'],
                     'Adj Close', True)
timesteps = 15
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
                  'hidden_dim': 4,
                  'batch_size': dataloader_params_train['batch_size'],  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0,
                  'num_layers': 1
                  }

Nice_model = Model(**network_params)
Nice_loss = torch.nn.MSELoss()
Nice_model.load_state_dict(checkpoint['state_dict'])
ys, ys_testing, loss_vals_test, loss_vals_train = eval_model(Nice_model, Nice_loss, train_dt,
                                                             test_dt, timesteps)

plot_and_save(ys, ys_testing, train_dt[1], test_dt[1], None, None, loss_vals_train,
              loss_vals_test, False, Nice_model,
              '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/HypOpt results 2/Run_w_15timesteps_4hiddenDim1layers' + '/')
sequences = predict_seq_avg(Nice_model, dataset, timesteps, 20)
plot_results_multiple(sequences, test_dt[1], 20, '/home/s/Dropbox/KU/BSc Stas/Python/Try_again/HypOpt results 2/Run_w_15timesteps_4hiddenDim1layers' + '/')
