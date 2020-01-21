import matplotlib.pyplot as plt
from core.dataloader import *
from core.model import *
from core.training import *
from core.evaluating import *
from plots.plots import *
from core.create_folder import *
from core.pred_sequence import *
import matplotlib.pyplot as plt

# detect the current working directory and print it
path = os.path.dirname(os.path.abspath(__file__))
print("The current working directory is %s" % path)

h = 4
step = 15

dataset = DataLoader(path='/home/s/Dropbox/KU/BSc Stas/Python/Data/Daily/DJI.csv', split=0.80,
                     cols=['Adj Close', 'Volume'],
                     label_col='Adj Close', MinMax=False)
timesteps = step
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
                  'hidden_dim': h,
                  'batch_size': dataloader_params_train['batch_size'],  # From dataloader_parameters
                  'output_dim': 1,
                  'dropout': 0,
                  'num_layers': 1
                  }
epochs = 50

folder_name = 'Test_with_window_normalisation'
new_folder = create_folder(path + '/results', folder_name)
Nice_model = Model(**network_params)
Nice_loss = torch.nn.MSELoss()
Nice_optimiser = torch.optim.Adam(Nice_model.parameters(), lr=0.05)
Nice_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Nice_optimiser, epochs)
Nice_model_trained, loss_train_mtrx, loss_test_mtrx, error = train_model(Nice_model, Nice_loss, Nice_optimiser,
                                                                         Nice_scheduler, epochs,
                                                                         training_generator,
                                                                         test_generator, timesteps,
                                                                         dataloader_params_train['batch_size'],
                                                                         new_folder, False)
for model in ['last_model', 'checkpoint']:
    path_to_checkpoint = new_folder + '/' + model + '.pth.tar'
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(path_to_checkpoint)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(path_to_checkpoint,
                                map_location=lambda storage,
                                                    loc: storage)
    Nice_model.load_state_dict(checkpoint['state_dict'])
    ys, ys_testing, ys__denormalised, ys_testing_denormalised, loss_vals_test, loss_vals_train = eval_model(Nice_model,
                                                                                                            Nice_loss,
                                                                                                            train_dt,
                                                                                                            test_dt,
                                                                                                            dataset,
                                                                                                            timesteps)
    train_dt = dataset.get_train_data(timesteps, False)
    test_dt = dataset.get_test_data(timesteps, False)
    y_training = train_dt[1]
    y_testing = test_dt[1]
    plot_and_save(ys__denormalised, ys_testing_denormalised, y_training, y_testing, loss_train_mtrx, loss_test_mtrx,
                  loss_vals_train,
                  loss_vals_test, False, model,
                  new_folder)
sequences = predict_seq_avg(Nice_model, test_dt[0], timesteps, 15)
plot_results_multiple(sequences, y_testing, 15, new_folder)
